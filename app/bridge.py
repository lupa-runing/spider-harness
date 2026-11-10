"""Bridge：并发驱动 LangGraph 图，把 StreamPart 翻译成统一事件流并多路广播。

事件协议（与前端 apply(evt) 对齐）：
  task_created  {id, wf, name, channel}
  stage_started {id, agent, stage}
  stage_progress{id, agent, prog}
  stage_done    {id, agent, stage}
  review        {id, agent, fail, failIdx, score, checks}
  say           {agent, text}
  token_delta   {agent, in, out, cached}
  delivered     {id, agent, channel, cycle, tokens, rework}
  task_error    {id, error}
  snapshot      全量状态（首连/重连）
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from .config import settings
from .graph import graph
from .roles import AGENTS, PROOF, WF_BY_ID, WORKFLOWS


@dataclass
class TaskRecord:
    id: str
    requirement: str
    wf_id: str
    wf_index: int
    name: str
    channel: int
    path: list[int]
    stage: int = 0
    agent: int = -1          # 当前持有者
    prog: float = 0.0
    verdict: dict = field(default_factory=dict)
    rework: int = 0
    tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0, "cached": 0})
    born: float = field(default_factory=time.time)
    status: str = "running"  # running | done | failed | rejected
    cycle: float = 0.0
    req: dict | None = None  # NEBULA 需求评审：{sections:[{ai,words:[...]}], score, final}
    pkg: dict = field(default_factory=dict)      # W12 内容包：title/body/tags/cover/video/frames/proof
    ai_video: bool = False                       # 是否生成 MiniMax AI 视频
    files: list = field(default_factory=list)    # 内容包交付文件（W12）
    briefs: list = field(default_factory=list)   # CHIEF 各站任务书
    notes: list = field(default_factory=list)    # 逐站审读结论 [KEY] 一句话
    outputs: list = field(default_factory=list)  # 执行阶段各站产物 [KEY] 文本


class Harness:
    def __init__(self) -> None:
        self.tasks: dict[str, TaskRecord] = {}
        self.seq = 0
        self.sem = asyncio.Semaphore(settings.max_concurrent_tasks)
        self.subs: set[asyncio.Queue] = set()
        self.counters = {
            "total": 0, "passN": 0, "failN": 0,
            "tokDay": 0.0, "tokIn": 0.0, "tokOut": 0.0, "cacheHit": 0.62,
            "shipped": 0, "wordsToday": 0, "fixesToday": 0,   # NEBULA 今日派单/读词/补齐
        }
        self.agent_tok = [0.0] * 14
        self.reviews: list[dict] = []
        self._bg: set[asyncio.Task] = set()

    def _word(self, rec: TaskRecord, sec: int, wi: int) -> dict | None:
        try:
            return rec.req["sections"][sec]["words"][wi]
        except (TypeError, KeyError, IndexError):
            return None

    # ---------------- 广播 ----------------
    def publish(self, event: str, data: dict) -> None:
        for q in list(self.subs):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait((event, data))

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self.subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.discard(q)

    # ---------------- 提交任务 ----------------
    def submit(self, requirement: str, wf_id: str | None = None, ai_video: bool = False) -> TaskRecord:
        self.seq += 1
        wf = WF_BY_ID[wf_id] if wf_id in WF_BY_ID else None
        if wf is None:
            wf = self._pick_wf(requirement)
        rec = TaskRecord(
            id=f"T-{self.seq:03d}", requirement=requirement, wf_id=wf["id"],
            wf_index=WORKFLOWS.index(wf), name=wf["names"][0], channel=wf["ch"], path=list(wf["path"]),
            ai_video=ai_video,
        )
        self.tasks[rec.id] = rec
        self.publish("task_created", {"id": rec.id, "wf": rec.wf_index, "wfId": rec.wf_id, "name": rec.name, "channel": rec.channel})
        t = asyncio.create_task(self._run(rec))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return rec

    def _pick_wf(self, requirement: str) -> dict:
        import hashlib
        h = int(hashlib.md5(requirement.encode()).hexdigest(), 16)
        return WORKFLOWS[h % len(WORKFLOWS)]

    # ---------------- 图执行与事件翻译 ----------------
    async def _run(self, rec: TaskRecord) -> None:
        async with self.sem:
            init = {
                "task_id": rec.id, "requirement": rec.requirement, "wf_id": rec.wf_id,
                "name": rec.name, "channel": rec.channel, "path": rec.path,
                "stage": 0, "briefs": [], "sections": [], "revision": 0, "revision_notes": [], "package": {},
                "ai_video": rec.ai_video,
                "dims": [], "final_score": 0, "rejected": False,
                "artifacts": [], "usage": [],
                "verdict": {}, "rework": 0, "version": 1, "born": rec.born, "done": False,
            }
            try:
                async for part in graph.astream(init, stream_mode=["updates", "custom"], subgraphs=False):
                    mode = part[0]
                    payload = part[1]
                    if mode == "custom":
                        self._on_custom(rec, payload)
                    elif mode == "updates":
                        self._on_updates(rec, payload)
            except Exception as e:  # 图执行异常：大屏不能冻住
                import traceback
                traceback.print_exc()
                rec.status = "failed"
                self.publish("task_error", {"id": rec.id, "error": str(e)[:200]})

    def _on_custom(self, rec: TaskRecord, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        if "req_plan" in payload:
            plan = payload["req_plan"]
            rec.req = {"sections": [{"ai": s["ai"],
                                     "words": [{**w, "state": "idle"} for w in s["words"]],
                                     "done": False} for s in plan["sections"]],
                       "score": 0, "final": 0, "name": plan["name"]}
            rec.name = plan["name"]
            self.publish("req_plan", {"id": rec.id, "wf": rec.wf_index, "path": rec.path,
                                      "name": plan["name"], "sections": rec.req["sections"]})
        elif "word_read" in payload:
            w = payload["word_read"]
            word = self._word(rec, w["sec"], w["wi"])
            if word:
                word["state"] = "read"
            self.counters["wordsToday"] += 1
            self.publish("word_read", {"id": rec.id, **w})
        elif "word_flag" in payload:
            w = payload["word_flag"]
            word = self._word(rec, w["sec"], w["wi"])
            if word:
                word["state"] = "flag"
                word["kind"] = "vague"
            self.publish("word_flag", {"id": rec.id, **w})
        elif "word_ask" in payload:
            w = payload["word_ask"]
            word = self._word(rec, w["sec"], w["wi"])
            if word:
                word["state"] = "ask"
            self.publish("word_ask", {"id": rec.id, **w})
        elif "word_fix" in payload:
            w = payload["word_fix"]
            word = self._word(rec, w["sec"], w["wi"])
            if word:
                word["state"] = "fix"
                word["fix"] = w["fix"]
            self.counters["fixesToday"] += 1
            self.publish("word_fix", {"id": rec.id, **w})
        elif "req_score" in payload:
            s = payload["req_score"]
            if rec.req is not None:
                rec.req["score"] = s["score"]
                rec.req["dims"] = s.get("dims") or rec.req.get("dims")
            self.publish("req_score", {"id": rec.id, **s})
        elif "req_dispatched" in payload:
            s = payload["req_dispatched"]
            if rec.req is not None:
                rec.req["final"] = s["score"]
            self.counters["shipped"] += 1
            # 过线派单：进入执行流水线，向 SPIDER 视图宣布第一站开工
            rec.stage = 0
            rec.agent = rec.path[0]
            rec.prog = 0.0
            self.publish("req_dispatched", {"id": rec.id, **s})
            self.publish("stage_started", {"id": rec.id, "agent": rec.agent, "stage": 0})
        elif "req_rejected" in payload:
            s = payload["req_rejected"]
            rec.status = "rejected"
            if rec.req is not None:
                rec.req["final"] = s["score"]
            self.publish("req_rejected", {"id": rec.id, **s})
        elif "pkg" in payload:
            p = payload["pkg"]
            rec.pkg = p.get("pkg", p) if isinstance(p, dict) else {}
        elif "files" in payload:
            rec.files = [str(x) for x in payload["files"].get("files", [])]
            self.publish("files", {"id": rec.id, "files": rec.files})
        elif "agent" in payload and "prog" in payload:
            rec.agent = payload["agent"]; rec.prog = float(payload["prog"])
            self.publish("stage_progress", {"id": rec.id, "agent": payload["agent"], "prog": rec.prog})
        elif "agent" in payload and "say" in payload:
            self.publish("say", {"agent": payload["agent"], "text": payload["say"]})
        elif "usage" in payload:
            u = payload["usage"]; a = u["agent"]
            self.agent_tok[a] += u["in"] + u["out"]
            self.counters["tokDay"] += u["in"] + u["out"]
            self.counters["tokIn"] += u["in"]
            self.counters["tokOut"] += u["out"]
            if u["in"] > 0:
                hit = u["cached"] / u["in"]
                self.counters["cacheHit"] = round(self.counters["cacheHit"] * 0.7 + hit * 0.3, 3)
            rec.tokens["in"] += u["in"]; rec.tokens["out"] += u["out"]; rec.tokens["cached"] += u["cached"]
            self.publish("token_delta", {"agent": a, "in": u["in"], "out": u["out"], "cached": u["cached"]})
        elif "verdict" in payload:
            v = payload["verdict"]
            rec.verdict = v
            if v.get("fail"):
                self.counters["failN"] += 1
            else:
                self.counters["passN"] += 1
            self.reviews.append({"id": rec.id, "name": rec.name, "ok": not v.get("fail"),
                                 "score": v.get("score", 0), "col": "#ff9a1f", "dur": round(time.time() - rec.born, 1)})
            self.reviews = self.reviews[-30:]
            self.publish("review", {"id": rec.id, "agent": PROOF, **v})
        elif "delivered" in payload:
            d = payload["delivered"]
            rec.status = "done"; rec.cycle = d["cycle"]
            self.counters["total"] += 1
            self.publish("delivered", {"id": rec.id, "channel": d["channel"], "cycle": d["cycle"],
                                       "tokens": d["tokens"], "rework": d["rework"]})

    def _on_updates(self, rec: TaskRecord, updates: dict) -> None:
        # updates: {node_name: state_delta}；rev_xxx=需求评审节点，其余角色节点名即 agent key 小写
        for node, delta in updates.items():
            if not isinstance(delta, dict):
                continue
            if node == "chief":
                if delta.get("briefs"):
                    rec.briefs = list(delta["briefs"])
                if delta.get("name") and delta["name"] != rec.name:
                    rec.name = delta["name"]
                    self.publish("task_renamed", {"id": rec.id, "name": rec.name})
                # 执行流水线的 stage_started 由 req_dispatched 宣布（过线才开工）
                if rec.req is not None and rec.path and rec.req.get("sections"):
                    self.publish("rev_station", {"id": rec.id, "agent": rec.path[0], "stage": 0})
                continue
            if node == "deliver":
                continue
            if node.startswith("rev_"):
                agent = _agent_of(node[4:])
                if agent < 0:
                    continue
                for a in delta.get("artifacts") or []:
                    if a:
                        rec.notes.append(a)
                self.publish("rev_station_done", {"id": rec.id, "agent": agent, "stage": rec.stage})
                nxt = delta.get("stage")
                if nxt is not None and 0 <= nxt < len(rec.path):
                    rec.stage = nxt
                    self.publish("rev_station", {"id": rec.id, "agent": rec.path[nxt], "stage": nxt})
                continue
            agent = _agent_of(node)
            if agent < 0:
                continue
            for a in delta.get("artifacts") or []:
                if a:
                    rec.outputs.append(a)
            # 节点刚完工：rec.stage 就是它执行的站
            self.publish("stage_done", {"id": rec.id, "agent": agent, "stage": rec.stage})
            nxt = delta.get("stage")
            if nxt is None:
                continue
            rec.stage = nxt
            if 0 <= nxt < len(rec.path):
                rec.agent = rec.path[nxt]
                rec.prog = 0.0
                self.publish("stage_started", {"id": rec.id, "agent": rec.agent, "stage": nxt})


def _agent_of(node: str) -> int:
    for a in AGENTS:
        if a["k"].lower() == node:
            return a["i"]
    return -1


# ---------------- 快照 ----------------
def snapshot(h: Harness) -> dict:
    active = next((t for t in h.tasks.values() if t.status == "running" and t.req), None)
    return {
        "provider": settings.llm_provider,
        "counters": h.counters,
        "agents": [{"i": a["i"], "tok": h.agent_tok[a["i"]]} for a in AGENTS],
        "tasks": [
            {"id": t.id, "wf": t.wf_index, "wfId": t.wf_id, "name": t.name, "channel": t.channel,
             "stage": t.stage, "agent": t.agent, "prog": t.prog, "rework": t.rework,
             "status": t.status, "born": t.born, "cycle": t.cycle,
             "tokens": t.tokens, "verdict": t.verdict}
            for t in h.tasks.values()
        ],
        "reviews": h.reviews,
        # NEBULA 需求评审快照：当前在审的需求 + 词表状态
        "req": ({"id": active.id, "wf": active.wf_index, "name": active.name, "path": active.path, "stage": active.stage,
                 "sections": active.req["sections"], "score": active.req["score"],
                 "final": active.req["final"]} if active else None),
    }


harness = Harness()
