"""SPIDER 流水线图：LangGraph StateGraph。

图结构（节点名 = 角色小写，与前端 AG 下标一一对应）：
  START → chief → (按 wf.path) scout/split/copy/design/code/test/proof/data/cut/ads/care/vault → deliver → END
PROOF 节点产出六项检查裁决；不过且打回次数 <2 → Command(goto=上一站)，过 → 下一站或 deliver。
每个节点用 get_stream_writer() 发 progress / say / usage / verdict 自定义事件，bridge 翻译成大屏 SSE。
"""
from __future__ import annotations

import asyncio
import json
import operator
import random
import time
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .provider import LLMResult, ROLE_TOKEN_BUDGET, Usage, complete
from .config import settings
from . import tools
from .imagelib import generate as liblib_gen
from .roles import (AGENTS, AGENT_WORDS, AXES, CHECK_NAMES, KEY_OF, PASS_SCORE, PROOF,
                    PROOF_JSON_INSTRUCTION, ROLE_PROMPTS, VAGUE, WORKER_INDICES, WORKFLOWS,
                    XHS_JSON_INSTRUCTIONS)


def _det_seed(*parts: str) -> random.Random:
    import hashlib
    return random.Random(int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16))


class TaskState(TypedDict):
    task_id: str
    requirement: str
    wf_id: str
    name: str
    channel: int
    path: list[int]
    stage: int
    briefs: Annotated[list[str], operator.add]   # CHIEF 写的各站任务书
    sections: list                               # 需求评审的词表计划 [{ai, words:[{w,kind,fix?,question?}]}]
    revision: int                                # 需求评审修订轮数
    revision_notes: Annotated[list[str], operator.add]   # CHIEF 修订补充（负责人/截止/验收等结构性约定）
    package: dict                                # W12 内容包：title/body/tags/cover/video/proof…
    dims: list                                   # 七维评分
    final_score: int
    rejected: bool
    artifacts: Annotated[list[str], operator.add]
    usage: Annotated[list[dict], operator.add]   # {agent, in, out, cached}
    verdict: dict
    rework: int
    version: int
    born: float
    done: bool


def node_for(agent_idx: int) -> str:
    return KEY_OF[agent_idx].lower()


def rev_node_for(agent_idx: int) -> str:
    return "rev_" + KEY_OF[agent_idx].lower()


def _plan_sections(path: list[int], requirement: str) -> list[dict]:
    """CHIEF 把需求按工位切成章节并排词表：每站 9–12 个工位词，插入 3–5 个模糊词（VAGUE 表）。
    词表本身是展示元素（与前端 AG.words 同源）；模糊与否的真裁决在逐站审词时给出。"""
    rnd = _det_seed("plan", requirement[:48], "|".join(map(str, path)))
    n_vague = rnd.randint(3, 5)
    pairs = VAGUE[:]
    rnd.shuffle(pairs)
    by_sec: dict[int, list] = {}
    for i in range(n_vague):
        by_sec.setdefault(rnd.randrange(len(path)), []).append(pairs[i])
    sections = []
    for k, ai in enumerate(path):
        vocab = AGENT_WORDS.get(ai, [])
        words = [{"w": w, "kind": "ok"} for w in rnd.sample(vocab, min(rnd.randint(9, 12), len(vocab)))]
        for v in by_sec.get(k, []):
            words.insert(rnd.randrange(len(words) + 1), {"w": v[0], "kind": "vague", "fix": v[1], "question": f"「{v[0]}」具体指什么？"})
        sections.append({"ai": ai, "words": words, "done": False})
    return sections


# ---------------------------------------------------------------- chief
async def chief_node(state: TaskState) -> dict:
    writer = get_stream_writer()
    from .roles import WF_BY_ID
    wf = WF_BY_ID[state["wf_id"]]
    req = state["requirement"]

    system = ROLE_PROMPTS["CHIEF"]
    prompt = (
        f"需求：{req}\n已指定工作流 {wf['id']}({wf['zh']})。"
        f"为 path={'→'.join(KEY_OF[i] for i in wf['path'])} 的每一站写一句话任务书，"
        f'并起任务名。输出 JSON：{{"name": str, "briefs": [按 path 顺序,每站一句话]}}'
    )
    try:
        result = await complete("CHIEF", system, prompt, _throttled_writer(writer, agent=0))
    except Exception:
        if writer:
            writer({"agent": 0, "say": "CHIEF 调用异常，按工作流模板直接拆解"})
        result = LLMResult()
    name, briefs = _parse_chief(result.text, wf, req)

    sections = _plan_sections(wf["path"], req)
    if writer:
        writer({"req_plan": {"task_id": state["task_id"], "name": name, "path": list(wf["path"]), "sections": sections}})
        writer({"agent": 0, "say": f"需求受理，拆成 {len(wf['path'])} 站审读"})
    return {"name": name, "briefs": briefs, "sections": sections}


def _throttled_writer(writer, agent: int, interval: float = 0.5):
    """把 provider 的 chunk 回调转成限频 progress 自定义事件（分母用该角色的 token 预算）。"""
    state = {"out": 0, "last": 0.0}

    async def on_event(kind: str, payload) -> None:
        if not writer:
            return
        if kind == "chunk":
            _, out_delta = payload
            state["out"] += out_delta
            now = asyncio.get_running_loop().time()
            if now - state["last"] >= interval:
                state["last"] = now
                budget = ROLE_TOKEN_BUDGET.get(KEY_OF[agent], (1000, 800))[1]
                writer({"agent": agent, "prog": min(0.95, state["out"] / max(1, budget))})
    return on_event


def _parse_chief(text: str, wf: dict, req: str) -> tuple[str, list[str]]:
    name, briefs = None, None
    try:
        s = text[text.index("{"): text.rindex("}") + 1]
        data = json.loads(s)
        name = str(data.get("name") or "").strip() or None
        b = data.get("briefs")
        if isinstance(b, list) and len(b) == len(wf["path"]):
            briefs = [str(x) for x in b]
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    rnd = _det_seed("chief", wf["id"], req)
    if not name:
        name = rnd.choice(wf["names"])
    if not briefs:
        briefs = [f"{KEY_OF[i]} 站：围绕「{req[:48]}」执行{wf['zh']}工序第 {n + 1} 步" for n, i in enumerate(wf["path"])]
    return name, briefs


# ---------------------------------------------------------------- 角色节点
def make_role_node(agent_idx: int):
    key = KEY_OF[agent_idx]

    async def role_node(state: TaskState) -> dict:
        writer = get_stream_writer()
        path = state["path"]
        stage = path.index(agent_idx)          # 自己的站号由 path 推导（dispatch 会把 stage 置为 len(path)）
        brief = (state["briefs"] or [""] * len(path))[stage] if stage < len(state["briefs"] or []) else ""
        upstream = state["artifacts"][-1] if state["artifacts"] else "（无上游产物）"
        system = ROLE_PROMPTS[key]
        if state.get("wf_id") == "W12" and key in XHS_JSON_INSTRUCTIONS:
            system = f"{system} {XHS_JSON_INSTRUCTIONS[key]}"   # 注入结构化产出指令（JSON/英文 imagePrompt）
        prompt = f"任务：{state['name']}（{state['requirement']}）\n本站任务书：{brief}\n上游产物摘要：{upstream[:600]}"

        if writer:
            writer({"agent": agent_idx, "say": f"接 {state['task_id']} · {brief[:24]}"})
        try:
            result = await complete(key, system, prompt, _throttled_writer(writer, agent=agent_idx))
        except Exception:
            if writer:
                writer({"agent": agent_idx, "say": f"{key} 调用异常，本站交付占位产物"})
            result = LLMResult(text="[调用异常占位] 本站产物待 LLM 恢复后重试补充。")
        u = result.usage

        # W12 小红书流水线：工位真实产出（写文案 / 画封面 / 合成视频 / 审校 / 打包）
        pkg, xhs_art = await _xhs_build(key, state, result, writer)
        if pkg is not None and writer:
            writer({"pkg": {"task_id": state["task_id"], "pkg": pkg}})

        if writer and u.output_tokens:
            writer({"usage": {"agent": agent_idx, "in": u.input_tokens, "out": u.output_tokens, "cached": u.cached_tokens}})
            writer({"agent": agent_idx, "prog": 1.0})

        update: dict = {
            "artifacts": [xhs_art or f"[{key}] {result.text[:400]}"],
            "usage": [{"agent": agent_idx, "in": u.input_tokens, "out": u.output_tokens, "cached": u.cached_tokens}],
        }
        if pkg is not None:
            update["package"] = pkg

        next_stage = stage + 1
        goto = node_for(path[next_stage]) if next_stage < len(path) else "deliver"
        if agent_idx == PROOF and state.get("wf_id") != "W12":
            verdict = _proof_verdict(state, result)
            update["verdict"] = verdict
            if writer:
                writer({"verdict": {"task_id": state["task_id"], **verdict}})
            if verdict["fail"] and state["rework"] < 2:
                update["stage"] = stage - 1
                update["rework"] = state["rework"] + 1
                update["version"] = state["version"] + 1
                return Command(update=update, goto=node_for(path[stage - 1]))
        update["stage"] = next_stage
        return Command(update=update, goto=goto)

    role_node.__name__ = f"role_{key.lower()}"
    return role_node


def _proof_verdict(state: TaskState, result: LLMResult) -> dict:
    """优先解析 PROOF 的 JSON 裁决；解析失败用确定性占位。打回上限 2 次，到顶强制放行。"""
    rnd = _det_seed("proof", state["task_id"], str(state["rework"]), state["requirement"][:32])
    fail = rnd.random() < 0.14 and state["rework"] < 2
    fail_idx = rnd.randrange(len(CHECK_NAMES)) if fail else 0
    score = round(rnd.uniform(0.58, 0.76) if fail else rnd.uniform(0.82, 0.97), 2)

    verdict = None
    try:
        s = result.text[result.text.index("{"): result.text.rindex("}") + 1]
        data = json.loads(s)
        if isinstance(data.get("fail"), bool):
            verdict = {
                "fail": data["fail"] and state["rework"] < 2,
                "failIdx": int(data.get("failIdx", 0)) % len(CHECK_NAMES),
                "score": float(data.get("score", score)),
                "checks": data.get("checks") or [],
            }
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    if verdict is None:
        checks = []
        for j, n in enumerate(CHECK_NAMES):
            ok = not (fail and j == fail_idx)
            checks.append({"n": n, "ok": ok, "detail": "通过" if ok else "不达标，需返工"})
        verdict = {"fail": fail, "failIdx": fail_idx, "score": score, "checks": checks}
    return verdict


# ---------------------------------------------------------------- 需求评审（NEBULA 星核蜘蛛）
async def _review_verdict(key: str, state: dict, sec: dict, on_event) -> dict:
    """逐站审词裁决：真实 LLM 判断哪些词模糊、如何补齐；mock 沿用计划里预插的模糊词。"""
    if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
        ws = "、".join(w["w"] for w in sec["words"])
        sys = ROLE_PROMPTS[key] + " 你同时在审需求：从本站词表里挑出与需求放在一起含义模糊、无法直接执行的词。严格克制：只标记确实无法执行的词（通常 0–3 处，最多不超过 4 处），其余视为清晰可执行。"
        prompt = (f"需求：{state['name']}（{state['requirement']}）\n本站词表：{ws}\n"
                  '输出严格 JSON：{"vague": [{"w": 词, "fix": "补成什么"}], "note": "一句话审读结论"}')
        try:
            r = await complete(key, sys, prompt, on_event)
            s = r.text[r.text.index("{"): r.text.rindex("}") + 1]
            data = json.loads(s)
            if isinstance(data.get("vague"), list):
                return {"vague": data["vague"], "note": str(data.get("note", "")), "usage": r.usage}
        except Exception:
            pass
    planned = [w for w in sec["words"] if w.get("kind") == "vague"]
    return {"vague": [{"w": w["w"], "fix": w.get("fix", "按工位标准明确")} for w in planned],
            "note": f"{key} 审读 {len(sec['words'])} 词，{len(planned)} 处待补齐", "usage": None}


def make_review_node(agent_idx: int):
    key = KEY_OF[agent_idx]

    async def review_node(state: TaskState) -> dict:
        writer = get_stream_writer()
        path = state["path"]
        sec_idx = path.index(agent_idx)
        sec = state["sections"][sec_idx]

        async def on_event(kind: str, payload) -> None:
            if writer and kind == "usage":
                u = payload
                writer({"usage": {"agent": agent_idx, "in": u.input_tokens, "out": u.output_tokens, "cached": u.cached_tokens}})

        verdict = await _review_verdict(key, state, sec, on_event)
        u = verdict.get("usage")
        if writer and u is None:
            i, o = ROLE_TOKEN_BUDGET.get(key, (1000, 800))
            writer({"usage": {"agent": agent_idx, "in": i, "out": o, "cached": int(i * .6)}})

        # 合并裁决：裁决为模糊的词标 vague + 补齐文本；计划预插但裁决否掉的退回 ok
        words = [dict(w) for w in sec["words"]]
        vmap = {v.get("w"): v.get("fix", "按工位标准明确") for v in verdict.get("vague", []) if isinstance(v, dict) and v.get("w")}
        for w in words:
            if w["w"] in vmap:
                w["kind"] = "vague"
                w["fix"] = vmap[w["w"]]
                w.setdefault("question", f"「{w['w']}」具体指什么？")
            elif w.get("kind") == "vague":
                w["kind"] = "ok"
                w.pop("fix", None)
                w.pop("question", None)

        # 逐词 paced 发事件：前端星云蛛边走边读；模糊词 flag → 问 CHIEF → 补齐
        if writer:
            await asyncio.sleep(1.7)                     # 等星云蛛跳到本站
            for wi, w in enumerate(words):
                writer({"word_read": {"task_id": state["task_id"], "sec": sec_idx, "wi": wi}})
                if w["kind"] == "vague":
                    await asyncio.sleep(.15)
                    writer({"word_flag": {"task_id": state["task_id"], "sec": sec_idx, "wi": wi,
                                          "question": w.get("question", f"「{w['w']}」具体指什么？")}})
                    await asyncio.sleep(.25)
                    writer({"word_ask": {"task_id": state["task_id"], "sec": sec_idx, "wi": wi}})
                    await asyncio.sleep(.55)
                    writer({"word_fix": {"task_id": state["task_id"], "sec": sec_idx, "wi": wi, "fix": w["fix"]}})
                    await asyncio.sleep(.15)
                else:
                    await asyncio.sleep(.13)

        new_sections = list(state["sections"])
        new_sections[sec_idx] = {**sec, "words": words, "done": True}
        nxt = sec_idx + 1
        usage = [{"agent": agent_idx, "in": u.input_tokens if u else ROLE_TOKEN_BUDGET.get(key, (1000, 800))[0],
                  "out": u.output_tokens if u else ROLE_TOKEN_BUDGET.get(key, (1000, 800))[1],
                  "cached": u.cached_tokens if u else int(ROLE_TOKEN_BUDGET.get(key, (1000, 800))[0] * .6)}]
        update = {"sections": new_sections, "stage": nxt,
                  "artifacts": [f"[{key}] {verdict.get('note', '')}"], "usage": usage}
        goto = rev_node_for(path[nxt]) if nxt < len(path) else "score"
        return Command(update=update, goto=goto)

    review_node.__name__ = f"review_{key.lower()}"
    return review_node


def _json_of(text: str) -> dict:
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError, TypeError):
        return {}


def _emit_files(writer, tid: str, files: list[str]) -> None:
    if writer and files:
        writer({"files": {"task_id": tid, "files": files}})


async def _xhs_build(key: str, state: dict, result: LLMResult, writer) -> tuple[dict | None, str | None]:
    """W12 小红书爆款：工位的真实产出动作——LLM 出创意决策，tools 出实体文件。
    返回 (package 更新, 友好产物摘要)。LLM 挂了走确定性兜底，保证内容包一定生成。"""
    if state.get("wf_id") != "W12" or key not in ("COPY", "DESIGN", "CUT", "PROOF", "VAULT"):
        return None, None
    tid = state["task_id"]
    pkg = dict(state.get("package") or {})
    data = _json_of(result.text) if result.text else {}
    accent = tuple(pkg.get("accent") or (255, 79, 121))

    if key == "COPY":
        title = str(data.get("title") or f"{state['name']}｜亲测分享 ✅")
        if len(title) > 20:
            title = title[:18].rstrip("（(·、，, ") + "…"
        body = str(data.get("body") or f"{state['requirement']}\n\n真实体验分享，细节看图，有问题评论区问我～")
        tags = [str(t) for t in (data.get("tags") or ["#种草", "#好物分享", "#新手必看"])][:6]
        hook = str(data.get("hook") or f"这条视频讲清楚「{state['name']}」，看完再决定。")
        beats = data.get("beats") or [{"say": hook, "os": state["name"][:12]}]
        pkg.update({"title": title, "body": body, "tags": tags, "hook": hook, "beats": beats})
        md = (f"# {title}\n\n{body}\n\n{' '.join(tags)}\n\n---\n"
              f"> 视频口播钩子：{hook}\n> 分镜口播：{' / '.join(str(b.get('say', '')) for b in beats)}")
        f = tools.write_file(tid, "文案.md", md)
        pkg["copy_file"] = f
        if writer:
            writer({"say": {"agent": 3, "text": f"文案落盘 · {title}"}})
            _emit_files(writer, tid, [f])
        return pkg, f"[COPY] 文案.md 已生成 · 标题「{title}」· 正文 {len(body)} 字 · {len(tags)} 个标签"

    if key == "DESIGN":
        title = str(data.get("title") or pkg.get("title", state["name"]))[:12]
        badge = str(data.get("badge") or "实测")
        lines = [str(x) for x in (data.get("lines") or ["真实体验", "细节看图"])]
        try:
            accent = tuple(int(c) for c in (data.get("accent") or accent))
        except (TypeError, ValueError):
            pass
        cover, used_ai = None, False
        ip = str(data.get("imagePrompt") or "").strip() or \
            f"xiaohongshu cover style product photography of {state['requirement'][:60]}, bright aesthetic, soft daylight, clean composition, no text"
        if settings.liblib_access_key and settings.liblib_secret_key:
            try:
                if writer:
                    writer({"say": {"agent": 4, "text": "调用 LiblibAI 生成封面…"}})
                raw = tools.task_dir(tid) / "_cover_raw.jpg"
                cover = await liblib_gen(ip, raw, settings.liblib_access_key, settings.liblib_secret_key, w=1080, h=1440)
                used_ai = True
            except Exception:
                cover = None
        out_cover = tools.task_dir(tid) / "封面_3x4.jpg"
        if cover is not None:
            tools.overlay_card(cover, out_cover, title, lines[:3], w=1080, h=1440, accent=accent, badge=badge)
            raw.unlink(missing_ok=True)
        else:
            tools.text_card(out_cover, title, lines[:3], w=1080, h=1440, accent=accent, badge=badge)
        pkg.update({"cover": tools.rel(out_cover), "accent": list(accent)})
        if writer:
            writer({"say": {"agent": 4, "text": "封面已出 · " + title}})
            _emit_files(writer, tid, [pkg["cover"]])
        return pkg, f"[DESIGN] 封面 {pkg['cover']} 已生成（{'AI 图' if used_ai else '文字卡'}，主色 {accent}）"

    if key == "CUT":
        if writer:
            writer({"say": {"agent": 9, "text": "开始逐段配音渲染…"}})
        segs = data.get("segments") or pkg.get("beats") or []
        if len(segs) < 3:
            # 兜底：正文句子拆分镜并入，保证成片至少 3 段
            import re as _re
            sents = [s for s in _re.split(r"[。！？\n]+", pkg.get("body", "")) if len(s) >= 6][:5]
            segs = (segs + [{"say": s[:60], "os": s[:12]} for s in sents])[:6]
        segs = segs[:6] or [{"say": pkg.get("hook", state["requirement"][:30]), "os": state["name"][:12]}]
        acc = tuple(pkg.get("accent") or accent)
        parts, ai_n = [], 0
        for i, sg in enumerate(segs):
            say = str(sg.get("say") or "")[:80] or f"第{i + 1}段"
            os_t = str(sg.get("os") or "")[:14] or f"{i + 1}"
            lines = [str(x)[:18] for x in (sg.get("lines") or [])][:2]
            ip = str(sg.get("imgPrompt") or "").strip()
            img = None
            raw = tools.task_dir(tid) / f"_raw{i}.jpg"
            if ip and settings.liblib_access_key and settings.liblib_secret_key:
                try:
                    img = await liblib_gen(ip, raw, settings.liblib_access_key, settings.liblib_secret_key, w=1080, h=1920)
                    ai_n += 1
                except Exception:
                    img = None
            frame_out = tools.task_dir(tid) / f"分镜_{i + 1}.jpg"
            if img is not None:
                tools.overlay_card(img, frame_out, os_t, lines, w=1080, h=1920, accent=acc, badge=f"{i + 1}/{len(segs)}")
            else:
                tools.text_card(frame_out, os_t, lines, w=1080, h=1920, accent=acc, badge=f"{i + 1}/{len(segs)}")
            raw.unlink(missing_ok=True)
            audio = await tools.tts(say, tools.task_dir(tid) / f"_a{i}.mp3")
            parts.append({"img": frame_out, "audio": audio, "text": say})
        rendered = await tools.render_video(tid, parts)
        pkg.update({"video": rendered["video"], "srt": rendered["srt"], "dur": rendered["dur"],
                    "frames": [tools.rel(p["img"]) for p in parts]})
        if writer:
            writer({"say": {"agent": 9, "text": f"成片合成 · {rendered['dur']} 秒"}})
            _emit_files(writer, tid, [rendered["video"], rendered["srt"]])
        return pkg, f"[CUT] video.mp4 已合成（{len(parts)} 段 · {rendered['dur']} 秒，AI 图 {ai_n}/{len(parts)}）+ 字幕.srt"

    if key == "PROOF":
        title, body, tags = pkg.get("title", ""), pkg.get("body", ""), pkg.get("tags", [])
        issues = []
        if len(title) > 20:
            issues.append("标题超 20 字")
        if len(body) < 100:
            issues.append("正文偏短（<100 字）")
        if len(tags) < 4:
            issues.append("标签少于 4 个")
        banned = ["绝对", "100%", "国家级", "全网最低", "全网第一", "销量第一", "行业第一", "排名第一", "史上最"]
        hit = [w for w in banned if w in (title + body)]
        if hit:
            issues.append("疑似极限词/绝对化用语：" + "、".join(hit))
        # 打回重写环：有问题就让 COPY 重写一版并重新落盘，直到本地校验通过
        if issues:
            try:
                repair_prompt = (
                    f"小红书内容审校发现以下问题，请重写修复：{'；'.join(issues)}\n"
                    f"原标题：{title}\n原正文：{body}\n原标签：{tags}\n"
                    "注意：严禁使用『最、第一、绝对、100%』等广告法极限词，用『很、超、闭眼冲』等替代表达。\n"
                    '输出严格 JSON：{"title": "≤20 字标题", "body": "300-400 字正文，口语化种草体", "tags": [5 个标签]}'
                )
                r2 = await complete("COPY", ROLE_PROMPTS["COPY"], repair_prompt, None)
                d2 = _json_of(r2.text)
                if d2.get("title") and d2.get("body"):
                    title = str(d2["title"])[:20]
                    body = str(d2["body"])
                    tags = [str(t) for t in (d2.get("tags") or tags)][:6]
                    issues = []
                    if len(title) > 20:
                        issues.append("标题超 20 字")
                    if len(body) < 100:
                        issues.append("正文偏短")
                    if len(tags) < 4:
                        issues.append("标签少于 4 个")
                    hit = [w for w in banned if w in (title + body)]
                    if hit:
                        issues.append("疑似极限词：" + "、".join(hit))
                    md = (f"# {title}\n\n{body}\n\n{' '.join(tags)}\n\n---\n"
                          f"> 视频口播钩子：{pkg.get('hook', '')}\n"
                          f"> 分镜口播：{' / '.join(str(b.get('say', '')) for b in pkg.get('beats', []))}")
                    pkg["copy_file"] = tools.write_file(tid, "文案.md", md)
                    pkg.update({"title": title, "body": body, "tags": tags})
                    if writer:
                        writer({"say": {"agent": 7, "text": "审校打回 → COPY 已重写落盘"}})
            except Exception:
                pass  # LLM 不可用时保留原稿，问题记录在发布说明里
        note = str(data.get("note") or "见审校问题清单；LLM 结论以本地校验为准")
        pkg["proof"] = {"pass": not issues, "issues": issues, "note": note,
                        "llm_issues": [str(x) for x in (data.get("issues") or [])]}
        if writer:
            writer({"say": {"agent": 7, "text": "审校通过" if not issues else f"审校：{len(issues)} 项待改"}})
        return pkg, (f"[PROOF] {'通过' if not issues else '待改 · ' + '；'.join(issues)}"
                     f"（LLM：{note[:60]}）")

    # VAULT：打包发布说明
    proof = pkg.get("proof") or {}
    pub = (
        f"# 发布说明 · {pkg.get('title', state['name'])}\n\n"
        f"## 文件清单\n"
        f"- 文案：{pkg.get('copy_file', '—')}\n- 封面：{pkg.get('cover', '—')}\n"
        f"- 成片：{pkg.get('video', '—')}（{pkg.get('dur', '?')} 秒）\n- 字幕：{pkg.get('srt', '—')}\n\n"
        f"## 发布内容（复制即用）\n\n**标题**\n\n{pkg.get('title', '')}\n\n**正文**\n\n{pkg.get('body', '')}\n\n"
        f"**标签**\n\n{' '.join(pkg.get('tags', []))}\n\n"
        f"## 审校结论\n\n{'✅ 通过' if proof.get('pass') else '⚠ 待修改：' + '；'.join(proof.get('issues', []))}\n\n"
        f"## 发布建议\n\n- 成片竖屏 1080×1920 直接上传；封面用 封面_3x4.jpg\n"
        f"- 发布时间建议 12:00-13:00 或 20:00-22:00\n- 话题带全标签，前 2 个标签决定初始流量池\n"
        f"- 评论区置顶：补充价格/购买方式，引导互动"
    )
    f = tools.write_file(tid, "发布说明.md", pub)
    files = [pkg.get(k) for k in ("copy_file", "cover", "video", "srt") if pkg.get(k)]
    files += [x for x in pkg.get("frames", []) if x]
    files.append(f)
    pkg["all_files"] = [x for x in dict.fromkeys(files) if x]
    if writer:
        writer({"say": {"agent": 12, "text": f"内容包就绪 · {len(pkg['all_files'])} 个文件"}})
        _emit_files(writer, tid, pkg["all_files"])
    return pkg, f"[VAULT] 内容包 {len(pkg['all_files'])} 个文件就绪，可直接发布"


async def _score_req(state: dict, revision: int, on_event) -> tuple[list, int]:
    """七维需求质量评分：真实 LLM 按维度打分；mock 由审读/补齐统计推导。返回 (dims[7], score)。"""
    reads = sum(len(s["words"]) for s in state["sections"])
    fixes = sum(1 for s in state["sections"] for w in s["words"] if w.get("kind") == "vague") + revision * 2
    rev_notes = state.get("revision_notes") or []
    if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
        notes = "；".join(a for a in state.get("artifacts", []) if a)
        fix_list = [f"{w['w']}→{w['fix']}" for s in state["sections"] for w in s["words"]
                    if w.get("kind") == "vague" and w.get("fix")]
        fix_txt = ("；各站已补齐：" + "；".join(fix_list[:24])) if fix_list else ""
        rev_txt = ("；CHIEF 修订补充：" + "；".join(rev_notes)) if rev_notes else ""
        sys = ROLE_PROMPTS["CHIEF"] + (" 你要为需求质量打分，评分对象是补齐后的整体方案（需求原文 + 各站补齐 + CHIEF 修订补充）"
                                       "是否可以直接执行；CHIEF 的修订补充视为需求的有效组成部分，若已覆盖某维度的缺失，该维度应正常给分而非按原文缺失扣分。")
        prompt = (f"需求：{state['name']}（{state['requirement']}）\n各站审读结论：{notes[:800]}{fix_txt}{rev_txt}\n"
                  f'七个维度依次为 {AXES}。输出严格 JSON：{{"dims": [7 个 0-100 整数], "score": 0-100 总分}}')
        try:
            r = await complete("CHIEF", sys, prompt, on_event)
            s = r.text[r.text.index("{"): r.text.rindex("}") + 1]
            data = json.loads(s)
            dims = [max(0, min(100, int(v))) for v in data.get("dims", [])][:7]
            if len(dims) == 7:
                return dims + [0] * 0, max(0, min(100, int(data.get("score", sum(dims) // 7))))
        except Exception:
            pass
    rnd = _det_seed("score", state["task_id"], str(revision))
    dims = [
        max(0, min(100, int(52 + reads * .8 + rnd.uniform(0, 6)))),     # 完整
        max(0, min(100, int(58 + fixes * 7 + rnd.uniform(0, 6)))),      # 可执行
        max(0, min(100, int(46 + fixes * 9 + rnd.uniform(0, 6)))),      # 可度量
        max(0, min(100, int(70 + rnd.uniform(2, 20)))),                 # 负责人
        max(0, min(100, int(55 + fixes * 8 + rnd.uniform(0, 6)))),      # 截止
        max(0, min(100, int(60 + fixes * 6 + rnd.uniform(0, 6)))),      # 无歧义
        max(0, min(100, int(48 + fixes * 8 + rnd.uniform(0, 10)))),     # 可验收
    ]
    score = int(sum(dims) / len(dims)) + revision * 6
    return dims, max(0, min(97, score))


async def score_node(state: TaskState) -> dict:
    writer = get_stream_writer()
    tid = state["task_id"]

    async def on_event(kind: str, payload) -> None:
        if writer and kind == "usage":
            u = payload
            writer({"usage": {"agent": 0, "in": u.input_tokens, "out": u.output_tokens, "cached": u.cached_tokens}})

    dims, score = await _score_req(state, state.get("revision", 0), on_event)
    if writer:
        writer({"req_score": {"task_id": tid, "dims": dims, "score": score}})

    def dispatch() -> Command:
        if writer:
            writer({"req_dispatched": {"task_id": tid, "score": score}})
        return Command(update={"dims": dims, "final_score": score, "stage": len(state["path"])},
                       goto=node_for(state["path"][0]))

    if score >= PASS_SCORE:
        return dispatch()
    if state.get("revision", 0) < 2:
        # CHIEF 修订：第 1 轮补结构性约定（负责人/截止/验收）；第 2 轮针对低于 75 的维度定向补强
        if state.get("revision", 0) == 0:
            additions = [
                "负责人：每站指定主责（COPY 主笔、PROOF 终审、ADS 放量）",
                "截止：本周五 18:00 前完成终审",
                "验收：审校六项全过、重复率 < 8%、完播 > 35%",
            ]
        else:
            additions = [
                {"完整": "补充交付物清单：脚本终稿、成片、封面 3 版、投放计划表，各站产出逐一登记",
                 "可执行": "补充操作步骤：每站给出可直接执行的步骤与所需输入，不留待口头确认",
                 "可度量": "补充量化口径：响应 < 3s、答案准确率抽检 > 95%、CTR/CVR/CPA 数值与数据源",
                 "负责人": "补充负责人表：COPY 主笔、CUT 剪辑、PROOF 终审、ADS 放量，责任到人",
                 "截止": "补充时间表：脚本 D1、剪辑 D2、审校 D3 上午、投放 D3 下午上线",
                 "无歧义": "补充口径统一：所有模糊词按补齐文本执行，仍有歧义的词直接删除",
                 "可验收": "补充验收清单：审校六项全过 + 数据回看时点（上线后 24h / 72h）"}[AXES[i]]
                for i, v in enumerate(dims) if v < 75][:4]
            if not additions:
                additions = ["补充：各站按补齐文本执行，PROOF 终审后即可交付"]
        if writer:
            for a in additions:
                writer({"say": {"agent": 0, "text": "修订补充 · " + a}})
        return Command(update={"dims": dims, "final_score": score, "revision": state.get("revision", 0) + 1,
                               "revision_notes": additions, "stage": len(state["path"])}, goto="score")
    if writer:
        writer({"req_rejected": {"task_id": tid, "score": score if score < PASS_SCORE else 0}})
    return Command(update={"dims": dims, "final_score": score, "rejected": True, "stage": len(state["path"])}, goto=END)


# ---------------------------------------------------------------- deliver
async def deliver_node(state: TaskState) -> dict:
    writer = get_stream_writer()
    total = {"in": 0, "out": 0, "cached": 0}
    for u in state["usage"]:
        total["in"] += u["in"]; total["out"] += u["out"]; total["cached"] += u["cached"]
    cycle = round(time.time() - state["born"], 1)
    if writer:
        writer({"delivered": {"task_id": state["task_id"], "channel": state["channel"],
                              "cycle": cycle, "tokens": total, "rework": state["rework"]}})
    return {"done": True}


# ---------------------------------------------------------------- 组装
def build_graph():
    g = StateGraph(TaskState)
    g.add_node("chief", chief_node)
    for i in WORKER_INDICES:
        g.add_node(rev_node_for(i), make_review_node(i))   # 需求评审：逐站审词
        g.add_node(node_for(i), make_role_node(i))         # 执行：逐站干活
    g.add_node("score", score_node)
    g.add_node("deliver", deliver_node)
    g.add_edge(START, "chief")

    def route_from_chief(state: TaskState) -> str:
        return rev_node_for(state["path"][0])              # 先评审需求，过线才进入执行

    g.add_conditional_edges("chief", route_from_chief)
    g.add_edge("deliver", END)
    return g.compile()


graph = build_graph()
