"""FastAPI 入口：
  POST /api/tasks        提需求（可选指定工作流）
  GET  /api/state        全量快照（首连/重连用）
  GET  /api/events       SSE 事件流
  GET  /api/workflows    工作流模板列表
  /                      静态托管 agent-display（同源，免 CORS）
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .bridge import harness, snapshot
from .config import settings
from .roles import WORKFLOWS
from . import tools

app = FastAPI(title="spider-harness", version="0.1.0")

# 内容包静态托管：结果面板的文件链接指向 /outputs/<task_id>/...
tools.WORKSPACE.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(tools.WORKSPACE)), name="outputs")


class SubmitBody(BaseModel):
    requirement: str
    wf: str | None = None
    ai_video: bool = False


@app.get("/api/workflows")
def workflows():
    return [{"id": w["id"], "zh": w["zh"], "path": w["path"], "ch": w["ch"], "names": w["names"]} for w in WORKFLOWS]


@app.post("/api/tasks")
async def submit(body: SubmitBody):
    requirement = body.requirement.strip()
    if not requirement:
        raise HTTPException(400, "requirement 不能为空")
    if len(requirement) > 2000:
        raise HTTPException(400, "requirement 过长")
    rec = harness.submit(requirement, body.wf, ai_video=body.ai_video)
    return {"id": rec.id, "wf": rec.wf_id, "name": rec.name, "aiVideo": rec.ai_video}


@app.get("/api/state")
def state():
    return snapshot(harness)


@app.get("/api/tasks")
def tasks_list():
    """任务列表（内存中的任务，重启后用 /api/history 看磁盘历史包）。"""
    items = []
    for t in sorted(harness.tasks.values(), key=lambda x: x.born, reverse=True):
        items.append({"id": t.id, "name": t.name, "wf": t.wf_id, "status": t.status,
                      "stage": t.stage, "final": (t.req or {}).get("final", 0),
                      "born": t.born, "files": len(t.files)})
    return {"tasks": items}


@app.get("/api/history")
def history():
    """磁盘历史内容包（outputs/ 下每个任务目录，重启不丢）。"""
    out = []
    root = tools.WORKSPACE
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir() or d.name.startswith(".") or d.name.startswith("_"):
                continue
            files = sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("_"))
            if not files:
                continue
            name = d.name
            try:
                for ln in (d / "发布说明.md").read_text(encoding="utf-8").splitlines():
                    if ln.startswith("# "):
                        name = ln[2:].strip()
                        break
            except OSError:
                pass
            out.append({"id": d.name, "name": name, "mtime": d.stat().st_mtime, "files": len(files)})
    return {"history": out}


@app.get("/api/tasks/{task_id}/result")
def task_result(task_id: str):
    """单个任务的完整结果：需求、逐站审读结论、模糊词补齐对照、七维评分、执行产物、内容包。"""
    rec = harness.tasks.get(task_id)
    if rec:
        req = rec.req or {}
        return {
            "id": rec.id, "name": rec.name, "requirement": rec.requirement, "wf": rec.wf_id,
            "path": rec.path, "status": rec.status, "stage": rec.stage,
            "score": req.get("score", 0), "final": req.get("final", 0), "dims": req.get("dims"),
            "sections": req.get("sections"),
            "briefs": rec.briefs, "notes": rec.notes, "outputs": rec.outputs,
            "pkg": rec.pkg, "files": rec.files,
            "tokens": rec.tokens, "cycle": rec.cycle, "born": rec.born,
            "provider": settings.llm_provider,
        }
    # 重启后的磁盘历史：从 outputs/<id>/ 重建详情
    d = tools.WORKSPACE / task_id
    if not d.is_dir():
        raise HTTPException(404, "task not found")

    def rd(name: str) -> str:
        p = d / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    files = sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("_"))
    return {
        "id": task_id, "name": task_id, "requirement": "", "wf": "—", "path": [], "status": "done",
        "stage": 0, "score": 0, "final": 0, "dims": None, "sections": [], "briefs": [],
        "notes": [], "outputs": [], "pkg": {}, "files": [f"outputs/{task_id}/{f}" for f in files],
        "tokens": {}, "cycle": 0, "born": d.stat().st_mtime, "provider": settings.llm_provider,
        "copy_md": rd("文案.md"), "pub_md": rd("发布说明.md"),
    }


@app.get("/api/events")
async def events():
    q = await harness.subscribe()

    async def gen():
        try:
            yield {"event": "snapshot", "data": json.dumps(snapshot(harness), ensure_ascii=False)}
            while True:
                with contextlib.suppress(asyncio.TimeoutError):
                    event, data = await asyncio.wait_for(q.get(), timeout=settings.token_tick_seconds)
                    yield {"event": event, "data": json.dumps(data, ensure_ascii=False)}
                    continue
                # 空闲期广播一次累计 token，保持 token 墙活跃
                yield {"event": "heartbeat", "data": json.dumps({"ts": time.time()}, ensure_ascii=False)}
        finally:
            harness.unsubscribe(q)

    return EventSourceResponse(gen(), ping=15)


# ---- 静态托管 agent-display：http://127.0.0.1:8000/#spider 即可用真数据看大屏 ----
_DISPLAY_DIR = (Path(__file__).resolve().parent.parent / "agent-display").resolve()
if _DISPLAY_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_DISPLAY_DIR), html=True), name="display")
else:  # 找不到大屏目录时给个根路径说明
    @app.get("/")
    def root():
        return {"hint": f"把 agent-display 克隆到 {_DISPLAY_DIR} 即可同源托管大屏"}
