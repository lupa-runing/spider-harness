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
    rec = harness.submit(requirement, body.wf)
    return {"id": rec.id, "wf": rec.wf_id, "name": rec.name}


@app.get("/api/state")
def state():
    return snapshot(harness)


@app.get("/api/tasks/{task_id}/result")
def task_result(task_id: str):
    """单个任务的完整结果：需求、逐站审读结论、模糊词补齐对照、七维评分、执行产物。"""
    rec = harness.tasks.get(task_id)
    if not rec:
        raise HTTPException(404, "task not found")
    req = rec.req or {}
    return {
        "id": rec.id, "name": rec.name, "requirement": rec.requirement, "wf": rec.wf_id,
        "path": rec.path, "status": rec.status, "stage": rec.stage,
        "score": req.get("score", 0), "final": req.get("final", 0), "dims": req.get("dims"),
        "sections": req.get("sections"),
        "briefs": rec.briefs, "notes": rec.notes, "outputs": rec.outputs,
        "files": rec.files,
        "tokens": rec.tokens, "cycle": rec.cycle,
        "provider": settings.llm_provider,
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
