"""RunningHub 图像生成接入（Seedream v5 Flash 文生图，Bearer Key 鉴权，异步轮询）。

文档要点：prompt 支持中文（5-5000 字符）；width/height 240-8192（或 resolution 1k/1.5k/2k）；
结果 URL 有效期 24 小时——生成后立即下载落盘。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

BASE = "https://www.runninghub.cn/openapi/v2"


async def generate(prompt: str, out: Path, api_key: str,
                   w: int = 1024, h: int = 1366, max_wait: int = 180) -> Path:
    """提交文生图任务并轮询到 SUCCESS，下载图片到 out。失败抛异常（调用方回退）。"""
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"prompt": prompt[:5000],
            "width": max(240, min(8192, w)), "height": max(240, min(8192, h)),
            "outputFormat": "jpeg"}
    async with httpx.AsyncClient(timeout=30) as c:
        j = None
        for attempt in range(3):
            try:
                r = await c.post(f"{BASE}/seedream-v5-flash/text-to-image", headers=headers, json=body)
                r.raise_for_status()
                j = r.json()
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
        task_id = (j or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"runninghub submit: {(j or {}).get('errorMessage') or j}")

        deadline = time.time() + max_wait
        while time.time() < deadline:
            await asyncio.sleep(3)
            try:
                q = await c.post(f"{BASE}/query", headers=headers, json={"taskId": task_id})
                q.raise_for_status()
                d = q.json()
            except httpx.TransportError:
                continue                          # 轮询网络抖动：下个周期再查
            st = d.get("status")
            if st == "SUCCESS":
                results = d.get("results") or []
                if not results:
                    raise RuntimeError("runninghub 成功但无结果")
                img = await c.get(results[0]["url"])
                img.raise_for_status()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(img.content)
                return out
            if st == "FAILED":
                raise RuntimeError(f"runninghub failed: {d.get('errorMessage')}")
            # QUEUED / RUNNING → 继续轮询
        raise RuntimeError("runninghub 超时未出图")
