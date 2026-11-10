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


async def upload_binary(path: Path, api_key: str) -> str:
    """上传本地文件换取 URL（1 天有效期，足够当次任务引用）。"""
    async with httpx.AsyncClient(timeout=60) as c:
        for attempt in range(3):
            try:
                r = await c.post(f"{BASE}/media/upload/binary",
                                 headers={"Authorization": f"Bearer {api_key}"},
                                 files={"file": (path.name, open(path, "rb"))})
                r.raise_for_status()
                j = r.json()
                url = (j.get("data") or {}).get("download_url")
                if not url:
                    raise RuntimeError(f"runninghub upload: {j}")
                return url
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError("runninghub upload: unreachable")


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


async def generate_video(prompt: str, out: Path, api_key: str,
                         aspect_ratio: str = "9:16", resolution: str = "768p",
                         duration: float = 10, ref_images: list[str] | None = None,
                         max_wait: int = 900) -> Path:
    """MiniMax H3 ref2va 文生/图参考视频：提交 → 轮询 /query → 下载 mp4（结果 URL 24h 有效）。"""
    uri = "/rhart-video/minimax-h3-rh-enhanced/ref2va"
    body = {"prompt": prompt[:20000], "aspectRatio": aspect_ratio, "resolution": resolution,
            "duration": max(4, min(15, duration)), "audioMode": "native"}
    for i, u in enumerate((ref_images or [])[:9], start=1):
        body[f"refImage{i}"] = u
    async with httpx.AsyncClient(timeout=60) as c:
        headers = {"Authorization": f"Bearer {api_key}"}
        task_id = None
        for attempt in range(3):
            try:
                r = await c.post(BASE + uri, headers=headers, json=body)
                r.raise_for_status()
                j = r.json()
                task_id = j.get("taskId")
                if not task_id:
                    raise RuntimeError(f"runninghub video submit: {j.get('errorMessage') or j}")
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        if not task_id:
            raise RuntimeError("runninghub video submit: unreachable")

        deadline = time.time() + max_wait
        while time.time() < deadline:
            await asyncio.sleep(6)
            try:
                q = await c.post(f"{BASE}/query", headers=headers, json={"taskId": task_id})
                q.raise_for_status()
                d = q.json()
            except httpx.TransportError:
                continue
            st = d.get("status")
            if st == "SUCCESS":
                results = d.get("results") or []
                vid = next((x["url"] for x in results if str(x.get("outputType", "")).lower() in ("mp4", "video")), None)
                if not vid:
                    raise RuntimeError("runninghub video 成功但无 mp4 结果")
                f = await c.get(vid)
                f.raise_for_status()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(f.content)
                return out
            if st == "FAILED":
                raise RuntimeError(f"runninghub video failed: {d.get('errorMessage')}")
        raise RuntimeError("runninghub video 超时未出片")
