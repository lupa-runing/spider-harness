"""统一图像生成入口：RunningHub（优先，中文 prompt）→ LiblibAI（兜底）。

两者都未配置或全部失败时抛异常，由调用方回退 PIL 文字卡。
"""
from __future__ import annotations

from pathlib import Path

from .config import settings


async def generate(prompt: str, out: Path, w: int = 1024, h: int = 1366) -> Path:
    last = None
    if settings.runninghub_api_key:
        try:
            from .imagerh import generate as rh
            return await rh(prompt, out, settings.runninghub_api_key, w=w, h=h)
        except Exception as e:
            last = e
    if settings.liblib_access_key and settings.liblib_secret_key:
        try:
            from .imagelib import generate as lib
            return await lib(prompt, out, settings.liblib_access_key, settings.liblib_secret_key, w=w, h=h)
        except Exception as e:
            last = e
    raise RuntimeError(f"所有图像服务失败: {last}")


def configured() -> bool:
    return bool(settings.runninghub_api_key or (settings.liblib_access_key and settings.liblib_secret_key))
