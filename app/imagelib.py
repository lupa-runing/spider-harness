"""LiblibAI 星流 Star-3 Alpha 文生图接入。

鉴权：query string 传 AccessKey / Signature / Timestamp / SignatureNonce；
Signature = urlsafe_b64_nopad(HmacSHA1(uri + "&" + ts + "&" + nonce, SecretKey))。
流程：POST 提交任务拿 generateUuid → 轮询 status → 下载 imageUrl。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
import uuid as _uuid
from pathlib import Path

import httpx

BASE = "https://openapi.liblibai.cloud"
TPL_TEXT2IMG = "5d7e67009b344550bc1aa6ccbfa1d7f4"   # 星流 Star-3 Alpha 文生图


def _sign(uri: str, ts: str, nonce: str, secret_key: str) -> str:
    content = "&".join((uri, ts, nonce))
    digest = hmac.new(secret_key.encode(), content.encode(), hashlib.sha1).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _auth_params(uri: str, ak: str, sk: str) -> dict:
    ts = str(int(time.time() * 1000))
    nonce = _uuid.uuid4().hex
    return {"AccessKey": ak, "Signature": _sign(uri, ts, nonce, sk),
            "Timestamp": ts, "SignatureNonce": nonce}


async def available(ak: str | None, sk: str | None) -> bool:
    return bool(ak and sk)


async def generate(prompt_en: str, out: Path, ak: str, sk: str,
                   w: int = 1024, h: int = 1366, max_wait: int = 150) -> Path:
    """生成一张图并下载到 out。抛异常表示失败（调用方自行回退文字卡）。"""
    uri = "/api/generate/webui/text2img/ultra"
    body = {"templateUuid": TPL_TEXT2IMG,
            "generateParams": {"prompt": prompt_en[:2000], "imgCount": 1, "steps": 30,
                               "imageSize": {"width": max(512, min(2048, w)), "height": max(512, min(2048, h))}}}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(BASE + uri, params=_auth_params(uri, ak, sk), json=body)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != 0:
            raise RuntimeError(f"liblib submit: {j.get('msg')}")
        gen_uuid = j["data"]["generateUuid"]

        quri = "/api/generate/webui/status"
        deadline = time.time() + max_wait
        images = []
        while time.time() < deadline:
            await asyncio.sleep(3)
            q = await client.post(BASE + quri, params=_auth_params(quri, ak, sk),
                                  json={"generateUuid": gen_uuid})
            q.raise_for_status()
            d = q.json().get("data") or {}
            images = d.get("images") or []
            if images:
                break
        if not images:
            raise RuntimeError(f"liblib 超时未出图（status={d.get('generateStatus')}）")
        img_url = images[0]["imageUrl"]
        img = await client.get(img_url)
        img.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img.content)
        return out
