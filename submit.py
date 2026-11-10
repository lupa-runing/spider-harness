"""提交需求的 CLI：uv run python submit.py "需求文本" [--wf W03] [--num 3]"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description="向 spider-harness 提交需求")
    ap.add_argument("requirement", help="需求文本")
    ap.add_argument("--wf", default=None, help="指定工作流模板，如 W01..W11")
    ap.add_argument("--num", type=int, default=1, help="提交数量（默认 1）")
    args = ap.parse_args()

    for i in range(args.num):
        try:
            resp = post("/api/tasks", {"requirement": args.requirement, "wf": args.wf})
        except Exception as e:
            print(f"提交失败：{e}（先启动服务：uv run uvicorn app.main:app --port 8000）")
            return 1
        print(f"[{i + 1}/{args.num}] 已受理 {resp['id']} · {resp['wf']} · {resp['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
