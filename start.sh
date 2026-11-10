#!/bin/bash
# 一键启动 spider-harness（LangGraph 后端 + agent-display 大屏 + outputs 内容包，全在 8000 端口）
# 用法：./start.sh          前台运行
#       nohup ./start.sh &  后台运行
cd "$(dirname "$0")"
[ -d .venv ] || uv sync
exec uv run uvicorn app.main:app --port 8000
