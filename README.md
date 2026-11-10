# spider-harness

用 **LangGraph** 驱动的真实多 agent 流水线，为 **agent-display**（SPIDER 蛛网战术 / NEBULA 星核蜘蛛 大屏）提供真实事件流与真实交付产物。前后端已合并为单项目：`agent-display/` 是大屏前端（PolyForm 非商业许可，原作者 cyberdaddyleo），`app/` 是 harness 后端。

## 架构

```
agent-display 大屏 (index.html #spider / #nebula)
    ↑ SSE 实时事件 + REST 快照（同源，StaticFiles 托管，免 CORS）
spider-harness (FastAPI)
    ↑ LangGraph StateGraph：chief 拆解 → 逐站审词(模糊追问) → 七维评分(不过线打回) → 派单 → 逐站执行 → 交付
    ↑ 工具层 tools.py：文件落盘 / PIL 文字卡 / edge-tts 配音 / ffmpeg 合成视频
    ↑ 图像生成 imagelib.py：LiblibAI 星流 Star-3 Alpha（AK/SK 签名，封面与分镜真图）
LLM provider：openai_compatible（DeepSeek 等 OpenAI 兼容网关）或 mock（占位，链路全真）
```

- 角色与工序配置在 `app/roles.py`，与 `index.html` mountSpider 的 `AG`/`WF` 常量**同构同源**（下标即契约，改动需两边同步）。
- 打回语义：PROOF 节点输出六项检查 JSON 裁决，`fail` 且打回次数 <2 时 `Command(goto=上一站)`，前端收到 `review` + `stage_done` 后走它原有的碎纸/打回动画。
- 大屏保留自己的状态机与全部动画，后端只喂事实（接手/进度/裁决/token），不喂动画指令。

## 启动

```bash
cd spider-harness
uv sync                                    # 首次安装
uv run uvicorn app.main:app --port 8000    # 起服务（默认 mock provider）
```

浏览器打开 **http://127.0.0.1:8000/#spider**（harness 会同源托管 agent-display）。
若 `/api/state` 不可达，大屏自动退回内置模拟模式，随时可切换。

## 提需求

```bash
uv run python submit.py "做一个双11推广短视频"            # 自动匹配工作流
uv run python submit.py "写技术方案标书" --wf W03         # 指定工作流模板
uv run python submit.py "跑一批素材" --num 3              # 并发 3 个任务
# 或 curl：curl -X POST localhost:8000/api/tasks -H 'Content-Type: application/json' -d '{"requirement":"..."}'
```

## 接入真实 LLM

编辑 `.env`（无 key 时用 mock，事件结构/token 账本/裁决全部真实流动，仅生成物为占位）：

```ini
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://llm.campusphere.net/v1   # 或 https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
```

重启即生效。审校裁决依赖模型输出 JSON，解析失败会退回确定性占位裁决。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/tasks` | `{requirement, wf?}` 提需求，返回任务 id |
| GET | `/api/state` | 全量快照（大屏首连/重连） |
| GET | `/api/events` | SSE 事件流 |
| GET | `/api/workflows` | 11 条工作流模板 |

SSE 事件：`snapshot / task_created / task_renamed / stage_started / stage_progress / stage_done / review / say / token_delta / delivered / task_error / heartbeat`。

## 已知边界

- 任务不落盘，服务重启即清空
- 并发上限 `MAX_CONCURRENT_TASKS`（默认 3）
- agent-display 为 PolyForm Noncommercial License，仅限非商业用途
