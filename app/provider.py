"""LLM provider 层：openai_compatible（任意 OpenAI 兼容网关）与 mock（确定性占位）双实现。

统一接口：complete(role_key, system, prompt, on_event) -> LLMResult
on_event(kind, payload) 用来边生成边上报，kind：
  - "chunk"  (text_delta, usage_delta)  —— 增量文本与增量 token，graph 层换算成进度
  - "usage"  Usage                      —— 结束时的累计用量
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .config import settings

OnEvent = Callable[[str, object], Awaitable[None]]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class LLMResult:
    text: str = ""
    usage: Usage = field(default_factory=Usage)


# 每个角色一次调用的典型 token 量（mock 用；也作为真实流式的进度分母参考）
ROLE_TOKEN_BUDGET: dict[str, tuple[int, int]] = {
    # role: (input_tokens, output_tokens)
    "CHIEF": (900, 350),
    "SCOUT": (2600, 900),
    "SPLIT": (1500, 500),
    "COPY": (1800, 1400),
    "DESIGN": (1200, 1100),
    "CODE": (2200, 1800),
    "TEST": (1600, 700),
    "PROOF": (3200, 600),
    "DATA": (2000, 800),
    "CUT": (1400, 600),
    "ADS": (1300, 500),
    "CARE": (1100, 600),
    "VAULT": (900, 300),
}


async def complete(role_key: str, system: str, prompt: str, on_event: OnEvent | None = None) -> LLMResult:
    if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
        return await _complete_openai(role_key, system, prompt, on_event)
    return await _complete_mock(role_key, system, prompt, on_event)


# ---------------------------------------------------------------- openai_compatible
async def _complete_openai(role_key: str, system: str, prompt: str, on_event: OnEvent | None) -> LLMResult:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        streaming=True,
        stream_usage=True,
        temperature=0.6,
    )
    result = LLMResult()
    chunks = 0
    async for chunk in llm.astream([SystemMessage(content=system), HumanMessage(content=prompt)]):
        result.text += chunk.content if isinstance(chunk.content, str) else ""
        chunks += 1
        if chunk.usage_metadata:
            u = chunk.usage_metadata
            result.usage = Usage(
                u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("input_token_details", {}).get("cache_read", 0) or 0
            )
            if on_event:
                await on_event("usage", result.usage)
        if on_event and chunk.content:
            # 流式进度：按块数逼近预算产出量，最后由 usage 校正
            budget = ROLE_TOKEN_BUDGET.get(role_key, (1000, 800))[1]
            await on_event("chunk", (str(chunk.content), max(0, budget // max(chunks, 1))))
    if not result.usage.input_tokens and not result.usage.output_tokens:
        # 网关不给 usage 时按预算估算，保证 token 墙仍有数据
        i, o = ROLE_TOKEN_BUDGET.get(role_key, (1000, 800))
        result.usage = Usage(i, o, int(i * 0.6))
        if on_event:
            await on_event("usage", result.usage)
    return result


# ---------------------------------------------------------------- mock
def _det_seed(*parts: str) -> random.Random:
    return random.Random(int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16))


MOCK_DRAFTS: dict[str, list[str]] = {
    "CHIEF": ["按工作流模板拆解完成，各站位任务书已下发。"],
    "SCOUT": ["检索到 18 篇相关资料，去重后保留 9 篇高价值源，引用出处已标注。",
              "全网扫描完成：3 份行业报告 + 6 条竞品动态，数据源可信度已分级。"],
    "SPLIT": ["需求切成 5 个子任务，依赖关系理顺，验收标准逐条写清。"],
    "COPY": ["脚本第三版定稿：开头 5 秒钩子 + 4 段正文 + 互动引导，口语化处理完成。"],
    "DESIGN": ["出了 6 版视觉稿，主色压暗 8%，焦点移至左上，选 2 版进入评审。"],
    "CODE": ["接口层联调通过，本地跑通全链路，热路径加了缓存，QPS 提升明显。"],
    "TEST": ["32 条用例全部通过，补 2 条边界用例，回归无新增失败。"],
    "PROOF": ["审校完成：事实核验通过，合规敏感词零命中，可交付。"],
    "DATA": ["本周完成率 87%，异常值剔除后趋势平稳，图表已出。"],
    "CUT": ["成片节奏重剪：高光切 6 段，字幕对齐，封面帧选定。"],
    "ADS": ["小预算 AB 跑完：B 组点击率高 21%，已放量至 2000。"],
    "CARE": ["47 条评论聚类完成，高频问题 Top5 已归因，回访话术同步更新。"],
    "VAULT": ["素材打好标签入库，版本留底，索引重建完成。"],
}

MOCK_REVIEW_CHECKS = ["需求覆盖", "事实核验", "品牌口径", "合规敏感词", "重复率", "交付时限"]


async def _complete_mock(role_key: str, system: str, prompt: str, on_event: OnEvent | None) -> LLMResult:
    rnd = _det_seed(role_key, prompt[:64])
    in_toks, out_toks = ROLE_TOKEN_BUDGET.get(role_key, (1000, 800))
    cached = int(in_toks * rnd.uniform(0.55, 0.72))
    usage = Usage(in_toks, out_toks, cached)

    # 4~9 秒分批发完，模拟真实流式节奏
    n_batches = rnd.randint(5, 9)
    text = " ".join(rnd.choice(MOCK_DRAFTS.get(role_key, ["阶段产物已生成。"])) for _ in range(2))
    per_call_in = in_toks / n_batches
    per_call_out = out_toks / n_batches
    for i in range(n_batches):
        await asyncio.sleep(rnd.uniform(0.6, 1.4))
        if on_event:
            await on_event("chunk", ("", int(per_call_out)))
    if on_event:
        await on_event("usage", usage)
    return LLMResult(text=text, usage=usage)
