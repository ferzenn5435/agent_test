"""LLM token、成本和延迟用量的纯计算工具。

本模块不做网络请求、不访问模型 API，仅负责：
- 将 provider usage 与本地估算归一化为统一 `TokenUsage`
- 聚合为 `UsageSummary`
- 在缺失信息时给出可解释的 `None`，而不是伪造数据
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Protocol

from model_provider import TokenUsage


class PricingLike(Protocol):
    """计算成本所需的 pricing 字段。

协议仅要求输入/输出每百万 token 单价。
"""

    @property
    def input_per_1m_tokens(self) -> float:
        """每 100 万输入 token 的价格。"""

        ...

    @property
    def output_per_1m_tokens(self) -> float:
        """每 100 万输出 token 的价格。"""

        ...


@dataclass(frozen=True)
class UsageCallRecord:
    """单次 LLM 调用的 usage 与延迟。"""

    latency_ms: float
    usage: TokenUsage


@dataclass(frozen=True)
class UsageSummary:
    """多次 LLM 调用的稳定汇总 schema。

`estimated_cost` 允许为 `None`：只要任一调用缺少可计算成本，整次聚合
也不返回具体成本。
    """

    llm_call_count: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_tokens: int = 0
    estimated_cost: float | None = None

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "llm_call_count": self.llm_call_count,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost": self.estimated_cost,
        }


def normalize_call_usage(
    prompt_text: str,
    completion_text: str,
    provider_usage: TokenUsage | None,
    pricing: PricingLike | None,
) -> TokenUsage:
    """标准化单次调用 usage，provider 返回值优先于本地估算。

优先级：
1) `provider_usage` 非空：原样保留返回 token，并补齐总 token。
2) provider 缺失：按字符数估算，估算规则 `ceil(len(text)/4)`。
3) 成本仅在可读取 pricing 且 token 全量可用时计算。
    """

    if provider_usage is not None:
        prompt_tokens = provider_usage.prompt_tokens
        completion_tokens = provider_usage.completion_tokens
        total_tokens = _normalize_total_tokens(
            prompt_tokens,
            completion_tokens,
            provider_usage.total_tokens,
        )
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated=False,
            estimated_cost=_calculate_cost(prompt_tokens, completion_tokens, pricing),
        )

    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(completion_text)
    total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated=True,
        estimated_cost=_calculate_cost(prompt_tokens, completion_tokens, pricing),
    )


def summarize_usage_calls(call_records: list[UsageCallRecord]) -> UsageSummary:
    """汇总多次 LLM 调用的延迟、token 和成本。

当任一记录 `estimated_cost is None` 时，最终汇总 `estimated_cost` 置为 `None`，
表示整体可比较性不完整，但仍保留各类 token 与耗时总和。
    """

    total_latency_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_tokens = 0
    estimated_cost = 0.0
    has_unknown_cost = False

    for call_record in call_records:
        usage = call_record.usage
        call_total_tokens = _known_token_count(usage.total_tokens)

        total_latency_ms += call_record.latency_ms
        prompt_tokens += _known_token_count(usage.prompt_tokens)
        completion_tokens += _known_token_count(usage.completion_tokens)
        total_tokens += call_total_tokens
        if usage.estimated:
            estimated_tokens += call_total_tokens
        if usage.estimated_cost is None:
            has_unknown_cost = True
        else:
            estimated_cost += usage.estimated_cost

    return UsageSummary(
        llm_call_count=len(call_records),
        total_latency_ms=total_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_tokens=estimated_tokens,
        estimated_cost=None if has_unknown_cost else _normalize_cost(estimated_cost),
    )


def _estimate_tokens(text: str) -> int:
    """按字符近似估算 token 数：`ceil(len/4)`。

该估算仅用于 provider 未返回 usage 的兜底场景，适用于输出和提示词。
"""
    return ceil(len(text) / 4)


def _calculate_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: PricingLike | None,
) -> float | None:
    """按 1M token 计价模型计算成本。

任一输入为 `None` 或未配置 `pricing` 时返回 `None`，由上层用于决定
是否可展示估算成本。
"""
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = prompt_tokens / 1_000_000 * pricing.input_per_1m_tokens
    output_cost = completion_tokens / 1_000_000 * pricing.output_per_1m_tokens
    return _normalize_cost(input_cost + output_cost)


def _normalize_cost(cost: float) -> float:
    """固定成本精度，降低日志差异抖动。"""
    return round(cost, 12)


def _normalize_total_tokens(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> int | None:
    """规范化 total token：provider 提供优先，否则回退为 prompt/completion 之和。"""
    if total_tokens is not None:
        return total_tokens
    if prompt_tokens is None or completion_tokens is None:
        return None
    return prompt_tokens + completion_tokens


def _known_token_count(token_count: int | None) -> int:
    """把 `None` 映射为 0，用于汇总。"""
    if token_count is None:
        return 0
    return token_count
