"""LLM provider 数据模型与协议。

该模块定义了 `LlmClient` 与 provider 实现之间的稳定边界，避免上层逻辑直接
依赖具体供应商返回格式。

边界要点：
- Response 使用 `LLMResponse` 统一字段（content/provider/model/profile/latency/usage/raw）。
- Usage 使用 `TokenUsage` 描述 prompt/completion/total tokens、是否估算、估算成本。
- Provider 仅需实现 `ModelProvider.call()` 接口即可接入。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenUsage:
    """单次 LLM 调用的 token 用量。

`estimated_cost` 仅表示本次调用的可计算成本，可能为 `None`。
`estimated` 标记该记录是否由本地估算产生。
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = False
    estimated_cost: float | None = None

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。

logger 与运行汇总仅依赖此结果，不要求持有原始对象引用。
        """

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class LLMResponse:
    """单次 LLM provider 调用的结构化响应。

`raw` 保留供应商原始 payload 以便排错与审计；上层默认依赖 `content`
与 `usage`，避免与供应商 schema 产生耦合。
    """

    content: str
    provider: str
    model: str
    profile_name: str
    latency_ms: float
    usage: TokenUsage
    raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "profile_name": self.profile_name,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict(),
            "raw": dict(self.raw),
        }


@runtime_checkable
class ModelProvider(Protocol):
    """执行一次 LLM 调用的 provider 协议。

协议约定：
- `messages` 输入为标准化聊天消息序列（列表中每项含 role/content）。
- 返回 `LLMResponse`；遇到错误可抛异常，由上层统一重抛。
"""

    def call(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """发送聊天消息并返回结构化响应。"""
        ...
