"""测试和离线评测使用的 mock provider。

用途：
- 单元测试与评测中的确定性替身，避免依赖网络。
- 通过 `simulate` 触发受控异常（provider error/timeout/invalid_json）。

边界：
- 不发起任何网络调用。
- 返回 `LLMResponse`，供 `LlmClient` 与日志链路保持与真实 provider 一致。
"""

from __future__ import annotations

from collections.abc import Sequence

from model_provider import LLMResponse, TokenUsage


class MockProvider:
    """返回可配置 LLMResponse，且不执行任何网络访问。"""

    def __init__(
        self,
        content: str = "mock response",
        model: str = "mock-model",
        profile_name: str = "mock",
        usage: TokenUsage | None = None,
        raw: dict[str, object] | None = None,
        latency_ms: float = 0.0,
        simulate: str | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.profile_name = profile_name
        self.usage = usage or TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated=True,
        )
        self.raw = raw or {"mock": True}
        self.latency_ms = latency_ms
        self.simulate = simulate

    def call(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """返回配置好的 mock 响应或模拟错误。

模拟开关：
- `provider_error` -> 抛 `RuntimeError`
- `timeout` -> 抛 `TimeoutError`
- `invalid_json` -> 返回非法 JSON 文本内容，配合上层解析路径做回归。
        """

        del messages
        if self.simulate == "provider_error":
            raise RuntimeError("mock provider_error")
        if self.simulate == "timeout":
            raise TimeoutError("mock timeout")

        content = self.content
        raw = dict(self.raw)
        if self.simulate == "invalid_json":
            content = "{invalid json"
            raw = {"mock": True, "simulate": "invalid_json"}

        return LLMResponse(
            content=content,
            provider="mock_provider",
            model=self.model,
            profile_name=self.profile_name,
            latency_ms=self.latency_ms,
            usage=self.usage,
            raw=raw,
        )
