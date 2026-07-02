"""LLM 客户端门面。

职责：统一构建 provider 与 profile，并只通过 `chat_response()` 暴露结构化
`LLMResponse` 调用结果。
"""

from __future__ import annotations

from collections.abc import Sequence

from config import ModelProfile, OPENAI_COMPATIBLE_PROVIDER, load_model_profile
from model_provider import LLMResponse, ModelProvider
from providers.openai_compatible import OpenAICompatibleProvider, ProviderError


MOCK_PROVIDER = "mock"
DEFAULT_MAX_OUTPUT_TOKENS = 4096

LlmClientError = ProviderError


class LlmClient:
    """Profile-aware provider facade。

    该类是主流程与 provider 的唯一入口，不直接发起 HTTP，而是
    委托给 `ModelProvider` 实现。测试/评测可注入 provider，以便无需外网。
    """

    def __init__(
        self,
        *,
        model_profile: str | ModelProfile | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        """解析 profile 并完成 provider 绑定。

        决策优先级：
1. 显式 `model_profile` 为 `ModelProfile` 实例
2. 注入 `provider` 时使用 mock profile 占位符
3. 无注入 provider 时按 `model_profile` 名称读取配置文件
   （默认 `default`）。
        """
        profile = self._resolve_profile(model_profile, provider is not None)
        self.profile = profile
        self.provider = provider or self._build_provider(profile)
        self._secret_values = self._collect_secret_values(profile)

    def chat_response(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        """发送聊天消息并返回包含 usage 的结构化响应。"""

        try:
            return self.provider.call(messages)
        except ProviderError as error:
            raise LlmClientError(self._sanitize_error_text(str(error))) from error

    def _resolve_profile(
        self,
        model_profile: str | ModelProfile | None,
        has_provider: bool,
    ) -> ModelProfile:
        """解析 profile：在注入 provider 且无显式 config 时返回 mock profile。"""
        if isinstance(model_profile, ModelProfile):
            return model_profile
        if has_provider:
            return self._profile_from_injected_provider(model_profile)
        profile_name = model_profile or "default"
        return load_model_profile(profile_name)

    def _profile_from_injected_provider(
        self,
        model_profile: str | ModelProfile | None,
    ) -> ModelProfile:
        """为外部 provider 注入创建占位 profile。

        该 profile 仅用于错误脱敏与测试注入，不参与 provider 调用参数传递。
        """
        profile_name = model_profile if isinstance(model_profile, str) else MOCK_PROVIDER
        return ModelProfile(
            name=profile_name,
            provider=MOCK_PROVIDER,
            base_url="",
            api_key="",
            model="",
            temperature=0.0,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            timeout_seconds=0,
            retry_count=0,
            pricing=None,
        )

    def _build_provider(self, profile: ModelProfile) -> ModelProvider:
        """按 profile 构建 provider。

当前仅支持 `openai_compatible` 与 `mock`；其余 provider 直接报错。
"""
        if profile.provider == OPENAI_COMPATIBLE_PROVIDER:
            return OpenAICompatibleProvider(profile)
        if profile.provider == MOCK_PROVIDER:
            from providers.mock_provider import MockProvider

            return MockProvider(model=profile.model, profile_name=profile.name)
        raise LlmClientError(f"不支持的 LLM provider: {profile.provider}")

    def _collect_secret_values(self, profile: ModelProfile) -> tuple[str, ...]:
        """收集用于错误信息脱敏的敏感字段。

当前仅收集 `api_key`，在日志/错误输出中按文本替换为 [redacted]。
"""
        secret_values = [profile.api_key.strip()]
        return tuple(secret for secret in secret_values if secret)

    def _sanitize_error_text(self, error_text: str) -> str:
        """脱敏 provider 错误文案，去除 api key 与 Authorization 字段痕迹。"""
        sanitized_text = error_text
        for secret_value in self._secret_values:
            sanitized_text = sanitized_text.replace(secret_value, "[redacted]")
        return sanitized_text.replace("Authorization", "[redacted-header]")
