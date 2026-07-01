"""LLM client provider facade 单元测试。"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from unittest.mock import patch

from config import LlmConfig, ModelProfile
from llm_client import LlmClient, LlmClientError
from model_provider import LLMResponse, TokenUsage
from providers.mock_provider import MockProvider
from providers.openai_compatible import ProviderError


class RecordingProvider:
    """测试用 provider：记录每次调用的 messages 和 profile_name，返回预设 LLMResponse。"""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    def call(
        self,
        messages: Sequence[dict[str, str]],
        profile_name: str,
    ) -> LLMResponse:
        self.calls.append(([dict(message) for message in messages], profile_name))
        return self.response


class FailingProvider:
    """测试用 provider：抛出包含 API key 和 Authorization 的错误消息，用于验证脱敏逻辑。"""

    def call(
        self,
        messages: Sequence[dict[str, str]],
        profile_name: str,
    ) -> LLMResponse:
        del messages, profile_name
        raise ProviderError("bad secret-key Authorization Bearer secret-key")


class TestLlmClientFacade(unittest.TestCase):
    """验证 LlmClient 兼容旧接口并暴露结构化响应。"""

    def test_chat_preserves_legacy_string_return_with_injected_provider(self) -> None:
        provider = MockProvider(content="legacy text", profile_name="mock-profile")
        client = LlmClient(provider=provider, model_profile="mock-profile")

        response_text = client.chat([{"role": "user", "content": "ping"}])

        self.assertEqual("legacy text", response_text)

    def test_chat_response_returns_llm_response_from_injected_provider(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        provider = MockProvider(
            content="structured text",
            model="mock-model",
            profile_name="mock-profile",
            usage=usage,
            raw={"mock": True},
            latency_ms=12.5,
        )
        client = LlmClient(provider=provider, model_profile="mock-profile")

        response = client.chat_response([{"role": "user", "content": "ping"}])

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual("structured text", response.content)
        self.assertEqual("mock_provider", response.provider)
        self.assertEqual("mock-model", response.model)
        self.assertEqual("mock-profile", response.profile_name)
        self.assertEqual(3, response.usage.total_tokens)
        self.assertEqual({"mock": True}, response.raw)
        self.assertEqual(12.5, response.latency_ms)

    def test_legacy_llm_config_uses_openai_compatible_provider_path(self) -> None:
        config = LlmConfig(
            base_url="https://example.test/v1",
            api_key="legacy-secret",
            model="legacy-model",
            timeout_seconds=9,
        )
        response = LLMResponse(
            content="legacy response",
            provider="openai_compatible",
            model="legacy-model",
            profile_name="legacy",
            latency_ms=1.0,
            usage=TokenUsage(total_tokens=1),
            raw={"ok": True},
        )
        provider = RecordingProvider(response)

        with patch("llm_client.OpenAICompatibleProvider", return_value=provider) as provider_factory:
            client = LlmClient(config)
            response_text = client.chat([{"role": "user", "content": "ping"}])

        self.assertEqual("legacy response", response_text)
        provider_factory.assert_called_once()
        profile = provider_factory.call_args.args[0]
        self.assertEqual("legacy", profile.name)
        self.assertEqual("openai_compatible", profile.provider)
        self.assertEqual("https://example.test/v1", profile.base_url)
        self.assertEqual("legacy-secret", profile.api_key)
        self.assertEqual("legacy-model", profile.model)
        self.assertEqual(0.0, profile.temperature)
        self.assertEqual(4096, profile.max_output_tokens)
        self.assertEqual(9, profile.timeout_seconds)
        self.assertEqual(0, profile.retry_count)
        self.assertEqual([([{"role": "user", "content": "ping"}], "legacy")], provider.calls)

    def test_model_profile_name_loads_profile_with_single_loader(self) -> None:
        profile = ModelProfile(
            name="default",
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key="profile-secret",
            model="profile-model",
            temperature=0.1,
            max_output_tokens=100,
            timeout_seconds=11,
            retry_count=2,
            pricing=None,
        )
        response = LLMResponse(
            content="profile response",
            provider="openai_compatible",
            model="profile-model",
            profile_name="default",
            latency_ms=1.0,
            usage=TokenUsage(total_tokens=1),
            raw={"ok": True},
        )
        provider = RecordingProvider(response)

        with patch("llm_client.load_model_profile", return_value=profile) as loader:
            with patch("llm_client.OpenAICompatibleProvider", return_value=provider):
                client = LlmClient(model_profile="default")
                llm_response = client.chat_response([{"role": "user", "content": "ping"}])

        loader.assert_called_once_with("default")
        self.assertEqual("profile response", llm_response.content)
        self.assertEqual([([{"role": "user", "content": "ping"}], "default")], provider.calls)

    def test_unsupported_provider_is_rejected_without_dynamic_plugins(self) -> None:
        profile = ModelProfile(
            name="custom",
            provider="dynamic_plugin",
            base_url="https://example.test/v1",
            api_key="secret",
            model="model",
            temperature=0.0,
            max_output_tokens=100,
            timeout_seconds=10,
            retry_count=0,
            pricing=None,
        )

        with self.assertRaisesRegex(LlmClientError, "不支持的 LLM provider"):
            LlmClient(model_profile=profile)

    def test_provider_error_message_redacts_api_key_and_authorization(self) -> None:
        profile = ModelProfile(
            name="safe",
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key="secret-key",
            model="model",
            temperature=0.0,
            max_output_tokens=100,
            timeout_seconds=10,
            retry_count=0,
            pricing=None,
        )
        client = LlmClient(model_profile=profile, provider=FailingProvider())

        with self.assertRaises(LlmClientError) as error_context:
            client.chat_response([{"role": "user", "content": "ping"}])

        error_text = str(error_context.exception)
        self.assertNotIn("secret-key", error_text)
        self.assertNotIn("Authorization", error_text)
        self.assertIn("[redacted]", error_text)
        self.assertIn("[redacted-header]", error_text)


if __name__ == "__main__":
    unittest.main()
