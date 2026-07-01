"""固定 LLM providers 单元测试。"""

from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from dataclasses import replace
from email.message import EmailMessage
from unittest.mock import patch

from config import ModelPricing, ModelProfile
from model_provider import LLMResponse, TokenUsage
from providers.mock_provider import MockProvider
from providers.openai_compatible import OpenAICompatibleProvider, ProviderError


class FakeHttpResponse:
    """模拟 urllib.response 的 context manager，用于 mock HTTP 响应。"""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return json.dumps(self.payload).encode("utf-8")


class TestOpenAICompatibleProvider(unittest.TestCase):
    """验证 OpenAI-compatible provider 行为。"""

    def setUp(self) -> None:
        self.profile = ModelProfile(
            name="fast",
            provider="openai_compatible",
            base_url="https://example.test/v1",
            api_key="secret-test-key",
            model="test-model",
            temperature=0.25,
            max_output_tokens=123,
            timeout_seconds=17,
            retry_count=0,
            pricing=ModelPricing(input_per_1m_tokens=1.0, output_per_1m_tokens=2.0),
        )
        self.messages = [{"role": "user", "content": "ping"}]

    def test_sends_chat_completion_payload_and_parses_usage(self) -> None:
        response_payload: dict[str, object] = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "pong"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }
        captured_requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHttpResponse:
            captured_requests.append((request, timeout))
            return FakeHttpResponse(response_payload)

        with patch("providers.openai_compatible.urlopen", side_effect=fake_urlopen):
            with patch("providers.openai_compatible.perf_counter", side_effect=[1.0, 1.25]):
                llm_response = OpenAICompatibleProvider(self.profile).call(
                    self.messages,
                    profile_name="ignored-profile-name",
                )

        self.assertEqual("pong", llm_response.content)
        self.assertEqual("openai_compatible", llm_response.provider)
        self.assertEqual("test-model", llm_response.model)
        self.assertEqual("fast", llm_response.profile_name)
        self.assertEqual(250.0, llm_response.latency_ms)
        self.assertEqual(10, llm_response.usage.prompt_tokens)
        self.assertEqual(3, llm_response.usage.completion_tokens)
        self.assertEqual(13, llm_response.usage.total_tokens)
        self.assertFalse(llm_response.usage.estimated)
        self.assertEqual(0.000016, llm_response.usage.estimated_cost)
        self.assertEqual(response_payload, llm_response.raw)
        self.assertEqual(1, len(captured_requests))
        request, timeout = captured_requests[0]
        self.assertEqual(17, timeout)
        self.assertEqual("https://example.test/v1/chat/completions", request.full_url)
        self.assertEqual("Bearer secret-test-key", request.headers["Authorization"])
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            {
                "model": "test-model",
                "messages": self.messages,
                "temperature": 0.25,
                "max_tokens": 123,
            },
            request_payload,
        )

    def test_retry_count_zero_makes_exactly_one_attempt(self) -> None:
        with patch("providers.openai_compatible.urlopen", side_effect=urllib.error.URLError("temporary")) as urlopen_mock:
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider(self.profile).call(self.messages, profile_name="fast")

        self.assertEqual(1, urlopen_mock.call_count)

    def test_retries_network_timeout_and_retryable_http_errors(self) -> None:
        retrying_profile = replace(self.profile, retry_count=3)
        http_error = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=EmailMessage(),
            fp=None,
        )
        response_payload = {"choices": [{"message": {"content": "ok"}}]}

        with patch(
            "providers.openai_compatible.urlopen",
            side_effect=[
                urllib.error.URLError("network down"),
                socket.timeout("slow"),
                http_error,
                FakeHttpResponse(response_payload),
            ],
        ) as urlopen_mock:
            response = OpenAICompatibleProvider(retrying_profile).call(
                self.messages,
                profile_name="fast",
            )

        self.assertEqual("ok", response.content)
        self.assertEqual(4, urlopen_mock.call_count)
        self.assertTrue(response.usage.estimated)

    def test_does_not_retry_non_retryable_http_error_and_redacts_secret(self) -> None:
        retrying_profile = replace(self.profile, retry_count=3)
        http_error = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=EmailMessage(),
            fp=None,
        )

        with patch("providers.openai_compatible.urlopen", side_effect=http_error) as urlopen_mock:
            with self.assertRaises(ProviderError) as error_context:
                OpenAICompatibleProvider(retrying_profile).call(
                    self.messages,
                    profile_name="fast",
                )

        error_text = str(error_context.exception)
        self.assertEqual(1, urlopen_mock.call_count)
        self.assertIn("HTTP 错误 400", error_text)
        self.assertNotIn("secret-test-key", error_text)
        self.assertNotIn("Authorization", error_text)

    def test_missing_provider_usage_is_estimated_with_profile_pricing(self) -> None:
        response_payload = {"choices": [{"message": {"content": "abcdefghi"}}]}

        with patch("providers.openai_compatible.urlopen", return_value=FakeHttpResponse(response_payload)):
            response = OpenAICompatibleProvider(self.profile).call(
                messages=[{"role": "user", "content": "abcde"}],
                profile_name="fast",
            )

        self.assertEqual(2, response.usage.prompt_tokens)
        self.assertEqual(3, response.usage.completion_tokens)
        self.assertEqual(5, response.usage.total_tokens)
        self.assertTrue(response.usage.estimated)
        self.assertEqual(0.000008, response.usage.estimated_cost)


class TestMockProvider(unittest.TestCase):
    """验证 mock provider 可配置且不会访问网络。"""

    def test_returns_configurable_response(self) -> None:
        provider = MockProvider(
            content="mock text",
            model="mock-model",
            profile_name="mock-profile",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            raw={"mock": True},
            latency_ms=4.5,
        )

        with patch("urllib.request.urlopen", side_effect=AssertionError("network called")):
            response = provider.call([{"role": "user", "content": "ignored"}], "ignored")

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual("mock text", response.content)
        self.assertEqual("mock_provider", response.provider)
        self.assertEqual("mock-model", response.model)
        self.assertEqual("mock-profile", response.profile_name)
        self.assertEqual(4.5, response.latency_ms)
        self.assertEqual({"mock": True}, response.raw)
        self.assertEqual(3, response.usage.total_tokens)

    def test_can_simulate_provider_error_timeout_and_invalid_json_content(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mock provider_error"):
            MockProvider(simulate="provider_error").call([], "mock")

        with self.assertRaisesRegex(TimeoutError, "mock timeout"):
            MockProvider(simulate="timeout").call([], "mock")

        response = MockProvider(simulate="invalid_json").call([], "mock")
        self.assertEqual("{invalid json", response.content)


if __name__ == "__main__":
    unittest.main()
