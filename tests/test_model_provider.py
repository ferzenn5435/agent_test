"""模型 provider 数据模型单元测试。"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from model_provider import LLMResponse, ModelProvider, TokenUsage


REPO_ROOT = Path(__file__).resolve().parents[1]


class EchoProvider:
    """最小 fake provider：将第一条消息内容原样返回，用于验证 ModelProvider 协议。"""

    def call(self, messages: list[dict[str, str]], profile_name: str) -> LLMResponse:
        return LLMResponse(
            content=messages[0]["content"],
            provider="fake",
            model="fake-model",
            profile_name=profile_name,
            latency_ms=12.5,
            usage=TokenUsage(
                prompt_tokens=3,
                completion_tokens=4,
                total_tokens=7,
                estimated=True,
            ),
            raw={"id": "response-1"},
        )


class TestTokenUsage(unittest.TestCase):
    """验证 TokenUsage schema。"""

    def test_serializes_estimated_cost_none_without_zero_conversion(self) -> None:
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated=False,
            estimated_cost=None,
        )

        usage_dict = usage.to_dict()

        self.assertEqual(10, usage.prompt_tokens)
        self.assertEqual(5, usage.completion_tokens)
        self.assertEqual(15, usage.total_tokens)
        self.assertFalse(usage.estimated)
        self.assertIsNone(usage.estimated_cost)
        self.assertIsNone(usage_dict["estimated_cost"])
        json.dumps(usage_dict)


class TestLLMResponse(unittest.TestCase):
    """验证 LLMResponse schema。"""

    def test_preserves_required_response_fields(self) -> None:
        usage = TokenUsage(
            prompt_tokens=8,
            completion_tokens=13,
            total_tokens=21,
            estimated=True,
            estimated_cost=0.0021,
        )
        raw_payload: dict[str, object] = {
            "choices": [
                {"message": {"content": "hello"}},
            ],
        }
        response = LLMResponse(
            content="hello",
            provider="openai-compatible",
            model="gpt-test",
            profile_name="default",
            latency_ms=42.75,
            usage=usage,
            raw=raw_payload,
        )

        response_dict = response.to_dict()

        self.assertEqual("hello", response.content)
        self.assertEqual("openai-compatible", response.provider)
        self.assertEqual("gpt-test", response.model)
        self.assertEqual("default", response.profile_name)
        self.assertEqual(42.75, response.latency_ms)
        self.assertEqual(usage, response.usage)
        self.assertEqual(raw_payload, response.raw)
        self.assertEqual("hello", response_dict["content"])
        self.assertEqual("openai-compatible", response_dict["provider"])
        self.assertEqual("gpt-test", response_dict["model"])
        self.assertEqual("default", response_dict["profile_name"])
        self.assertEqual(42.75, response_dict["latency_ms"])
        self.assertEqual(usage.to_dict(), response_dict["usage"])
        self.assertEqual(raw_payload, response_dict["raw"])
        json.dumps(response_dict)


class TestModelProviderProtocol(unittest.TestCase):
    """验证 ModelProvider 协议形状。"""

    def test_runtime_protocol_accepts_provider_with_call_method(self) -> None:
        provider = EchoProvider()

        self.assertIsInstance(provider, ModelProvider)
        response = provider.call(
            messages=[{"role": "user", "content": "ping"}],
            profile_name="fast",
        )

        self.assertEqual("ping", response.content)
        self.assertEqual("fast", response.profile_name)

    def test_module_does_not_import_env_http_or_concrete_providers(self) -> None:
        with (REPO_ROOT / "model_provider.py").open(encoding="utf-8") as source_file:
            module_ast = ast.parse(source_file.read())

        imported_modules: set[str] = set()
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

        forbidden_imports = {
            "os",
            "json",
            "requests",
            "providers.openai_compatible",
            "providers.mock_provider",
        }
        self.assertTrue(forbidden_imports.isdisjoint(imported_modules))


if __name__ == "__main__":
    unittest.main()
