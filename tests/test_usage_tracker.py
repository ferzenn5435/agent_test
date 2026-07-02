"""LLM usage 纯计算模块测试。"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from config import ModelPricing
from model_provider import TokenUsage
from usage_tracker import UsageCallRecord, normalize_call_usage, summarize_usage_calls


REPO_ROOT = Path(__file__).resolve().parents[1]


class DuckPricing:
    """鸭子类型 pricing 对象，用于验证 normalizer 接受相同字段对象。"""

    input_per_1m_tokens = 1.0
    output_per_1m_tokens = 2.0


class TestNormalizeCallUsage(unittest.TestCase):
    """验证单次调用 usage 标准化。"""

    def test_trusts_provider_usage_and_calculates_cost(self) -> None:
        usage = normalize_call_usage(
            prompt_text="ignored prompt text",
            completion_text="ignored completion text",
            provider_usage=TokenUsage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                estimated=True,
            ),
            pricing=DuckPricing(),
        )

        self.assertEqual(1000, usage.prompt_tokens)
        self.assertEqual(500, usage.completion_tokens)
        self.assertEqual(1500, usage.total_tokens)
        self.assertFalse(usage.estimated)
        self.assertEqual(0.002, usage.estimated_cost)
        json.dumps(usage.to_dict())

    def test_estimates_missing_usage_from_character_count(self) -> None:
        usage = normalize_call_usage(
            prompt_text="abcde",
            completion_text="abcdefghi",
            provider_usage=None,
            pricing=ModelPricing(input_per_1m_tokens=1.0, output_per_1m_tokens=2.0),
        )

        self.assertEqual(2, usage.prompt_tokens)
        self.assertEqual(3, usage.completion_tokens)
        self.assertEqual(5, usage.total_tokens)
        self.assertTrue(usage.estimated)
        self.assertEqual(0.000008, usage.estimated_cost)

    def test_null_pricing_keeps_cost_none(self) -> None:
        usage = normalize_call_usage(
            prompt_text="abcde",
            completion_text="abcdefghi",
            provider_usage=None,
            pricing=None,
        )

        self.assertEqual(5, usage.total_tokens)
        self.assertTrue(usage.estimated)
        self.assertIsNone(usage.estimated_cost)
        self.assertIsNone(usage.to_dict()["estimated_cost"])


class TestUsageSummary(unittest.TestCase):
    """验证多次调用 usage 汇总。"""

    def test_accumulates_multiple_calls_with_stable_dict(self) -> None:
        summary = summarize_usage_calls(
            [
                UsageCallRecord(
                    latency_ms=10.5,
                    usage=TokenUsage(
                        prompt_tokens=1000,
                        completion_tokens=500,
                        total_tokens=1500,
                        estimated=False,
                        estimated_cost=0.002,
                    ),
                ),
                UsageCallRecord(
                    latency_ms=20.25,
                    usage=TokenUsage(
                        prompt_tokens=2,
                        completion_tokens=3,
                        total_tokens=5,
                        estimated=True,
                        estimated_cost=0.000008,
                    ),
                ),
            ]
        )

        summary_dict = summary.to_dict()

        self.assertEqual(2, summary.llm_call_count)
        self.assertEqual(30.75, summary.total_latency_ms)
        self.assertEqual(1002, summary.prompt_tokens)
        self.assertEqual(503, summary.completion_tokens)
        self.assertEqual(1505, summary.total_tokens)
        self.assertEqual(5, summary.estimated_tokens)
        self.assertEqual(0.002008, summary.estimated_cost)
        self.assertEqual(
            {
                "llm_call_count": 2,
                "total_latency_ms": 30.75,
                "prompt_tokens": 1002,
                "completion_tokens": 503,
                "total_tokens": 1505,
                "estimated_tokens": 5,
                "estimated_cost": 0.002008,
            },
            summary_dict,
        )
        json.dumps(summary_dict)

    def test_summary_cost_is_none_when_any_call_cost_is_unknown(self) -> None:
        summary = summarize_usage_calls(
            [
                UsageCallRecord(
                    latency_ms=1.0,
                    usage=TokenUsage(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                        estimated=False,
                        estimated_cost=0.1,
                    ),
                ),
                UsageCallRecord(
                    latency_ms=2.0,
                    usage=TokenUsage(
                        prompt_tokens=2,
                        completion_tokens=2,
                        total_tokens=4,
                        estimated=False,
                        estimated_cost=None,
                    ),
                ),
            ]
        )

        self.assertIsNone(summary.estimated_cost)
        self.assertIsNone(summary.to_dict()["estimated_cost"])

    def test_module_does_not_import_side_effect_dependencies(self) -> None:
        with (REPO_ROOT / "usage_tracker.py").open(encoding="utf-8") as source_file:
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
            "logging",
            "pathlib",
            "config.load_model_profile",
        }
        self.assertTrue(forbidden_imports.isdisjoint(imported_modules))


if __name__ == "__main__":
    unittest.main()
