"""run_eval profile 与 usage 字段单元测试。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_eval


class FakeLlmClient:
    """轻量 fake LLM client，仅记录 run_eval 注入的 model_profile 名称，不涉及真实 LLM 调用。"""

    created_profiles: list[str] = []

    def __init__(self, *, model_profile: str) -> None:
        self.profile_name = model_profile
        self.created_profiles.append(model_profile)


class FakeRunLogger:
    """提供 run_eval 所需 payload 的轻量 logger，从类级别模板复制 payload。"""

    payload_template: dict[str, object] = {}

    def __init__(self, repo_path: Path, user_task: str) -> None:
        self.repo_path = repo_path
        self.user_task = user_task
        self.payload = dict(self.payload_template)


class FakeAgent:
    """模拟 CodeAnalysisAgent，返回预设答案或在 answer() 时抛出异常。"""

    answer_text = "包含 target 的答案"
    raised_error: Exception | None = None

    def __init__(self, *, llm_client, repository_tools, run_logger) -> None:
        self.llm_client = llm_client
        self.repository_tools = repository_tools
        self.run_logger = run_logger

    def answer(self, question: str) -> str:
        del question
        if self.raised_error is not None:
            raise self.raised_error
        return self.answer_text


class TestRunEvalCurrentEntry(unittest.TestCase):
    """验证 run_eval CLI 使用当前 profile 与 v0.7 字段。"""

    def setUp(self) -> None:
        FakeLlmClient.created_profiles = []
        FakeRunLogger.payload_template = {
            "model_profile": "default",
            "provider": "mock",
            "model": "mock-model",
            "usage_summary": {
                "llm_call_count": 2,
                "total_latency_ms": 12.5,
                "total_tokens": 30,
                "estimated_tokens": 4,
                "estimated_cost": 0.02,
            },
        }
        FakeAgent.answer_text = "包含 target 的答案"
        FakeAgent.raised_error = None

    def test_default_profile_and_v07_fields_are_emitted(self) -> None:
        with self._patched_run_eval():
            summary, output = self._run_main_with_case(
                [{"question": "默认 profile?", "must_contain": ["target"]}],
                extra_args=[],
            )

        result_payload = self._first_result_payload(output)
        self.assertEqual(0, summary)
        self.assertEqual(["default"], FakeLlmClient.created_profiles)
        self.assertTrue(result_payload["passed"])
        self.assertEqual("default", result_payload["model_profile"])
        self.assertEqual("mock", result_payload["provider"])
        self.assertEqual("mock-model", result_payload["model"])
        self.assertEqual(2, result_payload["llm_call_count"])
        self.assertEqual(12.5, result_payload["total_latency_ms"])
        self.assertEqual(30, result_payload["total_tokens"])
        self.assertEqual(4, result_payload["estimated_tokens"])
        self.assertEqual(0.02, result_payload["estimated_cost"])
        self.assertIsNone(result_payload["failure_type"])

    def test_explicit_profile_is_passed_to_llm_client(self) -> None:
        FakeRunLogger.payload_template["model_profile"] = "fast"

        with self._patched_run_eval():
            exit_code, output = self._run_main_with_case(
                [{"question": "显式 profile?", "must_contain": ["target"]}],
                extra_args=["--model-profile", "fast"],
            )

        result_payload = self._first_result_payload(output)
        self.assertEqual(0, exit_code)
        self.assertEqual(["fast"], FakeLlmClient.created_profiles)
        self.assertEqual("fast", result_payload["model_profile"])

    def test_invalid_json_exception_uses_shared_failure_classifier(self) -> None:
        FakeAgent.raised_error = RuntimeError("模型输出不是严格 JSON")

        with self._patched_run_eval():
            exit_code, output = self._run_main_with_case(
                [{"question": "坏 JSON", "must_contain": ["target"]}],
                extra_args=[],
            )

        result_payload = self._first_result_payload(output)
        self.assertEqual(1, exit_code)
        self.assertFalse(result_payload["passed"])
        self.assertEqual("invalid_json", result_payload["failure_type"])
        self.assertEqual("default", result_payload["model_profile"])

    def test_provider_error_exception_uses_shared_failure_classifier(self) -> None:
        FakeAgent.raised_error = RuntimeError("provider error: rate limited")

        with self._patched_run_eval():
            exit_code, output = self._run_main_with_case(
                [{"question": "provider 错误", "must_contain": ["target"]}],
                extra_args=["--model-profile", "strong"],
            )

        result_payload = self._first_result_payload(output)
        self.assertEqual(1, exit_code)
        self.assertFalse(result_payload["passed"])
        self.assertEqual("provider_error", result_payload["failure_type"])
        self.assertEqual("strong", result_payload["model_profile"])

    def _patched_run_eval(self):
        return patch.multiple(
            run_eval,
            CodeAnalysisAgent=FakeAgent,
            LlmClient=FakeLlmClient,
            RunLogger=FakeRunLogger,
        )

    def _run_main_with_case(
        self,
        cases: list[dict[str, object]],
        *,
        extra_args: list[str],
    ) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_root = Path(temp_directory)
            eval_file = temp_root / "eval_case.json"
            eval_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            repo_path = temp_root / "repo"
            repo_path.mkdir()
            stdout = io.StringIO()
            argv = [
                "run_eval.py",
                "--repo",
                str(repo_path),
                "--eval-file",
                str(eval_file),
                *extra_args,
            ]
            with patch("sys.argv", argv), redirect_stdout(stdout):
                exit_code = run_eval.main()
            return exit_code, stdout.getvalue()

    def _first_result_payload(self, output: str) -> dict[str, object]:
        result_line = next(line for line in output.splitlines() if line.startswith("result_data: "))
        payload = json.loads(result_line.removeprefix("result_data: "))
        self.assertIsInstance(payload, dict)
        return payload


if __name__ == "__main__":
    unittest.main()
