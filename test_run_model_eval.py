"""多模型 eval runner 的单元测试。"""

from __future__ import annotations

import csv
import json
import threading
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import run_model_eval
from config import ConfigError
from eval_runner import EditEvalResult


@dataclass(frozen=True)
class FakeCase:
    """最小 fake eval case，只提供 runner 所需的 id 字段。"""

    id: str


def _eval_result(
    case_id: str,
    *,
    passed: bool,
    steps: int = 2,
    total_latency_ms: float = 100.0,
    total_tokens: int = 50,
    estimated_cost: float | None = 0.01,
    failure_type: str | None = None,
    model_profile: str | None = None,
) -> EditEvalResult:
    reasons = () if passed else ("模型输出不是严格 JSON",)
    return EditEvalResult(
        case_id=case_id,
        passed=passed,
        reasons=reasons,
        changed_files=(),
        steps=steps,
        final_answer="ok" if passed else None,
        error=None if passed else "invalid json",
        test_results=None,
        context_stats=None,
        model_profile=model_profile,
        provider="mock",
        model="mock-model",
        llm_call_count=1,
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        estimated_tokens=0,
        estimated_cost=estimated_cost,
        failure_type=failure_type,
    )


class TestRunModelEval(unittest.TestCase):
    """验证多 profile/case/trial 聚合与输出。"""

    def test_runs_profile_case_trial_matrix_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            project_root = Path(temp_directory)
            cases_path = project_root / "cases.json"
            cases_path.write_text("[]", encoding="utf-8")
            calls: list[tuple[str, str]] = []
            fake_cases = [FakeCase("case-a"), FakeCase("case-b")]

            def fake_run_edit_case(case: FakeCase, project_root: Path, llm_client_factory):
                llm_client = llm_client_factory(case)
                calls.append((case.id, llm_client.profile_name))
                return _eval_result(
                    case.id,
                    passed=case.id == "case-a",
                    steps=3 if case.id == "case-a" else 5,
                    total_latency_ms=30.0 if case.id == "case-a" else 70.0,
                    total_tokens=10 if case.id == "case-a" else 30,
                    estimated_cost=0.1 if case.id == "case-a" else 0.3,
                    failure_type=None if case.id == "case-a" else "invalid_json",
                    model_profile=llm_client.profile_name,
                )

            with patch.object(run_model_eval, "load_edit_cases", return_value=fake_cases), patch.object(
                run_model_eval,
                "run_edit_case",
                side_effect=fake_run_edit_case,
            ), patch.object(run_model_eval, "LlmClient", side_effect=FakeProfileClient):
                output_dir = run_model_eval.run_model_eval(
                    cases_path=cases_path,
                    repo_root=project_root,
                    profiles=("fast", "strong"),
                    trials=2,
                    timestamp="20260630_120000",
                )

            self.assertEqual(project_root / ".repopilot" / "model_evals" / "20260630_120000", output_dir)
            self.assertEqual(
                [
                    ("case-a", "fast"),
                    ("case-b", "fast"),
                    ("case-a", "fast"),
                    ("case-b", "fast"),
                    ("case-a", "strong"),
                    ("case-b", "strong"),
                    ("case-a", "strong"),
                    ("case-b", "strong"),
                ],
                calls,
            )

            results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(8, len(results_payload["results"]))
            first_result = results_payload["results"][0]
            self.assertEqual("fast", first_result["profile"])
            self.assertEqual("fast", first_result["model_profile"])
            self.assertEqual("case-a", first_result["case_id"])
            self.assertEqual(1, first_result["trial_index"])
            self.assertTrue(first_result["passed"])

            with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(["fast", "strong"], [row["profile"] for row in summary_rows])
            self.assertEqual("0.500000", summary_rows[0]["pass_rate"])
            self.assertEqual("4.000000", summary_rows[0]["avg_steps"])
            self.assertEqual("50.000000", summary_rows[0]["avg_latency_ms"])
            self.assertEqual("20.000000", summary_rows[0]["avg_tokens"])
            self.assertEqual("0.200000", summary_rows[0]["avg_estimated_cost"])
            self.assertEqual('{"invalid_json": 2}', summary_rows[0]["failure_type_counts"])

            summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| profile | total | passed | pass_rate |", summary_md)
            self.assertIn("| fast | 4 | 2 | 0.500000 |", summary_md)

    def test_trial_exception_becomes_failed_result_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            project_root = Path(temp_directory)
            cases_path = project_root / "cases.json"
            cases_path.write_text("[]", encoding="utf-8")
            fake_cases = [FakeCase("case-a"), FakeCase("case-b")]
            call_count = 0

            def fake_run_edit_case(case: FakeCase, project_root: Path, llm_client_factory):
                nonlocal call_count
                call_count += 1
                if case.id == "case-a":
                    raise RuntimeError("provider error: rate limited")
                return _eval_result(case.id, passed=True, model_profile="fast")

            with patch.object(run_model_eval, "load_edit_cases", return_value=fake_cases), patch.object(
                run_model_eval,
                "run_edit_case",
                side_effect=fake_run_edit_case,
            ), patch.object(run_model_eval, "LlmClient", side_effect=FakeProfileClient):
                output_dir = run_model_eval.run_model_eval(
                    cases_path=cases_path,
                    repo_root=project_root,
                    profiles=("fast",),
                    trials=1,
                    timestamp="20260630_121000",
                )

            self.assertEqual(2, call_count)
            results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            failed_result = results_payload["results"][0]
            passed_result = results_payload["results"][1]
            self.assertFalse(failed_result["passed"])
            self.assertEqual("provider_error", failed_result["failure_type"])
            self.assertIn("case 执行异常", failed_result["reasons"][0])
            self.assertTrue(passed_result["passed"])

    def test_profile_preflight_failure_writes_failed_rows_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            project_root = Path(temp_directory)
            cases_path = project_root / "cases.json"
            cases_path.write_text("[]", encoding="utf-8")
            fake_cases = [FakeCase("case-a"), FakeCase("case-b")]
            calls: list[tuple[str, str]] = []

            def fake_llm_client(*, model_profile: str) -> FakeProfileClient:
                if model_profile == "broken":
                    raise ConfigError("缺少 model profile broken 必要环境变量: BROKEN_LLM_API_KEY")
                return FakeProfileClient(model_profile=model_profile)

            def fake_run_edit_case(case: FakeCase, project_root: Path, llm_client_factory):
                llm_client = llm_client_factory(case)
                calls.append((case.id, llm_client.profile_name))
                return _eval_result(case.id, passed=True, model_profile=llm_client.profile_name)

            with patch.object(run_model_eval, "load_edit_cases", return_value=fake_cases), patch.object(
                run_model_eval,
                "run_edit_case",
                side_effect=fake_run_edit_case,
            ), patch.object(run_model_eval, "LlmClient", side_effect=fake_llm_client):
                output_dir = run_model_eval.run_model_eval(
                    cases_path=cases_path,
                    repo_root=project_root,
                    profiles=("broken", "fast"),
                    trials=2,
                    timestamp="20260701_130000",
                )

            self.assertTrue((output_dir / "results.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "summary.md").is_file())
            self.assertEqual(
                [
                    ("case-a", "fast"),
                    ("case-b", "fast"),
                    ("case-a", "fast"),
                    ("case-b", "fast"),
                ],
                calls,
            )

            results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(8, len(results_payload["results"]))
            required_fields = {
                "case_id",
                "passed",
                "reasons",
                "changed_files",
                "steps",
                "final_answer",
                "error",
                "test_results",
                "context_stats",
                "model_profile",
                "provider",
                "model",
                "llm_call_count",
                "total_latency_ms",
                "total_tokens",
                "estimated_tokens",
                "estimated_cost",
                "failure_type",
                "profile",
                "trial_index",
            }
            broken_results = [row for row in results_payload["results"] if row["profile"] == "broken"]
            fast_results = [row for row in results_payload["results"] if row["profile"] == "fast"]
            self.assertEqual(4, len(broken_results))
            self.assertEqual(4, len(fast_results))
            self.assertTrue(all(required_fields.issubset(row) for row in broken_results))
            self.assertTrue(all(row["passed"] is False for row in broken_results))
            self.assertEqual({"case-a", "case-b"}, {row["case_id"] for row in broken_results})
            self.assertEqual({1, 2}, {row["trial_index"] for row in broken_results})
            self.assertTrue(all(row["llm_call_count"] == 0 for row in broken_results))
            self.assertTrue(all(row["failure_type"] == "unknown" for row in broken_results))
            self.assertTrue(all("profile 配置预检失败" in row["reasons"][0] for row in broken_results))
            self.assertTrue(all(row["passed"] is True for row in fast_results))

            with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(["broken", "fast"], [row["profile"] for row in summary_rows])
            self.assertEqual("4", summary_rows[0]["total"])
            self.assertEqual("0", summary_rows[0]["passed"])
            self.assertEqual('{"unknown": 4}', summary_rows[0]["failure_type_counts"])
            self.assertEqual("4", summary_rows[1]["passed"])

    def test_trial_timeout_becomes_failed_result_and_reports_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            project_root = Path(temp_directory)
            cases_path = project_root / "cases.json"
            cases_path.write_text("[]", encoding="utf-8")
            fake_cases = [FakeCase("slow-case"), FakeCase("fast-case")]
            slow_case_started = threading.Event()
            release_slow_case = threading.Event()

            def fake_run_edit_case(case: FakeCase, project_root: Path, llm_client_factory):
                if case.id == "slow-case":
                    slow_case_started.set()
                    release_slow_case.wait()
                    return _eval_result(case.id, passed=True, model_profile="fast")
                return _eval_result(case.id, passed=True, model_profile="fast")

            with patch.object(run_model_eval, "load_edit_cases", return_value=fake_cases), patch.object(
                run_model_eval,
                "run_edit_case",
                side_effect=fake_run_edit_case,
            ), patch.object(run_model_eval, "LlmClient", side_effect=FakeProfileClient):
                output_dir = run_model_eval.run_model_eval(
                    cases_path=cases_path,
                    repo_root=project_root,
                    profiles=("fast",),
                    trials=1,
                    trial_timeout_seconds=0.01,
                    timestamp="20260701_140000",
                )

            release_slow_case.set()
            self.assertTrue(slow_case_started.is_set())
            self.assertTrue((output_dir / "results.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "summary.md").is_file())

            results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len(results_payload["results"]))
            timeout_result = results_payload["results"][0]
            passed_result = results_payload["results"][1]
            self.assertFalse(timeout_result["passed"])
            self.assertEqual("timeout", timeout_result["failure_type"])
            self.assertIn("case 执行超时", timeout_result["reasons"][0])
            self.assertEqual("fast", timeout_result["profile"])
            self.assertEqual(1, timeout_result["trial_index"])
            self.assertTrue(passed_result["passed"])

            with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual("2", summary_rows[0]["total"])
            self.assertEqual("1", summary_rows[0]["passed"])
            self.assertEqual('{"timeout": 1}', summary_rows[0]["failure_type_counts"])

    def test_parse_args_requires_positive_trials(self) -> None:
        with self.assertRaises(SystemExit):
            run_model_eval.parse_args(["--trials", "0"])

    def test_parse_args_requires_positive_trial_timeout(self) -> None:
        with self.assertRaises(SystemExit):
            run_model_eval.parse_args(["--trial-timeout-seconds", "0"])


class FakeProfileClient:
    """最小 fake provider，仅记录 runner 注入的 model_profile 名称，不做网络请求。"""

    def __init__(self, *, model_profile: str) -> None:
        self.profile_name = model_profile


if __name__ == "__main__":
    unittest.main()
