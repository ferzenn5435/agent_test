"""Deterministic verifier unit tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_state import RunState
from schemas import MustContainRule, PlanStep, TaskPlan, VerificationResult, VerificationSpec
from verifier import (
    REASON_CONTEXT_BUDGET_EXCEEDED,
    REASON_FORBIDDEN_CHANGED_FILE,
    REASON_MAX_STEPS_EXCEEDED,
    REASON_MUST_CONTAIN_MISSING,
    REASON_NOT_FINISHED,
    REASON_PATCH_NOT_CONFIRMED,
    REASON_TESTS_FAILED,
    verify_run_state,
)


def _write_repo(repo_root: Path) -> None:
    (repo_root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo_root / "README.md").write_text(
        "This project validates safe editing ability.\n",
        encoding="utf-8",
    )


def _plan() -> TaskPlan:
    return TaskPlan(
        task_type="edit",
        risk_level="low",
        max_steps=3,
        steps=(
            PlanStep(id="inspect", title="Inspect context"),
            PlanStep(id="verify", title="Run verification"),
        ),
        requires_patch=True,
        requires_tests=True,
        expected_changed_files=("app.py",),
        verification=(
            VerificationSpec(
                must_contain=(
                    MustContainRule(
                        path="README.md",
                        strings=("safe editing ability",),
                    ),
                )
            ),
        ),
    )


def _finished_state(
    *,
    changed_files: tuple[str, ...] = ("app.py",),
    tests_result: VerificationResult | None = None,
    stage: str = "FINISH",
    execution_steps: tuple[str, ...] = ("inspect", "verify"),
    context_stats: dict[str, object] | None = None,
) -> RunState:
    return RunState(
        stage=stage,
        stage_history=("INIT", "PLAN", "EXECUTE", "VERIFY", stage),
        plan=_plan(),
        execution_steps=execution_steps,
        changed_files=changed_files,
        tests_result=tests_result,
        context_stats=context_stats,
    )


def _awaiting_approval_state() -> RunState:
    return RunState(
        stage="AWAITING_APPROVAL",
        stage_history=("INIT", "PLAN", "EXECUTE", "AWAITING_APPROVAL"),
        plan=_plan(),
        execution_steps=("inspect",),
        changed_files=(),
        tests_result=None,
    )


def _finished_pending_approval_state() -> RunState:
    return RunState(
        stage="FINISH",
        stage_history=("INIT", "PLAN", "EXECUTE", "AWAITING_APPROVAL", "FINISH"),
        plan=_plan(),
        execution_steps=("inspect",),
        changed_files=(),
        tests_result=None,
    )


class TestDeterministicVerifier(unittest.TestCase):
    def test_verification_passes_with_default_plan_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
                context_stats={
                    "full_file_reads": ["README.md"],
                    "total_tool_output_chars": 10,
                },
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertTrue(result.passed, result.output)
            self.assertEqual((), result.reasons)
            self.assertEqual(0, result.exit_code)
            self.assertEqual("verification passed", result.output)
            self.assertFalse(result.repairable)

    def test_allowed_changed_files_rejects_unauthorized_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                changed_files=("other.py",),
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_FORBIDDEN_CHANGED_FILE, result.reasons)
            self.assertIn("changed_files unauthorized: other.py", result.output)

    def test_allowed_changed_files_rejects_invalid_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                changed_files=("..\\evil.py",),
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_FORBIDDEN_CHANGED_FILE, result.reasons)
            self.assertIn("changed_files invalid", result.output)

    def test_must_contain_passes_existing_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertTrue(result.passed, result.output)
            self.assertNotIn(REASON_MUST_CONTAIN_MISSING, result.reasons)

    def test_must_contain_rejects_missing_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "README.md").write_text("missing the required phrase\n", encoding="utf-8")
            (repo_root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_MUST_CONTAIN_MISSING, result.reasons)
            self.assertIn("must_contain missing text: README.md", result.output)

    def test_tests_failed_is_repairable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=False,
                    exit_code=1,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_TESTS_FAILED, result.reasons)
            self.assertTrue(result.repairable)

    def test_patch_not_confirmed_is_not_repairable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=False)

            self.assertFalse(result.passed)
            self.assertIn(REASON_PATCH_NOT_CONFIRMED, result.reasons)
            self.assertFalse(result.repairable)

    def test_pending_approval_with_pending_patch_is_successful_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _awaiting_approval_state()

            result = verify_run_state(
                run_state,
                repo_root,
                pending_patch={
                    "status": "pending_approval",
                    "patch_id": "20260630_120000_abcdef123456",
                    "target_files": [{"path": "app.py", "operation": "modify"}],
                },
            )

            self.assertTrue(result.passed, result.output)
            self.assertEqual((), result.reasons)
            self.assertNotIn(REASON_NOT_FINISHED, result.reasons)
            self.assertNotIn(REASON_TESTS_FAILED, result.reasons)
            self.assertNotIn(REASON_PATCH_NOT_CONFIRMED, result.reasons)

    def test_finished_after_awaiting_approval_with_pending_patch_is_successful_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_pending_approval_state()

            result = verify_run_state(
                run_state,
                repo_root,
                pending_patch={
                    "status": "pending_approval",
                    "patch_id": "20260630_120000_abcdef123456",
                    "target_files": [{"path": "app.py", "operation": "modify"}],
                },
            )

            self.assertTrue(result.passed, result.output)
            self.assertEqual((), result.reasons)
            self.assertNotIn(REASON_TESTS_FAILED, result.reasons)
            self.assertNotIn(REASON_PATCH_NOT_CONFIRMED, result.reasons)

    def test_finished_after_awaiting_approval_with_invalid_pending_patch_uses_normal_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_pending_approval_state()

            result = verify_run_state(
                run_state,
                repo_root,
                pending_patch={"status": "pending_approval", "patch_id": ""},
            )

            self.assertFalse(result.passed)
            self.assertIn(REASON_TESTS_FAILED, result.reasons)
            self.assertIn(REASON_PATCH_NOT_CONFIRMED, result.reasons)

    def test_awaiting_approval_without_pending_patch_is_not_finished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)

            result = verify_run_state(_awaiting_approval_state(), repo_root)

            self.assertFalse(result.passed)
            self.assertIn(REASON_NOT_FINISHED, result.reasons)
            self.assertIn(REASON_TESTS_FAILED, result.reasons)
            self.assertIn(REASON_PATCH_NOT_CONFIRMED, result.reasons)

    def test_context_budget_exceeded_for_full_file_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
                context_stats={
                    "full_file_reads": ["README.md"],
                    "total_tool_output_chars": 10,
                },
            )

            result = verify_run_state(
                run_state,
                repo_root,
                must_not_read_full_files=("README.md",),
                patch_confirmed=True,
            )

            self.assertFalse(result.passed)
            self.assertIn(REASON_CONTEXT_BUDGET_EXCEEDED, result.reasons)
            self.assertIn("must_not_read_full_files violated", result.output)

    def test_context_budget_exceeded_for_tool_output_chars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
                context_stats={
                    "full_file_reads": [],
                    "total_tool_output_chars": 99,
                },
            )

            result = verify_run_state(
                run_state,
                repo_root,
                max_total_tool_output_chars=10,
                patch_confirmed=True,
            )

            self.assertFalse(result.passed)
            self.assertIn(REASON_CONTEXT_BUDGET_EXCEEDED, result.reasons)
            self.assertIn("max_total_tool_output_chars violated", result.output)

    def test_combined_failures_keep_stable_reason_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "README.md").write_text("missing required text\n", encoding="utf-8")
            (repo_root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            run_state = _finished_state(
                changed_files=("other.py",),
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=False,
                    exit_code=1,
                    output="failed",
                ),
                context_stats={
                    "full_file_reads": ["README.md"],
                    "total_tool_output_chars": 99,
                },
            )

            result = verify_run_state(
                run_state,
                repo_root,
                must_not_read_full_files=("README.md",),
                max_total_tool_output_chars=10,
                patch_confirmed=False,
            )

            self.assertFalse(result.passed)
            self.assertEqual(
                (
                    "forbidden_changed_file",
                    "must_contain_missing",
                    "tests_failed",
                    "patch_not_confirmed",
                    "context_budget_exceeded",
                ),
                result.reasons,
            )
            self.assertFalse(result.repairable)

    def test_not_finished_rejects_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                stage="VERIFY",
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_NOT_FINISHED, result.reasons)

    def test_max_steps_exceeded_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_repo(repo_root)
            run_state = _finished_state(
                execution_steps=("inspect", "verify", "repair", "extra"),
                tests_result=VerificationResult(
                    command_name="unit",
                    passed=True,
                    exit_code=0,
                ),
            )

            result = verify_run_state(run_state, repo_root, patch_confirmed=True)

            self.assertFalse(result.passed)
            self.assertIn(REASON_MAX_STEPS_EXCEEDED, result.reasons)
            self.assertIn("execution_steps=4 limit=3", result.output)

    def test_verification_result_serializes_new_fields(self) -> None:
        result = VerificationResult(
            command_name="deterministic_verifier",
            passed=False,
            exit_code=1,
            output="tests_failed",
            reasons=("tests_failed",),
            repairable=True,
        )

        result_dict = result.to_dict()

        self.assertEqual(
            {
                "command_name": "deterministic_verifier",
                "passed": False,
                "exit_code": 1,
                "output": "tests_failed",
                "reasons": ["tests_failed"],
                "repairable": True,
            },
            result_dict,
        )
        json.dumps(result_dict)


if __name__ == "__main__":
    unittest.main()
