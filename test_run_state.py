"""v0.6 schema 与运行状态机单元测试。"""

from __future__ import annotations

import json
import unittest

from run_state import RunState, RunStateError
from schemas import (
    MustContainRule,
    PlanStep,
    SchemaValidationError,
    TaskPlan,
    VerificationResult,
    VerificationSpec,
)


def _verification_spec() -> VerificationSpec:
    return VerificationSpec(
        must_contain=(
            MustContainRule(path="README.md", strings=("安全编辑能力",)),
        )
    )


def _valid_plan(task_type: str = "analysis") -> TaskPlan:
    verification = (_verification_spec(),) if task_type in {"edit", "refactor"} else ()
    return TaskPlan(
        task_type=task_type,
        risk_level="low",
        max_steps=3,
        steps=(
            PlanStep(id="inspect", title="检查上下文"),
            PlanStep(id="verify", title="运行验证"),
        ),
        verification=verification,
    )


class TestTaskPlanSchemaV06(unittest.TestCase):
    """验证 v0.6 TaskPlan schema 硬约束。"""

    def test_valid_task_plan_passes_and_serializes(self) -> None:
        task_plan = TaskPlan(
            task_type="edit",
            risk_level="low",
            max_steps=3,
            steps=(
                PlanStep(id="inspect", title="检查上下文"),
                PlanStep(id="verify", title="运行验证"),
            ),
            requires_patch=True,
            requires_tests=True,
            expected_changed_files=("docs\\guide.md",),
            verification=(_verification_spec(),),
        )

        plan_dict = task_plan.to_dict()

        self.assertEqual(
            [
                "task_type",
                "risk_level",
                "max_steps",
                "requires_patch",
                "requires_tests",
                "expected_changed_files",
                "steps",
                "verification",
            ],
            list(plan_dict.keys()),
        )
        self.assertEqual("edit", plan_dict["task_type"])
        self.assertEqual("low", plan_dict["risk_level"])
        self.assertTrue(plan_dict["requires_patch"])
        self.assertTrue(plan_dict["requires_tests"])
        self.assertEqual(["docs/guide.md"], plan_dict["expected_changed_files"])
        steps = plan_dict["steps"]
        if not isinstance(steps, list):
            raise AssertionError("steps 必须序列化为 list")
        self.assertEqual(2, len(steps))
        json.dumps(plan_dict)

    def test_verification_spec_serializes_must_contain_shape(self) -> None:
        verification_spec = VerificationSpec(
            must_contain=(
                MustContainRule(path="app.py", strings=("def add", "return a + b")),
            )
        )

        spec_dict = verification_spec.to_dict()

        self.assertEqual(
            {
                "must_contain": [
                    {
                        "path": "app.py",
                        "strings": ["def add", "return a + b"],
                    }
                ]
            },
            spec_dict,
        )
        json.dumps(spec_dict)

    def test_invalid_task_type_rejects(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "task_type"):
            TaskPlan(
                task_type="write",
                risk_level="low",
                max_steps=1,
                steps=(PlanStep(id="one", title="一步"),),
            )

    def test_invalid_risk_level_rejects(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "risk_level"):
            TaskPlan(
                task_type="analysis",
                risk_level="critical",
                max_steps=1,
                steps=(PlanStep(id="one", title="一步"),),
            )

    def test_duplicate_step_id_rejects(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "step id 重复"):
            TaskPlan(
                task_type="analysis",
                risk_level="low",
                max_steps=2,
                steps=(
                    PlanStep(id="same", title="第一步"),
                    PlanStep(id="same", title="第二步"),
                ),
            )

    def test_steps_greater_than_max_steps_rejects(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "max_steps"):
            TaskPlan(
                task_type="analysis",
                risk_level="low",
                max_steps=1,
                steps=(
                    PlanStep(id="one", title="第一步"),
                    PlanStep(id="two", title="第二步"),
                ),
            )

    def test_edit_and_refactor_without_verification_rejects(self) -> None:
        for task_type in ["edit", "refactor"]:
            with self.subTest(task_type=task_type):
                with self.assertRaisesRegex(SchemaValidationError, "verification"):
                    TaskPlan(
                        task_type=task_type,
                        risk_level="medium",
                        max_steps=1,
                        steps=(PlanStep(id="one", title="一步"),),
                    )

    def test_requires_patch_without_expected_changed_files_rejects(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "expected_changed_files"):
            TaskPlan(
                task_type="analysis",
                risk_level="low",
                max_steps=1,
                steps=(PlanStep(id="one", title="一步"),),
                requires_patch=True,
            )


class TestRunStateV06(unittest.TestCase):
    """验证 v0.6 RunState 状态机硬约束。"""

    def test_legal_init_plan_execute_verify_finish_transition_passes(self) -> None:
        run_state = RunState()
        run_state = run_state.transition("PLAN").with_plan(_valid_plan())
        run_state = run_state.transition("EXECUTE")
        run_state = run_state.with_execution_step("完成 inspect")
        run_state = run_state.transition("VERIFY")
        run_state = run_state.with_tests_result(
            VerificationResult(command_name="unit", passed=True, exit_code=0)
        )
        run_state = run_state.transition("FINISH")

        state_dict = run_state.to_dict()

        self.assertEqual("FINISH", run_state.stage)
        self.assertEqual(("INIT", "PLAN", "EXECUTE", "VERIFY", "FINISH"), run_state.stage_history)
        self.assertEqual(["INIT", "PLAN", "EXECUTE", "VERIFY", "FINISH"], state_dict["stage_history"])
        self.assertEqual(0, state_dict["repair_attempts"])
        json.dumps(state_dict)

    def test_skipping_plan_or_verify_rejects(self) -> None:
        with self.assertRaisesRegex(RunStateError, "INIT -> EXECUTE"):
            RunState().transition("EXECUTE")

        run_state = RunState().transition("PLAN").with_plan(_valid_plan())
        run_state = run_state.transition("EXECUTE")
        with self.assertRaisesRegex(RunStateError, "EXECUTE -> FINISH"):
            run_state.transition("FINISH")

    def test_execute_awaiting_approval_finish_transition_passes(self) -> None:
        run_state = RunState().transition("PLAN").with_plan(_valid_plan("edit"))
        run_state = run_state.transition("EXECUTE")
        run_state = run_state.with_execution_step("提交补丁等待批准")
        run_state = run_state.transition("AWAITING_APPROVAL")
        run_state = run_state.transition("FINISH")

        state_dict = run_state.to_dict()

        self.assertEqual("FINISH", run_state.stage)
        self.assertEqual(
            ("INIT", "PLAN", "EXECUTE", "AWAITING_APPROVAL", "FINISH"),
            run_state.stage_history,
        )
        self.assertEqual(
            ["INIT", "PLAN", "EXECUTE", "AWAITING_APPROVAL", "FINISH"],
            state_dict["stage_history"],
        )
        json.dumps(state_dict)

    def test_awaiting_approval_only_allows_finish(self) -> None:
        run_state = RunState().transition("PLAN").with_plan(_valid_plan("edit"))
        run_state = run_state.transition("EXECUTE").transition("AWAITING_APPROVAL")

        for next_stage in ["VERIFY", "REPAIR", "EXECUTE"]:
            with self.subTest(next_stage=next_stage):
                with self.assertRaisesRegex(
                    RunStateError,
                    f"AWAITING_APPROVAL -> {next_stage}",
                ):
                    run_state.transition(next_stage)

    def test_repair_attempt_greater_than_one_rejects(self) -> None:
        failed_result = VerificationResult(command_name="unit", passed=False, exit_code=1)
        run_state = RunState().transition("PLAN").with_plan(_valid_plan("edit"))
        run_state = run_state.transition("EXECUTE").transition("VERIFY")
        run_state = run_state.with_tests_result(failed_result)
        run_state = run_state.transition("REPAIR", reason="tests_failed")
        run_state = run_state.transition("EXECUTE").transition("VERIFY")
        run_state = run_state.with_tests_result(failed_result)

        with self.assertRaisesRegex(RunStateError, "最多允许一次"):
            run_state.transition("REPAIR", reason="tests_failed")

    def test_repair_for_non_tests_failed_rejects(self) -> None:
        run_state = RunState().transition("PLAN").with_plan(_valid_plan("edit"))
        run_state = run_state.transition("EXECUTE").transition("VERIFY")
        run_state = run_state.with_tests_result(
            VerificationResult(command_name="unit", passed=False, exit_code=1)
        )

        with self.assertRaisesRegex(RunStateError, "tests_failed"):
            run_state.transition("REPAIR", reason="lint_failed")


if __name__ == "__main__":
    unittest.main()
