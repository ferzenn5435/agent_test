"""v0.6 运行状态机。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from schemas import TaskPlan, VerificationResult


LEGAL_STAGES = {"INIT", "PLAN", "EXECUTE", "VERIFY", "REPAIR", "FINISH"}
LEGAL_TRANSITIONS = {
    "INIT": {"PLAN"},
    "PLAN": {"EXECUTE"},
    "EXECUTE": {"VERIFY"},
    "VERIFY": {"REPAIR", "FINISH"},
    "REPAIR": {"EXECUTE"},
    "FINISH": set(),
}


class RunStateError(ValueError):
    """运行状态迁移非法。"""


@dataclass(frozen=True)
class RunState:
    """记录 v0.6 plan-execute-verify 运行状态。"""

    stage: str = "INIT"
    stage_history: tuple[str, ...] = ("INIT",)
    plan: TaskPlan | None = None
    execution_steps: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    tests_result: VerificationResult | None = None
    repair_attempts: int = 0
    context_stats: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.stage not in LEGAL_STAGES:
            raise RunStateError(f"非法 stage: {self.stage}")
        if not self.stage_history:
            raise RunStateError("stage_history 不能为空")
        if self.stage_history[-1] != self.stage:
            raise RunStateError("stage_history 最后一项必须等于当前 stage")
        if self.repair_attempts < 0:
            raise RunStateError("repair_attempts 不能小于 0")
        if self.repair_attempts > 1:
            raise RunStateError("repair_attempts 最多只能为 1")

    def with_plan(self, plan: TaskPlan) -> RunState:
        """记录已生成的计划。"""

        if self.stage != "PLAN":
            raise RunStateError("只能在 PLAN stage 记录 plan")
        return replace(self, plan=plan)

    def with_execution_step(self, execution_step: str) -> RunState:
        """追加执行步骤记录。"""

        if self.stage not in {"EXECUTE", "REPAIR"}:
            raise RunStateError("只能在 EXECUTE/REPAIR stage 记录 execution_steps")
        if not execution_step.strip():
            raise RunStateError("execution_step 必须是非空字符串")
        return replace(
            self,
            execution_steps=(*self.execution_steps, execution_step),
        )

    def with_changed_file(self, changed_file: str) -> RunState:
        """追加变更文件记录。"""

        if self.stage not in {"EXECUTE", "REPAIR"}:
            raise RunStateError("只能在 EXECUTE/REPAIR stage 记录 changed_files")
        if not changed_file.strip():
            raise RunStateError("changed_file 必须是非空字符串")
        return replace(
            self,
            changed_files=(*self.changed_files, changed_file.replace("\\", "/")),
        )

    def with_tests_result(self, tests_result: VerificationResult) -> RunState:
        """记录验证结果。"""

        if self.stage != "VERIFY":
            raise RunStateError("只能在 VERIFY stage 记录 tests_result")
        return replace(self, tests_result=tests_result)

    def with_context_stats(self, context_stats: dict[str, object]) -> RunState:
        """记录上下文统计。"""

        return replace(self, context_stats=dict(context_stats))

    def transition(self, next_stage: str, reason: str | None = None) -> RunState:
        """执行硬约束状态迁移，非法迁移直接抛错。"""

        if next_stage not in LEGAL_STAGES:
            raise RunStateError(f"非法 next_stage: {next_stage}")
        if next_stage not in LEGAL_TRANSITIONS[self.stage]:
            raise RunStateError(f"非法状态迁移: {self.stage} -> {next_stage}")
        if next_stage == "REPAIR":
            self._validate_repair_transition(reason)
            repair_attempts = self.repair_attempts + 1
        else:
            repair_attempts = self.repair_attempts

        return replace(
            self,
            stage=next_stage,
            stage_history=(*self.stage_history, next_stage),
            repair_attempts=repair_attempts,
        )

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "stage": self.stage,
            "stage_history": list(self.stage_history),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "execution_steps": list(self.execution_steps),
            "changed_files": list(self.changed_files),
            "tests_result": self.tests_result.to_dict() if self.tests_result is not None else None,
            "repair_attempts": self.repair_attempts,
            "context_stats": dict(self.context_stats) if self.context_stats is not None else None,
        }

    def _validate_repair_transition(self, reason: str | None) -> None:
        if self.repair_attempts >= 1:
            raise RunStateError("repair transition 最多允许一次")
        if reason != "tests_failed":
            raise RunStateError("repair transition 只允许 reason=tests_failed")
        if self.tests_result is None:
            raise RunStateError("repair transition 需要先记录 tests_result")
        if self.tests_result.passed:
            raise RunStateError("tests_result 通过时不能进入 REPAIR")
