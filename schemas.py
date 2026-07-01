"""v0.6 运行协议与验证 schema。

定义任务计划、执行步骤、验证结果与最终运行结果的约束。
字段校验失败将抛出 SchemaValidationError，阻止非法计划入场。
"""

from __future__ import annotations

from dataclasses import dataclass


LEGAL_TASK_TYPES = {"analysis", "edit", "refactor", "test"}
LEGAL_RISK_LEVELS = {"low", "medium", "high"}


class SchemaValidationError(ValueError):
    """v0.6 schema 校验失败。"""


@dataclass(frozen=True)
class MustContainRule:
    """单条文件内容验证规则。

用于验证器中检查输出文件是否包含指定字符串集合。
"""

    path: str
    strings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise SchemaValidationError("must_contain path 必须是非空字符串")
        if not self.strings:
            raise SchemaValidationError("must_contain strings 必须是非空字符串数组")
        for required_string in self.strings:
            if not required_string.strip():
                raise SchemaValidationError("must_contain strings 必须是非空字符串数组")

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "path": self.path,
            "strings": list(self.strings),
        }


@dataclass(frozen=True)
class VerificationSpec:
    """计划要求执行的确定性验证规则集合。"""

    must_contain: tuple[MustContainRule, ...] = ()

    def __post_init__(self) -> None:
        for must_contain_rule in self.must_contain:
            if not isinstance(must_contain_rule, MustContainRule):
                raise SchemaValidationError("must_contain 必须由 MustContainRule 组成")

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "must_contain": [
                must_contain_rule.to_dict()
                for must_contain_rule in self.must_contain
            ],
        }


@dataclass(frozen=True)
class PlanStep:
    """计划中的单个执行步骤。

step id/title/description 将用于 planner 与 execute 提示词中的
计划绑定与执行定位。
"""

    id: str
    title: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise SchemaValidationError("step id 必须是非空字符串")
        if not self.title.strip():
            raise SchemaValidationError("step title 必须是非空字符串")

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
        }


@dataclass(frozen=True)
class TaskPlan:
    """v0.6 任务计划。

约束重点：
- task_type ∈ {analysis, edit, refactor, test}
- risk_level ∈ {low, medium, high}
- max_steps 为正整数，steps 数量不允许超过 max_steps
- edit/refactor 必须声明 verification
- requires_patch 时 expected_changed_files 非空
"""

    task_type: str
    risk_level: str
    max_steps: int
    steps: tuple[PlanStep, ...]
    requires_patch: bool = False
    requires_tests: bool = False
    expected_changed_files: tuple[str, ...] = ()
    verification: tuple[VerificationSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.task_type not in LEGAL_TASK_TYPES:
            raise SchemaValidationError(
                "task_type 必须是 analysis/edit/refactor/test 之一"
            )
        if self.risk_level not in LEGAL_RISK_LEVELS:
            raise SchemaValidationError("risk_level 必须是 low/medium/high 之一")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise SchemaValidationError("max_steps 必须是正整数")
        if self.max_steps <= 0:
            raise SchemaValidationError("max_steps 必须是正整数")
        if len(self.steps) > self.max_steps:
            raise SchemaValidationError("steps 数量不能超过 max_steps")

        seen_step_ids: set[str] = set()
        for step in self.steps:
            step_id = step.id.strip()
            if not step_id:
                raise SchemaValidationError("step id 必须是非空字符串")
            if step_id in seen_step_ids:
                raise SchemaValidationError(f"step id 重复: {step_id}")
            seen_step_ids.add(step_id)

        if self.task_type in {"edit", "refactor"} and not self.verification:
            raise SchemaValidationError("edit/refactor 计划必须包含 verification")
        if self.requires_patch and not self.expected_changed_files:
            raise SchemaValidationError(
                "requires_patch=True 时 expected_changed_files 不能为空"
            )
        for expected_changed_file in self.expected_changed_files:
            if not expected_changed_file.strip():
                raise SchemaValidationError("expected_changed_files 必须是非空字符串数组")

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "task_type": self.task_type,
            "risk_level": self.risk_level,
            "max_steps": self.max_steps,
            "requires_patch": self.requires_patch,
            "requires_tests": self.requires_tests,
            "expected_changed_files": [
                expected_changed_file.replace("\\", "/")
                for expected_changed_file in self.expected_changed_files
            ],
            "steps": [step.to_dict() for step in self.steps],
            "verification": [spec.to_dict() for spec in self.verification],
        }


@dataclass(frozen=True)
class VerificationResult:
    """单条验证命令结果。

包含命令名、是否通过、退出码、输出摘要、失败原因与可修复性标记。
"""

    command_name: str
    passed: bool
    exit_code: int | None = None
    output: str = ""
    reasons: tuple[str, ...] = ()
    repairable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.command_name, str) or not self.command_name.strip():
            raise SchemaValidationError("command_name 必须是非空字符串")
        for reason in self.reasons:
            if not isinstance(reason, str) or not reason.strip():
                raise SchemaValidationError("reasons 必须是非空字符串数组")
        if not isinstance(self.repairable, bool):
            raise SchemaValidationError("repairable 必须是 bool")

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "command_name": self.command_name,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "output": self.output,
            "reasons": list(self.reasons),
            "repairable": self.repairable,
        }


@dataclass(frozen=True)
class RunResult:
    """一次 v0.6 运行的最终结果。

success 控制总体状态，verification_results 记录阶段验证轨迹，error
在失败场景提供可读错误串。
"""

    success: bool
    final_answer: str | None = None
    verification_results: tuple[VerificationResult, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """返回稳定、可 JSON 序列化的 primitive schema。"""

        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "verification_results": [
                result.to_dict()
                for result in self.verification_results
            ],
            "error": self.error,
        }
