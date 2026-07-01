"""确定性 run-state verifier。

负责 `VERIFY` 阶段的结果判定：不相信模型自述，不以说明文本代替测试/约束
校验。只有当实际执行状态、测试、上下文、补丁审批、变更边界全部通过时，
`verify_run_state` 才返回通过。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eval_safety import normalize_relative_path
from run_state import RunState
from schemas import MustContainRule, VerificationResult


VERIFIER_COMMAND_NAME = "deterministic_verifier"
# 失败原因常量，供 `VerificationResult.reasons` 与审计日志复用。
REASON_NOT_FINISHED = "not_finished"
# 步数超限：执行 step 数量超过 `TaskPlan.max_steps`。
REASON_MAX_STEPS_EXCEEDED = "max_steps_exceeded"
# 非白名单文件被实际变更。
REASON_FORBIDDEN_CHANGED_FILE = "forbidden_changed_file"
# must_contain 约束不满足。
REASON_MUST_CONTAIN_MISSING = "must_contain_missing"
# 测试缺失或失败。
REASON_TESTS_FAILED = "tests_failed"
# 需要 patch 确认却未给出确认状态。
REASON_PATCH_NOT_CONFIRMED = "patch_not_confirmed"
# 上下文统计超限或上下文统计缺失。
REASON_CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"


def verify_run_state(
    run_state: RunState,
    repo_path: str | Path,
    allowed_changed_files: Iterable[str] | None = None,
    must_contain: Iterable[MustContainRule | Mapping[str, object]] | None = None,
    must_not_read_full_files: Iterable[str] | None = None,
    max_total_tool_output_chars: int | None = None,
    patch_confirmed: bool | None = None,
    pending_patch: Mapping[str, object] | None = None,
) -> VerificationResult:
    """按固定规则验证 `RunState`。

职责是聚合多个独立判定：阶段、步数、变更文件、文本证据、测试结果、补丁确
认、上下文预算。只要任一项失败即记录原因并返回失败结果；不因文本表述而
例外。
"""

    if not isinstance(run_state, RunState):
        raise TypeError("run_state must be RunState")

    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise ValueError(f"repo_path is not a directory: {repo_path}")

    reasons: list[str] = []
    details: list[str] = []
    pending_approval_terminal = _is_pending_approval_terminal(run_state, pending_patch)

    plan = run_state.plan
    # 非 pending approval 场景要求执行到 FINISH，保证 PLAN/EXECUTE 已收口。
    if run_state.stage != "FINISH" and not pending_approval_terminal:
        reasons.append(REASON_NOT_FINISHED)
        details.append(f"stage={run_state.stage}")

    # 执行步数为协议硬约束，超过上限直接阻断，防止超范围循环。
    if plan is not None and len(run_state.execution_steps) > plan.max_steps:
        reasons.append(REASON_MAX_STEPS_EXCEEDED)
        details.append(f"execution_steps={len(run_state.execution_steps)} limit={plan.max_steps}")

    effective_allowed_changed_files = (
        tuple(allowed_changed_files)
        if allowed_changed_files is not None
        else _plan_expected_changed_files(plan)
    )
    # changed_files 白名单：优先显式入参，否则回退 plan 中 expected_changed_files。
    if effective_allowed_changed_files:
        changed_file_issues = _check_allowed_changed_files(
            changed_files=run_state.changed_files,
            allowed_changed_files=effective_allowed_changed_files,
        )
        if changed_file_issues:
            reasons.append(REASON_FORBIDDEN_CHANGED_FILE)
            details.extend(changed_file_issues)

    effective_must_contain = (
        tuple(must_contain)
        if must_contain is not None
        else _plan_must_contain_rules(plan)
    )
    # must_contain 约束也可被外部调用覆盖；pending approval 终态不执行该检查。
    if effective_must_contain and not pending_approval_terminal:
        must_contain_issues = _check_must_contain(repo_root, effective_must_contain)
        if must_contain_issues:
            reasons.append(REASON_MUST_CONTAIN_MISSING)
            details.extend(must_contain_issues)

    # 测试结果缺失/未通过都记为失败，禁止模型凭借文本声明通过。
    tests_result = run_state.tests_result
    if not pending_approval_terminal and (tests_result is None or not tests_result.passed):
        reasons.append(REASON_TESTS_FAILED)
        if tests_result is None:
            details.append("tests_result is missing")
        else:
            details.append(
                f"tests_result command_name={tests_result.command_name!r} "
                f"exit_code={tests_result.exit_code!r}"
            )

    requires_patch_confirmation = bool(run_state.changed_files) or bool(
        plan is not None and plan.requires_patch
    )
    # patch 修改与 patch 预期修改同样受控：未确认则不能过 VERIFY。
    if pending_approval_terminal:
        pass
    elif patch_confirmed is False:
        reasons.append(REASON_PATCH_NOT_CONFIRMED)
        details.append("patch_confirmed=False")
    elif patch_confirmed is None and requires_patch_confirmation:
        reasons.append(REASON_PATCH_NOT_CONFIRMED)
        details.append("patch_confirmed missing for patch-bearing run")

    # 统一检查上下文预算：禁止过度读取大文件、过量输出导致的不可复现性风险。
    context_issues = _check_context_constraints(
        context_stats=run_state.context_stats,
        must_not_read_full_files=must_not_read_full_files,
        max_total_tool_output_chars=max_total_tool_output_chars,
    )
    if context_issues:
        reasons.append(REASON_CONTEXT_BUDGET_EXCEEDED)
        details.extend(context_issues)

    # repair 策略：只有“仅测试失败”时可重试修复，其他失败原因不能自动修复。
    repairable = reasons == [REASON_TESTS_FAILED]
    passed = not reasons
    output = "verification passed" if passed else "\n".join(details)
    return VerificationResult(
        command_name=VERIFIER_COMMAND_NAME,
        passed=passed,
        exit_code=0 if passed else 1,
        output=output,
        reasons=tuple(reasons),
        repairable=repairable,
    )


verify = verify_run_state


def _check_allowed_changed_files(
    changed_files: Iterable[str],
    allowed_changed_files: Iterable[str],
) -> list[str]:
    """检查 changed_files 与允许列表的一致性。

返回逐条问题，不直接抛异常，便于 `verify_run_state` 以完整问题列表返
回，形成可读的失败聚合输出。
"""
    issues: list[str] = []
    allowed_paths: set[str] = set()
    for allowed_file in allowed_changed_files:
        try:
            allowed_paths.add(normalize_relative_path(allowed_file))
        except ValueError as error:
            issues.append(f"allowed_changed_files invalid {allowed_file!r}: {error}")

    for changed_file in changed_files:
        try:
            normalized_changed_file = normalize_relative_path(changed_file)
        except ValueError as error:
            issues.append(f"changed_files invalid {changed_file!r}: {error}")
            continue

        if normalized_changed_file not in allowed_paths:
            issues.append(f"changed_files unauthorized: {normalized_changed_file}")

    return issues


def _is_pending_approval_terminal(
    run_state: RunState,
    pending_patch: Mapping[str, object] | None,
) -> bool:
    """判断是否处于 pending approval 的终态。

该返回值会放宽 `not_finished`/`must_contain`/`tests` 的严格阻断，用于 CLI 的
人工审批流程。
"""
    if run_state.stage not in {"AWAITING_APPROVAL", "FINISH"}:
        return False
    if "AWAITING_APPROVAL" not in run_state.stage_history:
        return False
    if pending_patch is None:
        return False
    if pending_patch.get("status") != "pending_approval":
        return False
    patch_id = pending_patch.get("patch_id")
    return isinstance(patch_id, str) and bool(patch_id.strip())


def _check_must_contain(
    repo_root: Path,
    rules: Iterable[MustContainRule | Mapping[str, object]],
) -> list[str]:
    """按路径和关键字符串检查文件内容是否存在。

该函数执行文本存在性断言，不修改任何状态，是 VERIFY 阶段的证据链检查。
"""
    issues: list[str] = []
    resolved_repo_root = repo_root.resolve()

    for rule in rules:
        raw_path, required_strings = _extract_must_contain_rule(rule)
        try:
            normalized_path = normalize_relative_path(raw_path)
        except ValueError as error:
            issues.append(f"must_contain invalid path {raw_path!r}: {error}")
            continue

        target_file = (resolved_repo_root / normalized_path).resolve()
        try:
            target_file.relative_to(resolved_repo_root)
        except ValueError:
            issues.append(f"must_contain file outside repo: {normalized_path}")
            continue

        if not target_file.is_file():
            issues.append(f"must_contain file missing: {normalized_path}")
            continue

        try:
            file_text = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"must_contain file is not valid UTF-8: {normalized_path}")
            continue
        except OSError as error:
            issues.append(f"must_contain file read failed: {normalized_path}: {error}")
            continue

        for required_string in required_strings:
            if required_string not in file_text:
                issues.append(
                    f"must_contain missing text: {normalized_path} -> {required_string}"
                )

    return issues


def _check_context_constraints(
    context_stats: dict[str, object] | None,
    must_not_read_full_files: Iterable[str] | None,
    max_total_tool_output_chars: int | None,
) -> list[str]:
    """检查上下文使用约束。

缺失 `context_stats` 被当作失败原因之一，避免 verifier 在无上下文统计时默
认为满足预算。
"""
    if must_not_read_full_files is None and max_total_tool_output_chars is None:
        return []

    issues: list[str] = []
    forbidden_paths = tuple(_normalize_path_list(must_not_read_full_files or ()))
    if context_stats is None:
        issues.append("context_stats missing")
        return issues

    full_file_reads = _normalize_path_list(context_stats.get("full_file_reads"))
    for forbidden_path in forbidden_paths:
        if forbidden_path in full_file_reads:
            issues.append(
                "must_not_read_full_files violated: "
                f"forbidden_path={forbidden_path}, actual_full_file_reads={full_file_reads}"
            )

    if max_total_tool_output_chars is not None:
        total_tool_output_chars = context_stats.get("total_tool_output_chars")
        if isinstance(total_tool_output_chars, int) and not isinstance(total_tool_output_chars, bool):
            if total_tool_output_chars > max_total_tool_output_chars:
                issues.append(
                    "max_total_tool_output_chars violated: "
                    f"limit={max_total_tool_output_chars}, actual={total_tool_output_chars}"
                )
        else:
            issues.append("context_stats.total_tool_output_chars is missing or invalid")

    return issues


def _normalize_path_list(value: Iterable[str] | object) -> tuple[str, ...]:
    """将路径集合标准化为仓库内相对路径。

对于非法条目不抛错，返回 `<invalid:...>` 作为失败信息的一部分，保证聚合输
出完整。
"""
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return ()

    normalized_paths: list[str] = []
    for item in value:
        if isinstance(item, str):
            try:
                normalized_paths.append(normalize_relative_path(item))
            except ValueError as error:
                normalized_paths.append(f"<invalid:{error}>")
    return tuple(normalized_paths)


def _plan_expected_changed_files(plan: Any | None) -> tuple[str, ...]:
    """从 plan 提取 `expected_changed_files`。

该字段用于 verifier 的默认白名单；plan 缺失时返回空元组，表示不加白名单
约束。
"""
    if plan is None:
        return ()
    expected_changed_files = getattr(plan, "expected_changed_files", ())
    if not isinstance(expected_changed_files, tuple):
        return tuple(str(item) for item in expected_changed_files if isinstance(item, str))
    return expected_changed_files


def _plan_must_contain_rules(plan: Any | None) -> tuple[MustContainRule, ...]:
    """聚合 plan 内 `verification` 的 must_contain 规则。

只收集 `MustContainRule` 类型实例，忽略非预期结构以保持协议稳健。
"""
    if plan is None:
        return ()
    verification = getattr(plan, "verification", ())
    collected_rules: list[MustContainRule] = []
    if not isinstance(verification, Sequence):
        return ()
    for spec in verification:
        must_contain_rules = getattr(spec, "must_contain", ())
        if isinstance(must_contain_rules, Sequence):
            collected_rules.extend(
                rule for rule in must_contain_rules if isinstance(rule, MustContainRule)
            )
    return tuple(collected_rules)


def _extract_must_contain_rule(
    rule: MustContainRule | Mapping[str, object],
) -> tuple[str, Sequence[str]]:
    """将 `MustContainRule` 与 dict 规则归一化为 `(path, strings)`。

解析容错返回 `("", ())`，上游再统一归类为缺失或无效问题。
"""
    if isinstance(rule, MustContainRule):
        return rule.path, rule.strings

    raw_path = rule.get("path")
    raw_strings = rule.get("strings")
    if not isinstance(raw_path, str):
        return "", ()
    if not isinstance(raw_strings, Sequence) or isinstance(raw_strings, (str, bytes)):
        return raw_path, ()
    strings = tuple(string for string in raw_strings if isinstance(string, str))
    return raw_path, strings
