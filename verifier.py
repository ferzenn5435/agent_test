"""Deterministic run-state verifier."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eval_safety import normalize_relative_path
from run_state import RunState
from schemas import MustContainRule, VerificationResult


VERIFIER_COMMAND_NAME = "deterministic_verifier"
REASON_NOT_FINISHED = "not_finished"
REASON_MAX_STEPS_EXCEEDED = "max_steps_exceeded"
REASON_FORBIDDEN_CHANGED_FILE = "forbidden_changed_file"
REASON_MUST_CONTAIN_MISSING = "must_contain_missing"
REASON_TESTS_FAILED = "tests_failed"
REASON_PATCH_NOT_CONFIRMED = "patch_not_confirmed"
REASON_CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"


def verify_run_state(
    run_state: RunState,
    repo_path: str | Path,
    allowed_changed_files: Iterable[str] | None = None,
    must_contain: Iterable[MustContainRule | Mapping[str, object]] | None = None,
    must_not_read_full_files: Iterable[str] | None = None,
    max_total_tool_output_chars: int | None = None,
    patch_confirmed: bool | None = None,
) -> VerificationResult:
    """Verify a RunState against deterministic repo constraints."""

    if not isinstance(run_state, RunState):
        raise TypeError("run_state must be RunState")

    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise ValueError(f"repo_path is not a directory: {repo_path}")

    reasons: list[str] = []
    details: list[str] = []

    plan = run_state.plan
    if run_state.stage != "FINISH":
        reasons.append(REASON_NOT_FINISHED)
        details.append(f"stage={run_state.stage}")

    if plan is not None and len(run_state.execution_steps) > plan.max_steps:
        reasons.append(REASON_MAX_STEPS_EXCEEDED)
        details.append(f"execution_steps={len(run_state.execution_steps)} limit={plan.max_steps}")

    effective_allowed_changed_files = (
        tuple(allowed_changed_files)
        if allowed_changed_files is not None
        else _plan_expected_changed_files(plan)
    )
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
    if effective_must_contain:
        must_contain_issues = _check_must_contain(repo_root, effective_must_contain)
        if must_contain_issues:
            reasons.append(REASON_MUST_CONTAIN_MISSING)
            details.extend(must_contain_issues)

    tests_result = run_state.tests_result
    if tests_result is None or not tests_result.passed:
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
    if patch_confirmed is False:
        reasons.append(REASON_PATCH_NOT_CONFIRMED)
        details.append("patch_confirmed=False")
    elif patch_confirmed is None and requires_patch_confirmation:
        reasons.append(REASON_PATCH_NOT_CONFIRMED)
        details.append("patch_confirmed missing for patch-bearing run")

    context_issues = _check_context_constraints(
        context_stats=run_state.context_stats,
        must_not_read_full_files=must_not_read_full_files,
        max_total_tool_output_chars=max_total_tool_output_chars,
    )
    if context_issues:
        reasons.append(REASON_CONTEXT_BUDGET_EXCEEDED)
        details.extend(context_issues)

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


def _check_must_contain(
    repo_root: Path,
    rules: Iterable[MustContainRule | Mapping[str, object]],
) -> list[str]:
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
    if plan is None:
        return ()
    expected_changed_files = getattr(plan, "expected_changed_files", ())
    if not isinstance(expected_changed_files, tuple):
        return tuple(str(item) for item in expected_changed_files if isinstance(item, str))
    return expected_changed_files


def _plan_must_contain_rules(plan: Any | None) -> tuple[MustContainRule, ...]:
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
