# planner.py - strict JSON task plan generator

from __future__ import annotations

import json

from collections.abc import Sequence
from typing import Protocol

from schemas import (
    LEGAL_RISK_LEVELS,
    LEGAL_TASK_TYPES,
    MustContainRule,
    PlanStep,
    SchemaValidationError,
    TaskPlan,
    VerificationSpec,
)


class LlmClientProtocol(Protocol):
    """Structural protocol: an object with chat(messages) -> str."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class PlannerError(ValueError):
    """Task plan generation or validation failed."""


def build_planner_prompt(
    user_task: str,
    inspect_repo_result: dict[str, object],
    max_steps: int,
) -> str:
    """Build prompt asking the model for strict TaskPlan JSON."""

    return (
        f"User task: {user_task}\n\n"
        "Project overview (inspect_repo result):\n"
        f"{json.dumps(inspect_repo_result, ensure_ascii=False, indent=2)}\n\n"
        "Generate a strict TaskPlan JSON object based on the task and project overview.\n\n"
        "Requirements:\n"
        "1. Output ONLY a single JSON object.\n"
        "2. Do NOT wrap in Markdown code fences.\n"
        "3. Do NOT include any explanatory text before or after.\n"
        "4. The JSON object MUST have these fields:\n"
        f"   - \"task_type\": one of {sorted(LEGAL_TASK_TYPES)}\n"
        f"   - \"risk_level\": one of {sorted(LEGAL_RISK_LEVELS)}\n"
        f"   - \"max_steps\": positive integer <= {max_steps}\n"
        "   - \"requires_patch\": true/false\n"
        "   - \"requires_tests\": true/false\n"
        "   - \"expected_changed_files\": string list (\"requires_patch\"=true requires non-empty)\n"
        "   - \"steps\": list of objects with \"id\" (unique, non-empty), \"title\", optional \"description\"\n"
        "   - \"verification\": list of objects, each may have \"must_contain\"\n"
        "     must_contain entries: {\"path\": \"...\", \"strings\": [\"...\"]}\n\n"
        "5. \"edit\" or \"refactor\" task types MUST include verification.\n"
        "6. Number of steps must not exceed max_steps.\n"
        "7. Each step id must be unique and non-empty.\n"
        "8. Output ONLY the JSON object, no other text."
    )


def _validate_llm_output(raw_output: str) -> dict[str, object]:
    """Parse and validate that model output is a strict JSON object."""

    stripped_output = raw_output.strip()

    if not stripped_output:
        raise PlannerError("model output is empty")

    # Reject markdown code fences
    if stripped_output.startswith("```"):
        raise PlannerError("model output contains markdown code fence, rejected")

    try:
        parsed_output = json.loads(stripped_output)
    except json.JSONDecodeError as error:
        raise PlannerError(f"model output is not valid JSON: {error}") from error

    if not isinstance(parsed_output, dict):
        raise PlannerError("model output must be a JSON object, not array or scalar")

    return parsed_output


def _parse_plan_steps(raw_steps: object) -> tuple[PlanStep, ...]:
    """Parse steps from a raw JSON list."""
    if not isinstance(raw_steps, list):
        raise PlannerError("steps must be a list")
    if not raw_steps:
        raise PlannerError("steps must not be empty")

    parsed_steps: list[PlanStep] = []
    for step_index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise PlannerError(f"steps[{step_index}] must be a dict")

        step_id = raw_step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise PlannerError(f"steps[{step_index}].id is required and must be a non-empty string")

        step_title = raw_step.get("title")
        if not isinstance(step_title, str) or not step_title.strip():
            raise PlannerError(f"steps[{step_index}].title is required and must be a non-empty string")

        step_description = raw_step.get("description", "")
        if not isinstance(step_description, str):
            raise PlannerError(f"steps[{step_index}].description must be a string")

        parsed_steps.append(
            PlanStep(id=step_id.strip(), title=step_title.strip(), description=step_description)
        )

    return tuple(parsed_steps)


def _parse_must_contain_rules(raw_rules: object) -> tuple[MustContainRule, ...]:
    """Parse must_contain rules from a raw JSON list."""
    if isinstance(raw_rules, dict):
        raw_rules = [raw_rules]
    if not isinstance(raw_rules, list):
        raise PlannerError("must_contain must be a list")

    parsed_rules: list[MustContainRule] = []
    for rule_index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise PlannerError(f"must_contain[{rule_index}] must be a dict")

        rule_path = raw_rule.get("path")
        if not isinstance(rule_path, str) or not rule_path.strip():
            raise PlannerError(f"must_contain[{rule_index}].path is required and must be a non-empty string")

        raw_strings = raw_rule.get("strings")
        if not isinstance(raw_strings, list) or not raw_strings:
            raise PlannerError(f"must_contain[{rule_index}].strings must be a non-empty list of strings")

        parsed_strings: list[str] = []
        for string_item in raw_strings:
            if not isinstance(string_item, str) or not string_item.strip():
                raise PlannerError(
                    f"must_contain[{rule_index}].strings each item must be a non-empty string"
                )
            parsed_strings.append(string_item)

        parsed_rules.append(
            MustContainRule(path=rule_path.strip(), strings=tuple(parsed_strings))
        )

    return tuple(parsed_rules)


def _parse_verification_specs(raw_verification: object) -> tuple[VerificationSpec, ...]:
    """Parse verification specs from a raw JSON list."""
    if not isinstance(raw_verification, list):
        raise PlannerError("verification must be a list")

    parsed_specs: list[VerificationSpec] = []
    for spec_index, raw_spec in enumerate(raw_verification, start=1):
        if not isinstance(raw_spec, dict):
            raise PlannerError(f"verification[{spec_index}] must be a dict")

        if "must_contain" in raw_spec:
            must_contain_rules = _parse_must_contain_rules(raw_spec["must_contain"])
        else:
            must_contain_rules = ()

        parsed_specs.append(VerificationSpec(must_contain=must_contain_rules))

    return tuple(parsed_specs)


def _parse_expected_changed_files(raw_value: object) -> tuple[str, ...]:
    """Parse expected_changed_files from a raw JSON value."""
    if raw_value is None:
        return ()

    if not isinstance(raw_value, list):
        raise PlannerError("expected_changed_files must be a list of strings")

    parsed_files: list[str] = []
    for file_index, raw_file in enumerate(raw_value, start=1):
        if not isinstance(raw_file, str) or not raw_file.strip():
            raise PlannerError(
                f"expected_changed_files[{file_index}] must be a non-empty string"
            )
        parsed_files.append(raw_file.strip().replace(chr(92), chr(47)))

    return tuple(parsed_files)


def _build_task_plan_from_dict(parsed: dict[str, object], max_steps: int) -> TaskPlan:
    """Convert a parsed dict into a validated TaskPlan dataclass."""

    # --- task_type ---
    raw_task_type = parsed.get("task_type")
    if not isinstance(raw_task_type, str) or not raw_task_type.strip():
        raise PlannerError("task_type is required and must be a non-empty string")
    task_type = raw_task_type.strip()
    if task_type not in LEGAL_TASK_TYPES:
        raise PlannerError(f"task_type must be one of {sorted(LEGAL_TASK_TYPES)}")

    # --- risk_level ---
    raw_risk_level = parsed.get("risk_level")
    if not isinstance(raw_risk_level, str) or not raw_risk_level.strip():
        raise PlannerError("risk_level is required and must be a non-empty string")
    risk_level = raw_risk_level.strip()
    if risk_level not in LEGAL_RISK_LEVELS:
        raise PlannerError(f"risk_level must be one of {sorted(LEGAL_RISK_LEVELS)}")

    # --- steps ---
    steps = _parse_plan_steps(parsed.get("steps"))

    # --- requires_patch ---
    raw_requires_patch = parsed.get("requires_patch")
    if not isinstance(raw_requires_patch, bool):
        raise PlannerError("requires_patch must be a boolean")
    requires_patch = raw_requires_patch

    # --- requires_tests ---
    raw_requires_tests = parsed.get("requires_tests")
    if not isinstance(raw_requires_tests, bool):
        raise PlannerError("requires_tests must be a boolean")
    requires_tests = raw_requires_tests

    # --- expected_changed_files ---
    expected_changed_files = _parse_expected_changed_files(
        parsed.get("expected_changed_files")
    )

    # --- verification ---
    raw_verification = parsed.get("verification")
    if raw_verification is not None:
        verification = _parse_verification_specs(raw_verification)
    else:
        verification = ()

    # Use TaskPlan.__post_init__ for cross-field validation
    try:
        return TaskPlan(
            task_type=task_type,
            risk_level=risk_level,
            max_steps=max_steps,
            steps=steps,
            requires_patch=requires_patch,
            requires_tests=requires_tests,
            expected_changed_files=expected_changed_files,
            verification=verification,
        )
    except SchemaValidationError as error:
        raise PlannerError(str(error)) from error


def create_plan(
    user_task: str,
    inspect_repo_result: dict[str, object],
    max_steps: int,
    llm_client: LlmClientProtocol,
) -> TaskPlan:
    """Use an LLM to generate and validate a task plan."""

    prompt = build_planner_prompt(
        user_task=user_task,
        inspect_repo_result=inspect_repo_result,
        max_steps=max_steps,
    )

    messages = [
        {"role": "system", "content": "You are a code analysis plan generator. Output strict JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw_output = llm_client.chat(messages)
    except Exception as error:
        raise PlannerError(f"LLM call failed: {error}") from error

    parsed = _validate_llm_output(raw_output)
    return _build_task_plan_from_dict(parsed, max_steps)
