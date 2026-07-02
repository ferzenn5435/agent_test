"""Planner：PLAN 阶段严格生成与校验 `TaskPlan` JSON。

本文件承接执行协议中的 `PLAN` 阶段：只负责把 LLM 的自然语言输出
转成可验证的结构化计划，不在此阶段执行业务改动。计划字段与约束一旦
通过校验，执行器才能进入 `EXECUTE` 并使用这些 `plan_step_id` 驱动工具调用。
"""

from __future__ import annotations

import json

from collections.abc import Sequence
from typing import Protocol

from model_provider import LLMResponse
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
    """约定外部 LLM 客户端的最小接口。

    `Planner` 不依赖具体厂商实现，只要求 `chat_response(messages)` 返回
    结构化 LLMResponse；后续 content 是否是 JSON、是否满足字段约束，
    由本文件自身完成，不可把“模型能力”当作信任边界。
    """

    def chat_response(self, messages: list[dict[str, str]]) -> LLMResponse:
        ...


class PlannerError(ValueError):
    """Plan 构建/校验失败的专用错误。

该异常一旦抛出，PLAN 阶段直接失败，不能进入 `EXECUTE`。
"""


def build_planner_prompt(
    user_task: str,
    inspect_repo_result: dict[str, object],
    max_steps: int,
) -> str:
    """构建用于计划生成的系统提示。

返回字符串要求模型只输出严格的 `TaskPlan` JSON，原因如下：

- 统一 `TaskPlan` 结构让执行层能逐步关联 `plan_step_id` 与真实 `run_state`。
- 约束 `max_steps`、`requires_patch`、`requires_tests`、`verification`
  等字段，保证执行与验证条件在计划层就被约束。
- 禁止 markdown、额外说明，避免 LLM 在边界外输出导致解析失败。
"""

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
    """把模型输出从文本解析为 JSON 并执行第一道协议校验。

这里是 `PLAN` 阶段的关键入口：只接受非空、无 code fence 的文本，且必须可
解析为 JSON object（禁止数组/标量）。该步骤失败会直接拒绝执行，防止后续
工具循环在弱格式输入上继续运行。
"""

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
    """解析并校验计划步骤列表。

返回的 `PlanStep` 顺序即执行顺序基线。这里仅做字段形状校验：每个步骤
必须有非空 `id` 与 `title`，因为后续 `model` 在 `EXECUTE` 阶段必须回填同
名 `plan_step_id`；空、重复或缺失会让执行链路失去追踪锚点。
"""
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
    """解析 `verification` 中的 `must_contain` 约束。

`must_contain` 用于 VERIFY 期望的证据文本检查。这里要求 `path` 与 `strings`
结构完整，以便后续 verifier 可以对结果文件执行“内容存在性”判断。
"""
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
    """解析 `verification` 规范列表。

每个 spec 支持仅有的 `must_contain` 字段；解析时将其转为标准对象，后续
由 `verify_run_state` 按规范统一聚合并验证。
"""
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
    """解析计划中预期变更文件清单。

该字段作为 verifier 的白名单约束基础，任何未落在列表内的实际变更都将被
`VERIFY` 拒绝，避免工具链无界改动。
"""
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
    """将 dict 映射转换为 `TaskPlan`，并触发跨字段语义校验。

字段校验采用两层：

1. 本函数做结构性与范围性检查（枚举值、类型、数量上界）。
2. 最终构造 `TaskPlan` 时由 `__post_init__` 完成跨字段语义检查
   （如任务类型、PATCH/测试要求与步数关系）。

两层校验形成 “parse-time + schema-time” 防线，保证后续 `EXECUTE/VERIFY` 有
稳定前置条件。
"""

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
    """通过 LLM 产出并校验任务计划。

返回的 `TaskPlan` 一旦构造成功，等价于 PLAN 阶段通过：
- 执行器可以依据 `steps` 组织 `plan_step_id`。
- Verifier 可直接复用该计划中的 `expected_changed_files` 与 `verification`
  作为 `VERIFY` 约束。
- 若模型只自称完成计划、未满足 JSON 协议，本函数抛错，`execute` 流程不会开
  始，防止“自我声明”进入系统。
"""

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
        response = llm_client.chat_response(messages)
    except Exception as error:
        raise PlannerError(f"LLM call failed: {error}") from error

    parsed = _validate_llm_output(response.content)
    return _build_task_plan_from_dict(parsed, max_steps)
