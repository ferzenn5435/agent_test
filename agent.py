"""LLM + 工具调用核心循环（v0.6 PLAN→EXECUTE→VERIFY）。

流程包含三段：
- PLAN：基于仓库检查与问题生成受约束的 TaskPlan。
- EXECUTE：按 plan_step_id 执行工具调用并记录上下文。
- VERIFY：在 finish 后执行确定性校验并决定是否允许一次 repair。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from config import MAX_STEPS
from context_stats import ContextStats
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from model_provider import LLMResponse, TokenUsage
import planner
from prompts import build_system_prompt
from run_state import RunState
from schemas import TaskPlan, VerificationResult
from tools import RepositoryTools, ToolError
from usage_tracker import UsageCallRecord, summarize_usage_calls
import verifier


FULL_OBSERVATION_FEEDBACK_COUNT = 3
PROMPT_VERSION = "v0.6"


@dataclass(frozen=True)
class _LlmCallResult:
    """一次 LLM 调用的日志安全快照。

该结构仅承载可序列化字段，不包含原始请求体，便于统一记录。
"""

    content: str
    latency_ms: float | None = None
    usage: dict[str, object] | None = None
    provider: str | None = None
    model: str | None = None
    profile_name: str | None = None


class _UsageTrackingLlmProxy:
    """给 planner 使用的结构化 LLM 调用代理。

    Planner 只读取 LLMResponse.content。真实统计仍由 CodeAnalysisAgent
    统一在 _call_llm 中更新。
"""

    def __init__(self, agent: CodeAnalysisAgent) -> None:
        """注入主 Agent，作为 chat 返回值的统一来源。"""
        self.agent = agent

    def chat_response(self, messages: list[dict[str, str]]) -> LLMResponse:
        """调用主 agent 的 LLM 接口并返回结构化响应。"""
        call_result = self.agent._call_llm(messages)
        return LLMResponse(
            content=call_result.content,
            provider=call_result.provider or "unknown",
            model=call_result.model or "unknown",
            profile_name=call_result.profile_name or "unknown",
            latency_ms=call_result.latency_ms or 0.0,
            usage=TokenUsage(),
            raw={},
        )


class AgentStepError(ValueError):
    """模型单步输出不符合协议。"""


class CodeAnalysisAgent:
    """最小本地代码库分析 agent。

主要职责：
- 组织执行流程的阶段切换与日志更新。
- 校验模型输出协议、执行工具调用与上下文统计。
- 在 verify 阶段对结果进行最终判定，控制 pending approval 与 repair。
"""

    def __init__(
        self,
        llm_client: LlmClient,
        repository_tools: RepositoryTools,
        run_logger: RunLogger,
        max_steps: int = MAX_STEPS,
        tool_runner: Callable[[dict[str, object]], object] | None = None,
        pending_approval_mode: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.repository_tools = repository_tools
        self.run_logger = run_logger
        self.max_steps = max_steps
        self.tool_runner = tool_runner or repository_tools.run_tool
        self.pending_approval_mode = pending_approval_mode
        self.context_stats = ContextStats()
        self._usage_call_records: list[UsageCallRecord] = []
        self._llm_call_count = 0
        self._model_profile = self._initial_model_profile()
        self._provider: str | None = None
        self._model: str | None = None

    def answer(self, question: str) -> str:
        """运行一次全量执行链路并返回最终可读答案。

PLAN 阶段先生成计划，再按计划执行工具调用。执行阶段将所有工具输入/
输出与统计落日志，支持失败恢复与重试。finish 后进入 VERIFY 阶段，做
deterministic 校验与最终汇总。

边界：
- propose_patch 命中 pending approval 时立即返回待审视结果，不继续执行。
- verify 仅在条件满足时允许一次 REPAIR（tests_failed 且 repair_attempts=0）。
"""

        context_stats = ContextStats()
        self._reset_usage_tracking()
        run_state = RunState()
        latest_test_result: VerificationResult | None = None
        patch_confirmed = False
        propose_patch_succeeded = False
        pending_patch_to_return: dict[str, object] | None = None
        observation_summaries: list[dict[str, object]] = []
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(self.repository_tools.tool_descriptions()),
            },
            {
                "role": "user",
                "content": f"用户问题: {question}",
            },
        ]

        try:
            # PLAN：先做仓库检查，避免 planner 依赖缺失上下文。
            inspect_repo_result = self._inspect_repo_safely()
            run_state = run_state.transition("PLAN")
            self._update_logger_state_payload(run_state, verify_status=None)
            plan = planner.create_plan(
                question,
                inspect_repo_result,
                self.max_steps,
                _UsageTrackingLlmProxy(self),
            )
            run_state = run_state.with_plan(plan)
            self._update_logger_state_payload(run_state, verify_status=None)
            run_state = run_state.transition("EXECUTE")
            messages.append(
                {
                    "role": "user",
                    "content": self._format_execute_stage_instructions(plan),
                }
            )

            step_limit = min(self.max_steps, plan.max_steps)
            step_number = 1
            # EXECUTE：循环执行计划步骤，未 finish 前可持续交互。
            while step_number <= step_limit:
                messages_for_llm = self._messages_for_llm(messages, observation_summaries)
                context_stats = replace(
                    context_stats,
                    messages_total_chars=self._messages_total_chars(messages_for_llm),
                )
                self.context_stats = context_stats
                llm_call_result = self._call_llm(messages_for_llm)
                model_output = llm_call_result.content
                messages.append({"role": "assistant", "content": model_output})

                try:
                    tool_call = self._parse_model_output(model_output)
                except AgentStepError as error:
                    tool_result = {"ok": False, "error": str(error)}
                    serialized_result = self._serialize_tool_result(tool_result)
                    context_stats = self._update_context_stats(
                        context_stats=context_stats,
                        tool_call=None,
                        tool_result=tool_result,
                        serialized_result=serialized_result,
                    )
                    self.context_stats = context_stats
                    self.run_logger.record_step(
                        step_number=step_number,
                        model_output=model_output,
                        tool_call=None,
                        tool_result=tool_result,
                        latency_ms=llm_call_result.latency_ms,
                        usage=llm_call_result.usage,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._format_serialized_tool_feedback(serialized_result),
                        }
                    )
                    observation_summaries.append(
                        self._build_observation_summary(
                            tool_call=None,
                            tool_result=tool_result,
                            serialized_result=serialized_result,
                            message_index=len(messages) - 1,
                        )
                    )
                    step_number += 1
                    continue

                plan_step_error = self._validate_plan_step_id(tool_call, run_state)
                execution_gate_error = plan_step_error
                if execution_gate_error is None:
                    execution_gate_error = self._validate_patch_sequence(
                        tool_call=tool_call,
                        propose_patch_succeeded=propose_patch_succeeded,
                    )

                if execution_gate_error is None:
                    tool_result = self._run_tool(tool_call, run_state, question)
                    run_state = self._record_execution_state(
                        run_state=run_state,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )
                    test_result = self._verification_result_from_tool_call(tool_call, tool_result)
                    if test_result is not None:
                        latest_test_result = test_result
                    if self._is_successful_propose_patch(tool_call, tool_result):
                        propose_patch_succeeded = True
                        if self.pending_approval_mode:
                            pending_patch_to_return = self._pending_patch_from_tool_result(tool_result)
                    if self._is_confirmed_apply_patch(tool_call, tool_result):
                        patch_confirmed = True
                else:
                    tool_result = {"ok": False, "error": execution_gate_error}
                serialized_result = self._serialize_tool_result(tool_result)
                context_stats = self._update_context_stats(
                    context_stats=context_stats,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    serialized_result=serialized_result,
                )
                self.context_stats = context_stats
                self.run_logger.record_step(
                    step_number=step_number,
                    model_output=model_output,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    latency_ms=llm_call_result.latency_ms,
                    usage=llm_call_result.usage,
                )

                if pending_patch_to_return is not None:
                    # pending approval：补丁已生成但尚未落盘应用，返回人工审查路径。
                    awaiting_state = run_state.transition("AWAITING_APPROVAL")
                    finished_state = awaiting_state.transition("FINISH")
                    verified_state, verify_status = self._verify_finished_run(
                        run_state=awaiting_state,
                        context_stats=context_stats,
                        patch_confirmed=False,
                        pending_patch=pending_patch_to_return,
                    )
                    self._update_logger_state_payload(finished_state, verify_status)
                    final_answer = self._format_pending_approval_final_answer(
                        pending_patch=pending_patch_to_return,
                        run_state=finished_state,
                        verify_status=verify_status,
                    )
                    self.run_logger.set_context_stats(context_stats.to_dict())
                    self.run_logger.set_final_answer(final_answer)
                    return final_answer

                if tool_call["tool"] == "finish" and tool_result["ok"] is True:
                    final_answer = str(tool_result["output"])
                    # VERIFY：模型 finish 后，不再执行工具调用，转入验证与结果汇总。
                    run_state = run_state.transition("VERIFY")
                    tests_result = self._build_stage_tests_result(
                        latest_test_result=latest_test_result,
                        run_state=run_state,
                    )
                    run_state = run_state.with_tests_result(tests_result)
                    finished_state = run_state.transition("FINISH")
                    verified_state, verify_status = self._verify_finished_run(
                        run_state=finished_state,
                        context_stats=context_stats,
                        patch_confirmed=patch_confirmed,
                    )
                    if verify_status.passed:
                        final_answer = self._format_final_answer(
                            base_answer=final_answer,
                            run_state=verified_state,
                            tests_result=tests_result,
                            verify_status=verify_status,
                        )
                        self._update_logger_state_payload(verified_state, verify_status)
                        self.run_logger.set_context_stats(context_stats.to_dict())
                        self.run_logger.set_final_answer(final_answer)
                        return final_answer

                    self._update_logger_state_payload(verified_state, verify_status)
                    if self._should_repair(verify_status, run_state):
                        run_state = run_state.transition("REPAIR", reason="tests_failed")
                        run_state = run_state.transition("EXECUTE")
                        messages.append(
                            {
                                "role": "user",
                                "content": self._format_verify_feedback(verify_status),
                            }
                        )
                        step_number += 1
                        continue

                    self.run_logger.set_context_stats(context_stats.to_dict())
                    error_message = self._format_verify_error(verify_status)
                    final_answer = self._format_final_answer(
                        base_answer=final_answer,
                        run_state=verified_state,
                        tests_result=tests_result,
                        verify_status=verify_status,
                    )
                    self.run_logger.set_error(error_message)
                    self.run_logger.set_final_answer(final_answer)
                    return final_answer

                messages.append(
                    {
                        "role": "user",
                        "content": self._format_serialized_tool_feedback(serialized_result),
                    }
                )
                observation_summaries.append(
                    self._build_observation_summary(
                        tool_call=tool_call,
                        tool_result=tool_result,
                        serialized_result=serialized_result,
                        message_index=len(messages) - 1,
                    )
                )
                step_number += 1
        except planner.PlannerError as error:
            final_answer = f"PLAN 阶段失败: {error}"
            self.run_logger.set_context_stats(context_stats.to_dict())
            self.run_logger.set_error(final_answer)
            self.run_logger.set_final_answer(final_answer)
            return final_answer
        except Exception:
            self.run_logger.set_context_stats(context_stats.to_dict())
            raise

        final_answer = f"达到最大循环步数 {self.max_steps}，模型未调用 finish(answer) 完成任务。"
        self.run_logger.set_context_stats(context_stats.to_dict())
        self.run_logger.set_error(final_answer)
        self.run_logger.set_final_answer(final_answer)
        return final_answer

    def _reset_usage_tracking(self) -> None:
        self._usage_call_records = []
        self._llm_call_count = 0
        self._model_profile = self._initial_model_profile()
        self._provider = None
        self._model = None
        self._update_logger_usage_payload()

    def _initial_model_profile(self) -> str | None:
        """从日志 payload 或 llm_client 推断本次运行使用的 profile。"""
        payload = getattr(self.run_logger, "payload", None)
        if isinstance(payload, dict):
            payload_profile = payload.get("model_profile")
            if isinstance(payload_profile, str) and payload_profile.strip():
                return payload_profile

        client_profile = getattr(self.llm_client, "profile_name", None)
        if isinstance(client_profile, str) and client_profile.strip():
            return client_profile
        return None

    def _call_llm(self, messages: list[dict[str, str]]) -> _LlmCallResult:
        """执行一次 LLM 调用并生成统一的调用快照。

        当前 LLM API 返回 LLMResponse，调用方统一读取 content 并记录
        latency/usage/provider/model。
"""
        response = self.llm_client.chat_response(messages)
        content = response.content
        latency_ms = self._optional_float(response.latency_ms)
        usage_value = response.usage
        usage_dict = self._usage_to_dict(usage_value)
        provider = self._optional_text(response.provider)
        model = self._optional_text(response.model)
        profile_name = self._optional_text(response.profile_name)
        self._record_llm_call(
            latency_ms=latency_ms,
            usage_value=usage_value,
            provider=provider,
            model=model,
            profile_name=profile_name,
        )
        return _LlmCallResult(
            content=content,
            latency_ms=latency_ms,
            usage=usage_dict,
            provider=provider,
            model=model,
            profile_name=profile_name,
        )

    def _record_llm_call(
        self,
        *,
        latency_ms: float | None,
        usage_value: object,
        provider: str | None,
        model: str | None,
        profile_name: str | None,
    ) -> None:
        """记录 LLM 调用次数与 usage，用于后续 cost 与调用统计。"""
        self._llm_call_count += 1
        if profile_name is not None:
            self._model_profile = profile_name
        if provider is not None:
            self._provider = provider
        if model is not None:
            self._model = model
        if isinstance(usage_value, TokenUsage) and latency_ms is not None:
            self._usage_call_records.append(
                UsageCallRecord(latency_ms=latency_ms, usage=usage_value)
            )
        self._update_logger_usage_payload()

    def _update_logger_usage_payload(self) -> None:
        """将模型调用统计同步写入 logger payload。"""
        usage_summary = summarize_usage_calls(self._usage_call_records).to_dict()
        set_usage_summary = getattr(self.run_logger, "set_usage_summary", None)
        if callable(set_usage_summary):
            set_usage_summary(
                model_profile=self._model_profile,
                provider=self._provider,
                model=self._model,
                prompt_version=PROMPT_VERSION,
                usage_summary=usage_summary,
            )
            return

        payload = getattr(self.run_logger, "payload", None)
        if isinstance(payload, dict):
            payload["model_profile"] = self._model_profile
            payload["provider"] = self._provider
            payload["model"] = self._model
            payload["prompt_version"] = PROMPT_VERSION
            payload["usage_summary"] = usage_summary
            save = getattr(self.run_logger, "save", None)
            if callable(save):
                save()

    def _usage_to_dict(self, usage_value: object) -> dict[str, object] | None:
        to_dict = getattr(usage_value, "to_dict", None)
        if callable(to_dict):
            usage_dict = to_dict()
            if isinstance(usage_dict, dict):
                return dict(usage_dict)
        return None

    def _optional_float(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _optional_text(self, value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _inspect_repo_safely(self) -> dict[str, object]:
        """执行仓库检查，失败时返回结构化错误而非抛异常。"""
        try:
            return self.repository_tools.inspect_repo()
        except (OSError, ToolError) as error:
            return {"ok": False, "error": str(error)}

    def _parse_model_output(self, model_output: str) -> dict[str, object]:
        """解析并校验模型输出，要求思路 + 工具名 + args/plan_step_id 协议。"""
        try:
            parsed_output = json.loads(model_output.strip())
        except json.JSONDecodeError as error:
            raise AgentStepError("模型输出不是严格 JSON 对象") from error

        if not isinstance(parsed_output, dict):
            raise AgentStepError("模型输出必须是 JSON 对象")

        thought = parsed_output.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            raise AgentStepError("thought 必须是非空字符串")

        tool_name = parsed_output.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise AgentStepError("tool 必须是非空字符串")
        if tool_name not in self.repository_tools.tools:
            raise AgentStepError(f"未知工具: {tool_name}")

        tool_args = parsed_output.get("args")
        if not isinstance(tool_args, dict):
            raise AgentStepError("args 必须是对象")

        tool_call: dict[str, object] = {
            "thought": thought,
            "tool": tool_name,
            "args": tool_args,
        }
        plan_step_id = parsed_output.get("plan_step_id")
        if plan_step_id is not None:
            tool_call["plan_step_id"] = plan_step_id
        return tool_call

    def _validate_plan_step_id(
        self,
        tool_call: dict[str, object],
        run_state: RunState,
    ) -> str | None:
        """确保 plan_step_id 来源于当前 plan，避免跳步或空步骤执行。"""
        raw_plan_step_id = tool_call.get("plan_step_id")
        if not isinstance(raw_plan_step_id, str) or not raw_plan_step_id.strip():
            return "plan_step_id 必须是计划中存在的非空字符串"
        plan = run_state.plan
        if plan is None:
            return "执行前缺少 plan"
        legal_step_ids = {step.id for step in plan.steps}
        if raw_plan_step_id.strip() not in legal_step_ids:
            valid_ids = ", ".join(sorted(legal_step_ids))
            return f"未知 plan_step_id: {raw_plan_step_id}；有效 plan_step_id: {valid_ids}"
        tool_call["plan_step_id"] = raw_plan_step_id.strip()
        return None

    def _format_execute_stage_instructions(self, plan: TaskPlan) -> str:
        """把可执行计划序列化为执行提示，固定 plan_step_id 白名单。"""
        plan_dict = plan.to_dict()
        valid_step_ids = []
        if isinstance(plan_dict, dict):
            raw_steps = plan_dict.get("steps")
            if isinstance(raw_steps, list):
                for raw_step in raw_steps:
                    if isinstance(raw_step, dict) and isinstance(raw_step.get("id"), str):
                        valid_step_ids.append(raw_step["id"])
        return (
            "PLAN 已生成，进入 EXECUTE 阶段。\n"
            "必须只输出严格 JSON: {\"thought\": \"...\", \"plan_step_id\": \"...\", "
            "\"tool\": \"...\", \"args\": {}}。\n"
            f"有效 plan_step_id 只能从这些值中选择: {', '.join(valid_step_ids)}。\n"
            "不要再次输出 plan/steps；完成分析或操作后必须调用 finish。\n"
            f"TaskPlan:\n{json.dumps(plan_dict, ensure_ascii=False, indent=2)}"
        )

    def _validate_patch_sequence(
        self,
        tool_call: dict[str, object],
        propose_patch_succeeded: bool,
    ) -> str | None:
        """限制 apply_patch 前提，确保会话内先走 propose_patch。"""
        if tool_call.get("tool") == "apply_patch" and not propose_patch_succeeded:
            return "apply_patch 必须在同一次执行中成功调用 propose_patch 之后才能执行"
        return None

    def _run_tool(
        self,
        tool_call: dict[str, object],
        run_state: RunState,
        user_task: str,
    ) -> dict[str, object]:
        """将 tool_call 标准化后执行工具，统一将 ToolError 转换为失败结果。"""
        prepared_tool_call = self._prepare_tool_call(tool_call, run_state, user_task)
        tool_call.clear()
        tool_call.update(prepared_tool_call)
        try:
            tool_output = self.tool_runner(tool_call)
        except ToolError as error:
            return {"ok": False, "error": str(error)}

        return {"ok": True, "output": tool_output}

    def _prepare_tool_call(
        self,
        tool_call: dict[str, object],
        run_state: RunState,
        user_task: str,
    ) -> dict[str, object]:
        """为 propose_patch 注入审批可追踪元信息，不影响其他工具参数。"""
        if tool_call.get("tool") != "propose_patch":
            return dict(tool_call)

        prepared_tool_call = dict(tool_call)
        tool_args = self._tool_args(tool_call)
        prepared_args = dict(tool_args)
        instruction = prepared_args.get("instruction")
        summary = instruction if isinstance(instruction, str) and instruction.strip() else user_task
        prepared_args["run_id"] = self._current_run_id()
        prepared_args["task"] = user_task
        prepared_args["summary"] = summary
        prepared_args["plan_snapshot"] = self._build_patch_plan_snapshot(run_state.plan)
        if run_state.plan is not None:
            prepared_args["risk_level"] = run_state.plan.risk_level
        prepared_tool_call["args"] = prepared_args
        return prepared_tool_call

    def _current_run_id(self) -> str:
        log_path = getattr(self.run_logger, "log_path", None)
        if log_path is not None:
            return str(log_path)
        payload = getattr(self.run_logger, "payload", None)
        if isinstance(payload, dict):
            started_at = payload.get("started_at")
            if isinstance(started_at, str) and started_at.strip():
                return started_at
        return "agent-run"

    def _build_patch_plan_snapshot(self, plan: TaskPlan | None) -> dict[str, object]:
        if plan is None:
            return {"test_commands": []}

        plan_snapshot = plan.to_dict()
        plan_snapshot["test_commands"] = self._plan_test_commands(plan)
        return plan_snapshot

    def _plan_test_commands(self, plan: TaskPlan) -> list[dict[str, str]]:
        if plan.requires_tests:
            return [{"command_name": "unit"}]
        return []

    def _record_execution_state(
        self,
        run_state: RunState,
        tool_call: dict[str, object],
        tool_result: dict[str, object],
    ) -> RunState:
        plan_step_id = str(tool_call["plan_step_id"])
        updated_state = run_state.with_execution_step(plan_step_id)
        if tool_call.get("tool") == "apply_patch" and tool_result.get("ok") is True:
            tool_output = tool_result.get("output")
            if isinstance(tool_output, dict):
                modified_files = tool_output.get("modified_files")
                if isinstance(modified_files, list):
                    for modified_file in modified_files:
                        if isinstance(modified_file, str) and modified_file.strip():
                            updated_state = updated_state.with_changed_file(modified_file)
        return updated_state

    def _verification_result_from_tool_call(
        self,
        tool_call: dict[str, object],
        tool_result: dict[str, object],
    ) -> VerificationResult | None:
        """将 run_tests 工具输出转换为 VerificationResult。"""
        if tool_call.get("tool") != "run_tests" or tool_result.get("ok") is not True:
            return None
        tool_output = tool_result.get("output")
        if not isinstance(tool_output, dict):
            return VerificationResult(
                command_name="run_tests",
                passed=False,
                exit_code=None,
                output="run_tests output is not an object",
                reasons=("tests_failed",),
                repairable=True,
            )
        exit_code = tool_output.get("exit_code")
        normalized_exit_code = exit_code if isinstance(exit_code, int) else None
        stdout = str(tool_output.get("stdout", ""))
        stderr = str(tool_output.get("stderr", ""))
        passed = normalized_exit_code == 0 and tool_output.get("timed_out") is not True
        return VerificationResult(
            command_name=str(tool_output.get("command_name", "run_tests")),
            passed=passed,
            exit_code=normalized_exit_code,
            output=f"{stdout}\n{stderr}".strip(),
            reasons=() if passed else ("tests_failed",),
            repairable=not passed,
        )

    def _build_stage_tests_result(
        self,
        latest_test_result: VerificationResult | None,
        run_state: RunState,
    ) -> VerificationResult:
        """缺失 run_tests 时给出缺省判定，保证 VERIFY 可继续收敛。"""
        if latest_test_result is not None:
            return latest_test_result
        plan = run_state.plan
        tests_required = bool(run_state.changed_files) or bool(
            plan is not None and plan.requires_tests
        )
        if tests_required:
            return VerificationResult(
                command_name="run_tests",
                passed=False,
                exit_code=None,
                output="run_tests tool call is required but missing",
                reasons=("tests_failed",),
                repairable=True,
            )
        return VerificationResult(
            command_name="not_required",
            passed=True,
            exit_code=0,
            output="no run_tests tool call required by plan",
        )

    def _is_confirmed_apply_patch(
        self,
        tool_call: dict[str, object],
        tool_result: dict[str, object],
    ) -> bool:
        if tool_call.get("tool") != "apply_patch" or tool_result.get("ok") is not True:
            return False
        tool_output = tool_result.get("output")
        return isinstance(tool_output, dict) and tool_output.get("status") == "applied"

    def _is_successful_propose_patch(
        self,
        tool_call: dict[str, object],
        tool_result: dict[str, object],
    ) -> bool:
        if tool_call.get("tool") != "propose_patch" or tool_result.get("ok") is not True:
            return False
        tool_output = tool_result.get("output")
        return isinstance(tool_output, dict) and tool_output.get("ok") is True

    def _verify_finished_run(
        self,
        run_state: RunState,
        context_stats: ContextStats,
        patch_confirmed: bool,
        pending_patch: dict[str, object] | None = None,
    ) -> tuple[RunState, VerificationResult]:
        """完成状态快照后，调用 verifier 的确定性验证流程。"""
        state_with_tests = run_state.with_context_stats(context_stats.to_dict())
        verify_status = verifier.verify_run_state(
            state_with_tests,
            self.repository_tools.repo_root,
            patch_confirmed=patch_confirmed if patch_confirmed else None,
            pending_patch=pending_patch,
        )
        return state_with_tests, verify_status

    def _pending_patch_from_tool_result(self, tool_result: dict[str, object]) -> dict[str, object]:
        tool_output = tool_result.get("output")
        if not isinstance(tool_output, dict):
            return {}

        patch_id = tool_output.get("patch_id")
        target_files = tool_output.get("target_files")
        diff_preview = tool_output.get("diff_preview")
        next_commands = tool_output.get("next_commands")
        paths = tool_output.get("paths")
        return {
            "status": "pending_approval",
            "patch_id": patch_id if isinstance(patch_id, str) else "",
            "target_files": target_files if isinstance(target_files, list) else [],
            "paths": paths if isinstance(paths, list) else [],
            "diff_preview": diff_preview if isinstance(diff_preview, str) else "",
            "next_commands": next_commands if isinstance(next_commands, list) else [],
        }

    def _format_pending_approval_final_answer(
        self,
        pending_patch: dict[str, object],
        run_state: RunState,
        verify_status: VerificationResult,
    ) -> str:
        patch_id = str(pending_patch.get("patch_id", ""))
        target_file_lines = self._format_pending_target_files(pending_patch)
        diff_preview = str(pending_patch.get("diff_preview", "")) or "(empty diff preview)"
        show_command = f"python main.py patch show --repo . {patch_id}"
        apply_command = f"python main.py patch apply --repo . {patch_id}"
        reject_command = f"python main.py patch reject --repo . {patch_id}"
        verification_label = "passed" if verify_status.passed else "failed"
        return "\n".join(
            [
                "补丁已生成但尚未应用，当前运行已进入 awaiting approval。",
                f"patch_id: {patch_id}",
                "target files:",
                target_file_lines,
                "diff preview summary:",
                diff_preview,
                "exact commands:",
                f"- show: {show_command}",
                f"- apply: {apply_command}",
                f"- reject: {reject_command}",
                "",
                "v0.6 summary:",
                f"- execution steps: {', '.join(run_state.execution_steps) or 'none'}",
                "- changed files: none (pending approval, not applied)",
                "- tests: not_run (pending approval)",
                f"- verification: {verification_label} ({verify_status.command_name})",
                f"- repair: attempts={run_state.repair_attempts} status=not_attempted",
            ]
        )

    def _format_pending_target_files(self, pending_patch: dict[str, object]) -> str:
        target_files = pending_patch.get("target_files")
        if isinstance(target_files, list) and target_files:
            lines: list[str] = []
            for target_file in target_files:
                if isinstance(target_file, dict):
                    path = target_file.get("path")
                    operation = target_file.get("operation", "modify")
                    if isinstance(path, str) and path:
                        lines.append(f"- {path} ({operation})")
                elif isinstance(target_file, str) and target_file:
                    lines.append(f"- {target_file}")
            if lines:
                return "\n".join(lines)

        paths = pending_patch.get("paths")
        if isinstance(paths, list) and paths:
            return "\n".join(f"- {path}" for path in paths if isinstance(path, str))
        return "- none"

    def _should_repair(
        self,
        verify_status: VerificationResult,
        run_state: RunState,
    ) -> bool:
        """仅 edit/refactor 且 tests_failed 且第一次失败时允许一次 repair。"""
        plan = run_state.plan
        return (
            plan is not None
            and plan.task_type in {"edit", "refactor"}
            and verify_status.repairable
            and verify_status.reasons == ("tests_failed",)
            and run_state.repair_attempts == 0
        )

    def _format_verify_feedback(self, verify_status: VerificationResult) -> str:
        """失败时输出可继续执行 repair 的模型提示文本。"""
        return (
            "VERIFY 失败且仅由 tests_failed 导致，允许一次 REPAIR。\n"
            f"验证结果:\n{json.dumps(verify_status.to_dict(), ensure_ascii=False, indent=2)}\n"
            "继续。只输出下一步严格 JSON，且必须包含有效 plan_step_id。"
        )

    def _format_verify_error(self, verify_status: VerificationResult) -> str:
        return "VERIFY 失败: " + json.dumps(
            verify_status.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

    def _format_final_answer(
        self,
        base_answer: str,
        run_state: RunState,
        tests_result: VerificationResult,
        verify_status: VerificationResult,
    ) -> str:
        state_tests_result = run_state.tests_result or tests_result
        execution_steps = ", ".join(run_state.execution_steps) or "none"
        changed_files = ", ".join(run_state.changed_files) or "none"
        tests_label = "passed" if state_tests_result.passed else "failed"
        verification_label = "passed" if verify_status.passed else "failed"
        repair_status = "attempted" if run_state.repair_attempts else "not_attempted"
        return "\n".join(
            [
                base_answer,
                "",
                "v0.6 summary:",
                f"- execution steps: {execution_steps}",
                f"- changed files: {changed_files}",
                f"- tests: {tests_label} ({state_tests_result.command_name})",
                f"- verification: {verification_label} ({verify_status.command_name})",
                f"- repair: attempts={run_state.repair_attempts} status={repair_status}",
            ]
        )

    def _update_logger_state_payload(
        self,
        run_state: RunState,
        verify_status: VerificationResult | None,
    ) -> None:
        payload = getattr(self.run_logger, "payload", None)
        if not isinstance(payload, dict):
            return
        state_dict = run_state.to_dict()
        payload["stage"] = state_dict["stage"]
        payload["stage_history"] = state_dict["stage_history"]
        payload["plan"] = state_dict["plan"]
        payload["verify_status"] = (
            verify_status.to_dict() if verify_status is not None else None
        )
        payload["repair_attempts"] = state_dict["repair_attempts"]
        save = getattr(self.run_logger, "save", None)
        if callable(save):
            save()

    def _format_tool_feedback(self, tool_result: dict[str, object]) -> str:
        serialized_result = self._serialize_tool_result(tool_result)
        return self._format_serialized_tool_feedback(serialized_result)

    def _serialize_tool_result(self, tool_result: dict[str, object]) -> str:
        return json.dumps(tool_result, ensure_ascii=False, indent=2)

    def _format_serialized_tool_feedback(self, serialized_result: str) -> str:
        return f"工具结果:\n{serialized_result}\n继续。只输出下一步严格 JSON。"

    def _update_context_stats(
        self,
        context_stats: ContextStats,
        tool_call: dict[str, object] | None,
        tool_result: dict[str, object],
        serialized_result: str,
    ) -> ContextStats:
        """更新 step、消息、文件读取与搜索调用统计，用于后续预算评估。"""
        steps_used = context_stats.steps_used + 1
        total_tool_output_chars = (
            context_stats.total_tool_output_chars + len(serialized_result)
        )
        files_read = context_stats.files_read
        full_file_reads = context_stats.full_file_reads
        ranges_read = context_stats.ranges_read
        search_calls = context_stats.search_calls

        if tool_call is not None:
            tool_name = str(tool_call["tool"])
            tool_args = self._tool_args(tool_call)
            if tool_name == "search_text":
                search_calls += 1
            if tool_result.get("ok") is True:
                path_value = tool_args.get("path")
                if tool_name == "read_file" and isinstance(path_value, str):
                    files_read = (*files_read, path_value)
                    full_file_reads = (*full_file_reads, path_value)
                if tool_name == "read_file_range" and isinstance(path_value, str):
                    start_line = tool_args.get("start_line")
                    end_line = tool_args.get("end_line")
                    if isinstance(start_line, int) and isinstance(end_line, int):
                        ranges_read = (
                            *ranges_read,
                            {
                                "path": path_value,
                                "start_line": start_line,
                                "end_line": end_line,
                            },
                        )

        return replace(
            context_stats,
            steps_used=steps_used,
            total_tool_output_chars=total_tool_output_chars,
            files_read=files_read,
            ranges_read=ranges_read,
            search_calls=search_calls,
            full_file_reads=full_file_reads,
        )

    def _tool_args(self, tool_call: dict[str, object]) -> dict[str, object]:
        tool_args = tool_call.get("args")
        if isinstance(tool_args, dict):
            return tool_args
        return {}

    def _build_observation_summary(
        self,
        tool_call: dict[str, object] | None,
        tool_result: dict[str, object],
        serialized_result: str,
        message_index: int,
    ) -> dict[str, object]:
        tool_name = "parse_error"
        path_text = "-"
        if tool_call is not None:
            tool_name = str(tool_call["tool"])
            path_value = self._tool_args(tool_call).get("path")
            if isinstance(path_value, str) and path_value:
                path_text = path_value

        return {
            "message_index": message_index,
            "content": (
                f"[compact observation] tool={tool_name} path={path_text} "
                f"chars={len(serialized_result)} ok={str(tool_result.get('ok') is True).lower()}"
            ),
        }

    def _messages_for_llm(
        self,
        messages: list[dict[str, str]],
        observation_summaries: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        """压缩早期观测为摘要片段，减少重复上下文噪音。"""
        compact_message_indexes = {
            message_index
            for summary in observation_summaries[:-FULL_OBSERVATION_FEEDBACK_COUNT]
            for message_index in [self._summary_message_index(summary)]
        }
        summary_by_index = {
            message_index: str(summary["content"])
            for summary in observation_summaries
            for message_index in [self._summary_message_index(summary)]
            if message_index in compact_message_indexes
        }

        messages_for_llm: list[dict[str, str]] = []
        for message_index, message in enumerate(messages):
            content = message["content"]
            if message_index in summary_by_index:
                content = summary_by_index[message_index]
            messages_for_llm.append({"role": message["role"], "content": content})
        return messages_for_llm

    def _summary_message_index(self, summary: dict[str, object]) -> int:
        """将 observation summary 映射为消息下标，异常类型用于早露。

该异常不视作程序错误，是上游 summary 数据结构损坏的防线。
"""
        message_index = summary["message_index"]
        if not isinstance(message_index, int) or isinstance(message_index, bool):
            raise RuntimeError("observation summary message_index 必须是整数")
        return message_index

    def _messages_total_chars(self, messages: list[dict[str, str]]) -> int:
        """粗粒度累计本轮消息长度，用于上下文预算统计。"""
        return sum(len(message["role"]) + len(message["content"]) for message in messages)
