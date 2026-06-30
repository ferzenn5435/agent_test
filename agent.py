"""LLM + 工具调用循环，默认步数来自 MAX_STEPS。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from config import MAX_STEPS
from context_stats import ContextStats
from llm_client import LlmClient
from logger import RunLogger
import planner
from prompts import build_system_prompt
from run_state import RunState
from schemas import VerificationResult
from tools import RepositoryTools, ToolError
import verifier


FULL_OBSERVATION_FEEDBACK_COUNT = 3


class AgentStepError(ValueError):
    """模型单步输出不符合协议。"""


class CodeAnalysisAgent:
    """最小本地代码库分析 agent。"""

    def __init__(
        self,
        llm_client: LlmClient,
        repository_tools: RepositoryTools,
        run_logger: RunLogger,
        max_steps: int = MAX_STEPS,
        tool_runner: Callable[[dict[str, object]], object] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.repository_tools = repository_tools
        self.run_logger = run_logger
        self.max_steps = max_steps
        self.tool_runner = tool_runner or repository_tools.run_tool
        self.context_stats = ContextStats()

    def answer(self, question: str) -> str:
        """运行 agent 并返回最终答案。"""

        context_stats = ContextStats()
        run_state = RunState()
        latest_test_result: VerificationResult | None = None
        patch_confirmed = False
        propose_patch_succeeded = False
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
            inspect_repo_result = self._inspect_repo_safely()
            run_state = run_state.transition("PLAN")
            plan = planner.create_plan(
                question,
                inspect_repo_result,
                self.max_steps,
                self.llm_client,
            )
            run_state = run_state.with_plan(plan)
            self._update_logger_state_payload(run_state, verify_status=None)
            run_state = run_state.transition("EXECUTE")

            step_limit = min(self.max_steps, plan.max_steps)
            step_number = 1
            while step_number <= step_limit:
                messages_for_llm = self._messages_for_llm(messages, observation_summaries)
                context_stats = replace(
                    context_stats,
                    messages_total_chars=self._messages_total_chars(messages_for_llm),
                )
                self.context_stats = context_stats
                model_output = self.llm_client.chat(messages_for_llm)
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
                    tool_result = self._run_tool(tool_call)
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
                )

                if tool_call["tool"] == "finish" and tool_result["ok"] is True:
                    final_answer = str(tool_result["output"])
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
        except Exception:
            self.run_logger.set_context_stats(context_stats.to_dict())
            raise

        final_answer = f"达到最大循环步数 {self.max_steps}，模型未调用 finish(answer) 完成任务。"
        self.run_logger.set_context_stats(context_stats.to_dict())
        self.run_logger.set_error(final_answer)
        self.run_logger.set_final_answer(final_answer)
        return final_answer

    def _inspect_repo_safely(self) -> dict[str, object]:
        try:
            return self.repository_tools.inspect_repo()
        except (OSError, ToolError) as error:
            return {"ok": False, "error": str(error)}

    def _parse_model_output(self, model_output: str) -> dict[str, object]:
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
        raw_plan_step_id = tool_call.get("plan_step_id")
        if not isinstance(raw_plan_step_id, str) or not raw_plan_step_id.strip():
            return "plan_step_id 必须是计划中存在的非空字符串"
        plan = run_state.plan
        if plan is None:
            return "执行前缺少 plan"
        legal_step_ids = {step.id for step in plan.steps}
        if raw_plan_step_id.strip() not in legal_step_ids:
            return f"未知 plan_step_id: {raw_plan_step_id}"
        tool_call["plan_step_id"] = raw_plan_step_id.strip()
        return None

    def _validate_patch_sequence(
        self,
        tool_call: dict[str, object],
        propose_patch_succeeded: bool,
    ) -> str | None:
        if tool_call.get("tool") == "apply_patch" and not propose_patch_succeeded:
            return "apply_patch 必须在同一次执行中成功调用 propose_patch 之后才能执行"
        return None

    def _run_tool(self, tool_call: dict[str, object]) -> dict[str, object]:
        try:
            tool_output = self.tool_runner(tool_call)
        except ToolError as error:
            return {"ok": False, "error": str(error)}

        return {"ok": True, "output": tool_output}

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
        return isinstance(tool_output, dict) and tool_output.get("ok") is True

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
    ) -> tuple[RunState, VerificationResult]:
        state_with_tests = run_state.with_context_stats(context_stats.to_dict())
        verify_status = verifier.verify_run_state(
            state_with_tests,
            self.repository_tools.repo_root,
            patch_confirmed=patch_confirmed if patch_confirmed else None,
        )
        return state_with_tests, verify_status

    def _should_repair(
        self,
        verify_status: VerificationResult,
        run_state: RunState,
    ) -> bool:
        plan = run_state.plan
        return (
            plan is not None
            and plan.task_type in {"edit", "refactor"}
            and verify_status.repairable
            and verify_status.reasons == ("tests_failed",)
            and run_state.repair_attempts == 0
        )

    def _format_verify_feedback(self, verify_status: VerificationResult) -> str:
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
        payload["repair_attempt"] = state_dict["repair_attempts"]
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
        message_index = summary["message_index"]
        if not isinstance(message_index, int) or isinstance(message_index, bool):
            raise RuntimeError("observation summary message_index 必须是整数")
        return message_index

    def _messages_total_chars(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["role"]) + len(message["content"]) for message in messages)
