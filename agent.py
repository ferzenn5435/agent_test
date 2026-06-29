"""LLM + 工具调用循环。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from config import MAX_STEPS
from context_stats import ContextStats
from llm_client import LlmClient
from logger import RunLogger
from prompts import build_system_prompt
from tools import RepositoryTools, ToolError


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
            for step_number in range(1, self.max_steps + 1):
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
                    continue

                tool_result = self._run_tool(tool_call)
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
                    self.run_logger.set_context_stats(context_stats.to_dict())
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
        except Exception:
            self.run_logger.set_context_stats(context_stats.to_dict())
            raise

        final_answer = f"达到最大循环步数 {self.max_steps}，模型未调用 finish(answer) 完成任务。"
        self.run_logger.set_context_stats(context_stats.to_dict())
        self.run_logger.set_error(final_answer)
        self.run_logger.set_final_answer(final_answer)
        return final_answer

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

        return {
            "thought": thought,
            "tool": tool_name,
            "args": tool_args,
        }

    def _run_tool(self, tool_call: dict[str, object]) -> dict[str, object]:
        try:
            tool_output = self.tool_runner(tool_call)
        except ToolError as error:
            return {"ok": False, "error": str(error)}

        return {"ok": True, "output": tool_output}

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
