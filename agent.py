"""LLM + 工具调用循环。"""

from __future__ import annotations

import json

from config import MAX_STEPS
from llm_client import LlmClient
from logger import RunLogger
from prompts import build_system_prompt
from tools import RepositoryTools, ToolError


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
    ) -> None:
        self.llm_client = llm_client
        self.repository_tools = repository_tools
        self.run_logger = run_logger
        self.max_steps = max_steps

    def answer(self, question: str) -> str:
        """运行 agent 并返回最终答案。"""

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

        for step_number in range(1, self.max_steps + 1):
            model_output = self.llm_client.chat(messages)
            messages.append({"role": "assistant", "content": model_output})

            try:
                tool_call = self._parse_model_output(model_output)
            except AgentStepError as error:
                tool_result = {"ok": False, "error": str(error)}
                self.run_logger.record_step(
                    step_number=step_number,
                    model_output=model_output,
                    tool_call=None,
                    tool_result=tool_result,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": self._format_tool_feedback(tool_result),
                    }
                )
                continue

            tool_result = self._run_tool(tool_call)
            self.run_logger.record_step(
                step_number=step_number,
                model_output=model_output,
                tool_call=tool_call,
                tool_result=tool_result,
            )

            if tool_call["tool"] == "finish" and tool_result["ok"] is True:
                final_answer = str(tool_result["output"])
                self.run_logger.set_final_answer(final_answer)
                return final_answer

            messages.append(
                {
                    "role": "user",
                    "content": self._format_tool_feedback(tool_result),
                }
            )

        final_answer = f"达到最大循环步数 {self.max_steps}，模型未调用 finish(answer) 完成任务。"
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
            tool_output = self.repository_tools.run_tool(tool_call)
        except ToolError as error:
            return {"ok": False, "error": str(error)}

        return {"ok": True, "output": tool_output}

    def _format_tool_feedback(self, tool_result: dict[str, object]) -> str:
        serialized_result = json.dumps(tool_result, ensure_ascii=False, indent=2)
        return f"工具结果:\n{serialized_result}\n继续。只输出下一步严格 JSON。"
