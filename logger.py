"""运行日志。

该模块仅负责把一次 agent 运行持久化为 JSON，字段 schema 固定且由上层
主动填充。

边界说明：
- Logger 不做日志脱敏，敏感值脱敏应在上游 error 与调用链里完成；
- 统计字段（如 usage）只是接收端传入值，logger 本身不重复计算。
- 任何 step 记录都立刻 `save()`，便于外部审计或异常中途复盘。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import LOG_DIR


DEFAULT_PROMPT_VERSION = "v0.6"
DEFAULT_USAGE_SUMMARY: dict[str, object] = {
    "llm_call_count": 0,
    "total_latency_ms": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_tokens": 0,
    "estimated_cost": None,
}


class RunLogger:
    """将一次 agent 运行记录为 JSON 文件。

`payload` 采用可追加扩展的平面结构（阶段、步骤、错误、usage 汇总等），
用于 CLI、agent 及评测链路统一消费。
    """

    def __init__(self, repo_path: Path, user_task: str, log_dir: str | Path = LOG_DIR) -> None:
        """创建 run 文件并写入初始化阶段。

初始化字段包括 stage/stage_history/plan 状态位/usage 初始化值；
后续调用各 setter 只做局部覆盖，避免丢失既有字段。
        """
        started_at = datetime.now().astimezone()
        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)

        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        self.log_path = log_root / f"run_{timestamp}.json"
        self.payload: dict[str, object] = {
            "started_at": started_at.isoformat(),
            "repo_path": str(repo_path),
            "user_task": user_task,
            "stage": "INIT",
            "stage_history": ["INIT"],
            "plan": None,
            "plan_step_id": None,
            "repair_attempts": 0,
            "verify_status": None,
            "steps": [],
            "final_answer": None,
            "error": None,
            "context_stats": None,
            "model_profile": None,
            "provider": None,
            "model": None,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "usage_summary": dict(DEFAULT_USAGE_SUMMARY),
        }
        self.save()

    def record_step(
        self,
        step_number: int,
        model_output: str,
        tool_call: dict[str, object] | None,
        tool_result: dict[str, object],
        latency_ms: float | None = None,
        usage: dict[str, object] | None = None,
    ) -> None:
        """记录单步模型输出、工具调用和工具结果。

该方法是唯一持续追加 `steps` 的入口：
- `step` 号、模型输出、工具输入输出与延迟
- 选填 usage（每步上报）
- 若 tool_call 含 `plan_step_id` 会回写到 run 根节点，保持当前计划步骤定位。
        """

        steps = self.payload["steps"]
        if not isinstance(steps, list):
            raise RuntimeError("日志结构错误: steps 不是列表")

        step_payload: dict[str, object] = {
            "step": step_number,
            "model_output": model_output,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "latency_ms": latency_ms,
            "usage": dict(usage) if usage is not None else None,
        }
        if isinstance(tool_call, dict):
            plan_step_id = tool_call.get("plan_step_id")
            if plan_step_id is not None:
                step_payload["plan_step_id"] = plan_step_id
                self.payload["plan_step_id"] = plan_step_id

        steps.append(step_payload)
        self.save()

    def set_final_answer(self, final_answer: str) -> None:
        """记录最终答案并持久化。"""

        self.payload["final_answer"] = final_answer
        self.save()

    def set_error(self, error_message: str) -> None:
        """记录运行错误并持久化。"""

        self.payload["error"] = error_message
        self.save()

    def set_context_stats(self, stats: dict[str, object]) -> None:
        """记录 agent 提供的最终上下文统计。"""

        self.payload["context_stats"] = stats
        self.save()

    def set_usage_summary(
        self,
        *,
        model_profile: str | None,
        provider: str | None,
        model: str | None,
        prompt_version: str | None,
        usage_summary: dict[str, object],
    ) -> None:
        """记录 agent 提供的 LLM usage 汇总。

`usage_summary` 直接采用 `UsageSummary.to_dict()` 的结果；logger 不在此
阶段二次计算 token 或 cost。
        """

        self.payload["model_profile"] = model_profile
        self.payload["provider"] = provider
        self.payload["model"] = model
        self.payload["prompt_version"] = prompt_version
        self.payload["usage_summary"] = dict(usage_summary)
        self.save()

    def save(self) -> None:
        """落盘 JSON 日志。

输出使用 UTF-8 与 ensure_ascii=False，保留中文字段可读性；
任何 IO 错误会抛给调用方，由上层处理。
        """

        serialized_log = json.dumps(self.payload, ensure_ascii=False, indent=2)
        self.log_path.write_text(serialized_log, encoding="utf-8")
