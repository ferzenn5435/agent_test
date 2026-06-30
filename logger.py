"""运行日志。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import LOG_DIR


class RunLogger:
    """将一次 agent 运行记录为 JSON 文件。"""

    def __init__(self, repo_path: Path, user_task: str, log_dir: str | Path = LOG_DIR) -> None:
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
            "repair_attempt": 0,
            "repair_attempts": 0,
            "verify_status": None,
            "steps": [],
            "final_answer": None,
            "error": None,
            "context_stats": None,
        }
        self.save()

    def record_step(
        self,
        step_number: int,
        model_output: str,
        tool_call: dict[str, object] | None,
        tool_result: dict[str, object],
    ) -> None:
        """记录单步模型输出、工具调用和工具结果。"""

        steps = self.payload["steps"]
        if not isinstance(steps, list):
            raise RuntimeError("日志结构错误: steps 不是列表")

        step_payload: dict[str, object] = {
            "step": step_number,
            "model_output": model_output,
            "tool_call": tool_call,
            "tool_result": tool_result,
        }
        if isinstance(tool_call, dict):
            plan_step_id = tool_call.get("plan_step_id")
            if plan_step_id is not None:
                step_payload["plan_step_id"] = plan_step_id
                self.payload["plan_step_id"] = plan_step_id

        steps.append(step_payload)
        self.save()

    def set_final_answer(self, final_answer: str) -> None:
        """记录最终答案。"""

        self.payload["final_answer"] = final_answer
        self.save()

    def set_error(self, error_message: str) -> None:
        """记录运行错误。"""

        self.payload["error"] = error_message
        self.save()

    def set_context_stats(self, stats: dict[str, object]) -> None:
        """记录 agent 提供的最终上下文统计。"""

        self.payload["context_stats"] = stats
        self.save()

    def save(self) -> None:
        """保存日志文件。"""

        serialized_log = json.dumps(self.payload, ensure_ascii=False, indent=2)
        self.log_path.write_text(serialized_log, encoding="utf-8")
