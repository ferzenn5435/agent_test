"""命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from agent import CodeAnalysisAgent
from config import ConfigError, MAX_STEPS, load_llm_config_from_env
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from tools import RepositoryTools, ToolError


APPROVAL_TOKENS = {"yes", "y", "approve"}


def _positive_int(raw_value: str) -> int:
    """解析正整数命令行参数。"""

    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "max_steps must be a positive integer"
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("max_steps must be a positive integer")

    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="最小本地代码库分析 agent")
    parser.add_argument("--repo", dest="repo_path", help="要分析的本地 repo 路径")
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=MAX_STEPS,
        help=f"最大工具调用步数 (默认: {MAX_STEPS})",
    )
    parser.add_argument("arguments", nargs="*", help="位置参数：repo 路径和用户问题")
    args = parser.parse_args(argv)

    if args.repo_path is None:
        if len(args.arguments) < 2:
            parser.error(
                "缺少参数：请使用 python main.py <repo_path> <question>，"
                "或 python main.py --repo <repo_path> <question>"
            )
        repo_path = args.arguments[0]
        question_parts = args.arguments[1:]
    else:
        if not args.arguments:
            parser.error(
                "缺少参数：使用 --repo 时必须提供 question，"
                "例如 python main.py --repo . \"工具调用是如何实现的？\""
            )
        repo_path = args.repo_path
        question_parts = args.arguments

    return argparse.Namespace(
        repo_path=repo_path,
        question=" ".join(question_parts),
        max_steps=args.max_steps,
    )


class CliApplyPatchApproval:
    """CLI 模式下的 apply_patch 人工确认门。"""

    def __init__(self, repository_tools: RepositoryTools) -> None:
        self.repository_tools = repository_tools

    def run_tool(self, tool_call: dict[str, object]) -> object:
        """执行工具调用，并在 apply_patch 前要求人工确认。"""

        if tool_call.get("tool") != "apply_patch":
            return self.repository_tools.run_tool(tool_call)

        patch_id = self._extract_patch_id(tool_call)
        patch_info = self._load_patch_info(patch_id)
        self._print_confirmation_prompt(patch_info)
        approval_text = input("Approve apply_patch? [yes/y/approve]: ")
        approved = approval_text.strip().lower() in APPROVAL_TOKENS
        self._log_confirmation_decision(patch_id, approved)

        if not approved:
            raise ToolError(
                f"apply_patch rejected by user for patch_id {patch_id}; "
                "no files were modified"
            )

        return self.repository_tools.run_tool(tool_call)

    def _extract_patch_id(self, tool_call: dict[str, object]) -> str:
        tool_args = tool_call.get("args")
        if not isinstance(tool_args, dict):
            raise ToolError("args 必须是对象")

        patch_id = tool_args.get("patch_id")
        if not isinstance(patch_id, str) or not patch_id.strip():
            raise ToolError("patch_id 必须是非空字符串")

        return patch_id.strip()

    def _load_patch_info(self, patch_id: str) -> dict[str, Any]:
        self.repository_tools._validate_patch_id(patch_id)
        patch_text = self.repository_tools._read_patch_file(patch_id)
        metadata = self.repository_tools._read_patch_metadata(patch_id)
        validated_paths = self.repository_tools.validate_unified_diff(patch_text)
        metadata_paths = metadata.get("paths")
        if not isinstance(metadata_paths, list) or not all(
            isinstance(path, str) for path in metadata_paths
        ):
            raise ToolError("metadata.paths must be a list of strings")
        if metadata_paths != validated_paths:
            raise ToolError("metadata.paths do not match saved patch diff targets")

        warnings = metadata.get("warnings")
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            warnings = []

        patch_path = self.repository_tools._get_patch_file_path(patch_id)
        return {
            "patch_id": patch_id,
            "patch_path": self.repository_tools._format_repo_path(patch_path),
            "paths": validated_paths,
            "warnings": warnings,
            "preview": self.repository_tools._generate_diff_preview(patch_text),
        }

    def _print_confirmation_prompt(self, patch_info: dict[str, Any]) -> None:
        print("\napply_patch requires human approval before files are modified.")
        print(f"Patch ID: {patch_info['patch_id']}")
        print(f"Patch path: {patch_info['patch_path']}")
        print("Touched paths:")
        for touched_path in patch_info["paths"]:
            print(f"- {touched_path}")
        print("Risk warnings:")
        warnings = patch_info["warnings"] or [
            "Review this patch carefully before applying it."
        ]
        for warning in warnings:
            print(f"- {warning}")
        print("Patch preview:")
        print(str(patch_info["preview"]))

    def _log_confirmation_decision(self, patch_id: str, approved: bool) -> None:
        status = "approved" if approved else "rejected"
        self.repository_tools._append_run_event(
            patch_id=patch_id,
            event_type="apply_confirmation",
            status=status,
            details={"approved": approved},
        )


def main() -> int:
    """运行命令行程序。"""

    args = parse_args()
    repo_path = Path(args.repo_path).expanduser().resolve()
    question = str(args.question).strip()
    if not question:
        print("错误: question 不能为空", file=sys.stderr)
        return 1

    run_logger: RunLogger | None = None
    try:
        repository_tools = RepositoryTools(repo_path)
        run_logger = RunLogger(repo_path=repo_path, user_task=question)
        llm_config = load_llm_config_from_env()
        llm_client = LlmClient(llm_config)
        approval_gate = CliApplyPatchApproval(repository_tools)
        agent = CodeAnalysisAgent(
            llm_client=llm_client,
            repository_tools=repository_tools,
            run_logger=run_logger,
            max_steps=args.max_steps,
            tool_runner=approval_gate.run_tool,
        )
        final_answer = agent.answer(question)
    except (ConfigError, LlmClientError, ToolError, OSError) as error:
        error_message = f"运行失败: {error}"
        if run_logger is not None:
            run_logger.set_error(error_message)
        print(error_message, file=sys.stderr)
        return 1

    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
