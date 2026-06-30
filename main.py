"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent import CodeAnalysisAgent
from config import ConfigError, MAX_STEPS, load_llm_config_from_env
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from tools import RepositoryTools, ToolError
from eval_safety import EvalSafetyError, validate_eval_temp_repo


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

    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv and raw_argv[0] == "patch":
        return _parse_patch_args(raw_argv[1:])

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
        command="answer",
        repo_path=repo_path,
        question=" ".join(question_parts),
        max_steps=args.max_steps,
    )


def _parse_patch_args(argv: list[str]) -> argparse.Namespace:
    """解析确定性 patch 子命令参数。"""

    parser = argparse.ArgumentParser(description="管理已保存的 patch 提案")
    subparsers = parser.add_subparsers(dest="patch_command", required=True)

    list_parser = subparsers.add_parser("list", help="列出 patch 提案")
    list_parser.add_argument("--repo", dest="repo_path", required=True, help="本地 repo 路径")

    show_parser = subparsers.add_parser("show", help="显示 patch 元数据和 diff")
    show_parser.add_argument("--repo", dest="repo_path", required=True, help="本地 repo 路径")
    show_parser.add_argument("patch_id", help="patch ID")
    show_parser.add_argument("--full", action="store_true", help="显示完整 diff")

    apply_parser = subparsers.add_parser("apply", help="确定性应用 pending patch")
    apply_parser.add_argument("--repo", dest="repo_path", required=True, help="本地 repo 路径")
    apply_parser.add_argument("patch_id", help="patch ID")

    reject_parser = subparsers.add_parser("reject", help="拒绝 pending patch")
    reject_parser.add_argument("--repo", dest="repo_path", required=True, help="本地 repo 路径")
    reject_parser.add_argument("patch_id", help="patch ID")

    args = parser.parse_args(argv)
    return argparse.Namespace(
        command="patch",
        patch_command=args.patch_command,
        repo_path=args.repo_path,
        patch_id=getattr(args, "patch_id", ""),
        full=getattr(args, "full", False),
        max_steps=MAX_STEPS,
        question="",
    )


def _write_json_output(payload: dict[str, object]) -> None:
    """输出稳定 JSON，方便 CLI 使用方和测试读取。"""

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _patch_error_text(result: dict[str, object]) -> str:
    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)
    return "patch command failed"


def _run_patch_command(args: argparse.Namespace) -> int:
    """运行不依赖 LLM/agent 的确定性 patch 子命令。"""

    try:
        repository_tools = RepositoryTools(Path(args.repo_path).expanduser().resolve())
        if args.patch_command == "list":
            result = repository_tools.list_patches()
            patches = result.get("patches", [])
            if not isinstance(patches, list):
                patches = []
            output: dict[str, object] = {
                "ok": result.get("ok") is True,
                "patches": [
                    {
                        "patch_id": patch.get("patch_id", ""),
                        "status": patch.get("status", ""),
                        "created_at": patch.get("created_at", ""),
                        "summary": patch.get("summary", ""),
                        "target_files": patch.get("target_files", []),
                    }
                    for patch in patches
                    if isinstance(patch, dict)
                ],
                "errors": result.get("errors", []),
            }
        elif args.patch_command == "show":
            output = repository_tools.show_patch(args.patch_id, full=args.full)
        elif args.patch_command == "apply":
            output = repository_tools.apply_pending_patch(args.patch_id)
        elif args.patch_command == "reject":
            output = repository_tools.reject_patch(args.patch_id)
        else:
            print(f"错误: unsupported patch command {args.patch_command}", file=sys.stderr)
            return 1
    except (OSError, ToolError) as error:
        print(f"patch command failed: {error}", file=sys.stderr)
        return 1

    if output.get("ok") is not True:
        print(f"patch command failed: {_patch_error_text(output)}", file=sys.stderr)
        return 1

    _write_json_output(output)
    return 0


class CliApplyPatchApproval:
    """CLI 模式下的 apply_patch 确认门。"""

    def __init__(
        self,
        repository_tools: RepositoryTools,
        approval_mode: str = "manual",
        eval_run_id: str | None = None,
    ) -> None:
        if approval_mode not in {"manual_pending", "manual", "auto_for_eval"}:
            raise ValueError(
                "approval_mode must be 'manual_pending', 'manual', or 'auto_for_eval'"
            )

        self.repository_tools = repository_tools
        self.approval_mode = approval_mode
        self.eval_run_id = eval_run_id

    def run_tool(self, tool_call: dict[str, object]) -> object:
        """执行工具调用，并在 apply_patch 前要求人工确认。"""

        if tool_call.get("tool") != "apply_patch":
            return self.repository_tools.run_tool(tool_call)

        patch_id = self._extract_patch_id(tool_call)
        patch_info = self._load_patch_info(patch_id)
        if self.approval_mode == "auto_for_eval":
            return self._run_auto_for_eval_apply(tool_call, patch_id)
        if self.approval_mode == "manual_pending":
            return self._manual_pending_apply_result(patch_info)

        self._print_confirmation_prompt(patch_info)
        approval_text = input("Approve apply_patch? [yes/y/approve]: ")
        approved = approval_text.strip().lower() in APPROVAL_TOKENS
        self._log_confirmation_decision(patch_id, approved, self.approval_mode)

        if not approved:
            raise ToolError(
                f"apply_patch rejected by user for patch_id {patch_id}; "
                "no files were modified"
            )

        return self.repository_tools.run_tool(tool_call)

    def _run_auto_for_eval_apply(
        self,
        tool_call: dict[str, object],
        patch_id: str,
    ) -> object:
        eval_run_id = self.eval_run_id if self.eval_run_id is not None else ""
        try:
            validate_eval_temp_repo(self.repository_tools.repo_root, eval_run_id)
        except EvalSafetyError as error:
            raise ToolError(str(error)) from error

        self._log_confirmation_decision(patch_id, True, self.approval_mode)
        return self.repository_tools.run_tool(tool_call)

    def _manual_pending_apply_result(self, patch_info: dict[str, Any]) -> dict[str, object]:
        patch_id = str(patch_info["patch_id"])
        return {
            "ok": True,
            "status": "pending_approval",
            "patch_id": patch_id,
            "applied": False,
            "target_files": patch_info["target_files"],
            "paths": patch_info["paths"],
            "diff_preview": patch_info["preview"],
            "next_commands": [
                f"python main.py patch show --repo . {patch_id}",
                f"python main.py patch apply --repo . {patch_id}",
                f"python main.py patch reject --repo . {patch_id}",
            ],
        }

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
        target_files = metadata.get("target_files")
        if not isinstance(target_files, list):
            target_files = []
        return {
            "patch_id": patch_id,
            "patch_path": self.repository_tools._format_repo_path(patch_path),
            "paths": validated_paths,
            "target_files": target_files,
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

    def _log_confirmation_decision(
        self,
        patch_id: str,
        approved: bool,
        approval_mode: str,
    ) -> None:
        status = "approved" if approved else "rejected"
        details: dict[str, object] = {"approved": approved}
        if approval_mode == "auto_for_eval":
            details["approval_mode"] = approval_mode
        self.repository_tools._append_run_event(
            patch_id=patch_id,
            event_type="apply_confirmation",
            status=status,
            details=details,
        )


def main() -> int:
    """运行命令行程序。"""

    args = parse_args()
    if getattr(args, "command", "answer") == "patch":
        return _run_patch_command(args)

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
        approval_gate = CliApplyPatchApproval(
            repository_tools,
            approval_mode="manual_pending",
        )
        agent = CodeAnalysisAgent(
            llm_client=llm_client,
            repository_tools=repository_tools,
            run_logger=run_logger,
            max_steps=args.max_steps,
            tool_runner=approval_gate.run_tool,
            pending_approval_mode=True,
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
