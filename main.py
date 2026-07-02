"""命令行入口与运行编排。

本文件只负责以下三件事：

1. 解析用户输入参数，支持 answer 主流程与 patch 子命令两条互斥路径。
2. 在 answer 流程中组装 `RepositoryTools + LlmClient + CliApplyPatchApproval +
   CodeAnalysisAgent`，形成一次完整的运行链路。
3. 将 apply_patch 的执行动作固定在 CLI 的人工审批边界内，避免在主流程中
   无条件应用文件变更。

关键边界：
- `main.py patch ...` 完全不进入 LLM/Config/Logger 的初始化，保持确定性路径；
- answer 流程对异常做统一兜底，只写入日志并返回非零码，避免进程崩溃。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent import CodeAnalysisAgent
from config import ConfigError, MAX_STEPS
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from tools import RepositoryTools, ToolError
from eval_safety import EvalSafetyError, validate_eval_temp_repo


APPROVAL_TOKENS = {"yes", "y", "approve"}


def _positive_int(raw_value: str) -> int:
    """解析正整数命令行参数。

用于 `--max-steps` 的严格类型校验；如果用户输入非整数或非正值，返回
`argparse.ArgumentTypeError` 让 CLI 直接打印帮助与错误信息。
    """

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
    """解析 answer 主流程参数。

    解析策略保留两种公开入口形式：既支持 `python main.py <repo> <question>`
    的位置参数，也支持 `--repo` 显式参数。

返回值统一归一化为一个 `Namespace`，下游只读取:
- command
- repo_path
- question
- max_steps
- model_profile
    """

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
    parser.add_argument(
        "--model-profile",
        default="default",
        help="模型 profile 名称 (默认: default)",
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
        model_profile=args.model_profile,
    )


def _parse_patch_args(argv: list[str]) -> argparse.Namespace:
    """解析确定性 patch 子命令参数。

此分支用于 `python main.py patch <list|show|apply|reject>`，其目标是让
patch 生命周期完全与 answer 主流程隔离：不读取 `.env`、不加载 profile、
不创建 LLM 客户端，也不发起网络调用。
    """

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
    """执行 patch 子命令并输出标准 JSON。

执行边界：
- `list/show/apply/reject` 全部在本地 repo_tools 完成。
- 成功仅返回 0；失败统一输出 stderr 并返回 1。
- `apply/reject` 之后的副作用由 tools 层约束（备份、回滚、白名单测试）。
    """

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
            output = repository_tools.apply_patch(args.patch_id)
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

    # patch 子命令输出稳定 JSON，便于脚本和测试读取；错误路径只写 stderr，避免混入 JSON。
    _write_json_output(output)
    return 0


class CliApplyPatchApproval:
    """CLI 模式下的 apply_patch 确认门。

`CodeAnalysisAgent` 在 answer 流程中可生成 apply_patch 的工具调用，
该类把真正的“是否落盘”分离为三种模式：
- `manual_pending`：返回 pending_approval，不写文件；
- `manual`：交互式读取用户输入（yes/y/approve）；
- `auto_for_eval`：仅评测上下文下允许自动批准。
"""

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
        """执行工具调用，并对 apply_patch 实施审批边界。

除了 `apply_patch` 外，其他工具保持透传；
`apply_patch` 走人工审批后才交给 RepositoryTools 执行。
        """

        if tool_call.get("tool") != "apply_patch":
            return self.repository_tools.run_tool(tool_call)

        patch_id = self._extract_patch_id(tool_call)
        patch_info = self._load_patch_info(patch_id)
        if self.approval_mode == "auto_for_eval":
            return self._run_auto_for_eval_apply(tool_call, patch_id)
        if self.approval_mode == "manual_pending":
            # 普通 answer 流程默认停在 pending_approval：模型可以提出补丁，
            # 但真实仓库的写入必须由用户后续显式运行 patch apply 完成。
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
        """评测模式下的非交互执行。

在 auto_for_eval 模式里只允许通过 eval 安全目录校验（marker）后执行，
避免误将真实仓库直接作为测试目录执行应用动作。
        """
        eval_run_id = self.eval_run_id if self.eval_run_id is not None else ""
        try:
            validate_eval_temp_repo(self.repository_tools.repo_root, eval_run_id)
        except EvalSafetyError as error:
            raise ToolError(str(error)) from error

        self._log_confirmation_decision(patch_id, True, self.approval_mode)
        return self.repository_tools.run_tool(tool_call)

    def _manual_pending_apply_result(self, patch_info: dict[str, Any]) -> dict[str, object]:
        """返回 pending_approval 响应，并给出用户下一步命令清单。"""

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
        """从工具调用参数中提取 patch_id。

该提取只是签名与边界校验，不涉及文件系统路径拼接，确保未经过滤输入
不会提前触发文件写入。
        """

        tool_args = tool_call.get("args")
        if not isinstance(tool_args, dict):
            raise ToolError("args 必须是对象")

        patch_id = tool_args.get("patch_id")
        if not isinstance(patch_id, str) or not patch_id.strip():
            raise ToolError("patch_id 必须是非空字符串")

        return patch_id.strip()

    def _load_patch_info(self, patch_id: str) -> dict[str, Any]:
        """读取并校验 patch 元数据与 diff 目标文件。

关键边界：
- `metadata.paths` 与 diff 解析出的路径必须一致；
- 不一致时拒绝返回，防止越权修改非预期文件集合。
        """
        self.repository_tools._validate_patch_id(patch_id)
        patch_text = self.repository_tools._read_new_patch_file(patch_id)
        metadata = self.repository_tools._read_new_patch_metadata(patch_id)
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

        patch_path = self.repository_tools._get_new_patch_file_path(patch_id)
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
        """展示待确认差异内容与风险提示。

打印内容只用于用户决策，未在任何环节落盘，属于只读提示层。
        """
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
        """将审批结果追加到事件日志，供审计和追踪。"""
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
    """运行命令行程序。

主流程按以下顺序构建运行上下文：
1. 解析参数。
2. 创建 `RunLogger`（INIT 阶段）并记录 `model_profile`。
3. 构建 `RepositoryTools`、`LlmClient`、`CodeAnalysisAgent`。
4. 通过 approval gate 约束 apply_patch。
5. 输出答案并返回 0，异常时写入 error 并返回 1。
    """

    args = parse_args()
    if getattr(args, "command", "answer") == "patch":
        return _run_patch_command(args)

    repo_path = Path(args.repo_path).expanduser().resolve()
    question = str(args.question).strip()
    model_profile = str(args.model_profile)
    if not question:
        print("错误: question 不能为空", file=sys.stderr)
        return 1

    run_logger: RunLogger | None = None
    try:
        repository_tools = RepositoryTools(repo_path)
        run_logger = RunLogger(repo_path=repo_path, user_task=question)
        run_logger.payload["model_profile"] = model_profile
        run_logger.save()
        llm_client = LlmClient(model_profile=model_profile)
        # 主交互流程只允许生成待审批补丁；确定性 apply 独立放在 `python main.py patch apply`。
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
