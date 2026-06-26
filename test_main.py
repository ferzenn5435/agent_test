"""命令行参数解析单元测试。"""

from __future__ import annotations

import unittest
import io
import json
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast
from unittest.mock import patch

from config import MAX_STEPS
from agent import CodeAnalysisAgent
from eval_safety import write_eval_temp_marker
from llm_client import LlmClient
from logger import RunLogger
from main import CliApplyPatchApproval, parse_args
from prompts import build_system_prompt
from tools import RepositoryTools, ToolError


V03_TOOL_DESCRIPTIONS = "\n".join(
    [
        "- list_dir: 列出 repo 内目录条目",
        "- read_file: 读取 repo 内 UTF-8 文本文件并返回带行号内容",
        "- read_file_range: 读取 repo 内 UTF-8 文本文件的闭区间行范围",
        "- search_text: 在 repo 内搜索文本关键字",
        "- propose_patch: 提交统一差异补丁草稿",
        "- apply_patch: 在用户确认后应用已保存补丁",
        "- run_tests: 执行白名单测试命令",
        "- finish: 完成任务并输出最终答案",
    ]
)


class MainParserTest(unittest.TestCase):
    """验证 main.py 的 CLI 参数解析行为。"""

    def test_parse_args_uses_default_max_steps(self) -> None:
        args = parse_args(["--repo", ".", "question"])

        self.assertEqual(".", args.repo_path)
        self.assertEqual("question", args.question)
        self.assertEqual(MAX_STEPS, args.max_steps)

    def test_parse_args_accepts_explicit_max_steps_with_repo_option(self) -> None:
        args = parse_args(["--repo", ".", "--max-steps", "3", "question"])

        self.assertEqual(".", args.repo_path)
        self.assertEqual("question", args.question)
        self.assertEqual(3, args.max_steps)

    def test_parse_args_accepts_max_steps_before_positional_repo(self) -> None:
        args = parse_args(["--max-steps", "5", ".", "question"])

        self.assertEqual(".", args.repo_path)
        self.assertEqual("question", args.question)
        self.assertEqual(5, args.max_steps)

    def test_parse_args_rejects_invalid_max_steps(self) -> None:
        for raw_value in ["0", "-1", "abc"]:
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(SystemExit):
                    parse_args(["--repo", ".", "--max-steps", raw_value, "question"])


class TestPromptV03SafetyFlow(unittest.TestCase):
    """验证 v0.3 prompt 的工具契约与人工确认安全流程。"""

    def setUp(self) -> None:
        self.prompt = build_system_prompt(V03_TOOL_DESCRIPTIONS)

    def test_lists_all_v03_tools_and_uses_dynamic_descriptions(self) -> None:
        for tool_name in [
            "list_dir",
            "read_file",
            "read_file_range",
            "search_text",
            "propose_patch",
            "apply_patch",
            "run_tests",
            "finish",
        ]:
            with self.subTest(tool_name=tool_name):
                self.assertIn(tool_name, self.prompt)

        self.assertIn(V03_TOOL_DESCRIPTIONS, self.prompt)
        self.assertIn("tool 只能是", self.prompt)

    def test_requires_read_before_patch_and_propose_before_apply(self) -> None:
        self.assertIn("准备修改代码时，必须先读取相关文件", self.prompt)
        self.assertIn("修改任何代码前，必须先用 read_file 或 read_file_range", self.prompt)
        self.assertIn("任何修改都必须先调用 propose_patch", self.prompt)
        self.assertIn("propose_patch 只保存补丁提案，不修改目标文件", self.prompt)

    def test_apply_requires_cli_approval_and_tests_after_success(self) -> None:
        self.assertIn("apply_patch 需要用户确认", self.prompt)
        self.assertIn("通过 CLI 明确批准", self.prompt)
        self.assertIn("绝对禁止调用 apply_patch", self.prompt)
        self.assertIn("apply_patch 成功后，必须调用 run_tests", self.prompt)
        self.assertIn("command_name: unit 或 compile", self.prompt)

    def test_finish_requires_changed_files_tests_and_manual_review(self) -> None:
        self.assertIn("changed files", self.prompt)
        self.assertIn("test results", self.prompt)
        self.assertIn("manual review", self.prompt)
        self.assertIn("args.answer", self.prompt)

    def test_forbids_out_of_scope_execution_and_agent_frameworks(self) -> None:
        self.assertIn("禁止任意 shell 命令", self.prompt)
        self.assertIn("禁止绕过工具直接读写或修改文件", self.prompt)
        self.assertIn("禁止使用 LangChain、LangGraph、LlamaIndex", self.prompt)
        self.assertIn("禁止 MCP", self.prompt)
        self.assertIn("禁止多 agent", self.prompt)


class TestMainApplyApproval(unittest.TestCase):
    """验证 CLI apply_patch 人工确认门。"""

    def test_default_constructor_uses_manual_approval_and_calls_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            repository_tools, patch_id = self._create_saved_patch(repo_path)
            approval_gate = CliApplyPatchApproval(repository_tools)
            tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

            with patch("builtins.input", return_value="yes") as mocked_input:
                with redirect_stdout(io.StringIO()):
                    tool_output = approval_gate.run_tool(tool_call)

            mocked_input.assert_called_once_with("Approve apply_patch? [yes/y/approve]: ")
            self.assertIsInstance(tool_output, dict)
            apply_output = cast(dict[str, object], tool_output)
            self.assertTrue(apply_output["ok"])
            self.assertEqual("new line\n", (repo_path / "sample.txt").read_text())

    def test_approval_tokens_allow_apply_patch_after_preview(self) -> None:
        for approval_text in ["yes", " Y ", "APPROVE"]:
            with self.subTest(approval_text=approval_text):
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_path = Path(temp_dir)
                    repository_tools, patch_id = self._create_saved_patch(repo_path)
                    approval_gate = CliApplyPatchApproval(repository_tools)
                    tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}
                    stdout_buffer = io.StringIO()

                    with patch("builtins.input", return_value=approval_text):
                        with redirect_stdout(stdout_buffer):
                            tool_output = approval_gate.run_tool(tool_call)

                    self.assertIsInstance(tool_output, dict)
                    apply_output = cast(dict[str, object], tool_output)
                    self.assertTrue(apply_output["ok"])
                    self.assertEqual("new line\n", (repo_path / "sample.txt").read_text())
                    stdout_text = stdout_buffer.getvalue()
                    self.assertIn("Patch path:", stdout_text)
                    self.assertIn("Touched paths:", stdout_text)
                    self.assertIn("sample.txt", stdout_text)
                    self.assertIn("Risk warnings:", stdout_text)
                    self.assertIn("Patch preview:", stdout_text)
                    self.assertIn("-old line", stdout_text)
                    self.assertIn("+new line", stdout_text)
                    events = self._read_run_events(repository_tools, patch_id)
                    self.assertTrue(
                        any(
                            event["event_type"] == "apply_confirmation"
                            and event["status"] == "approved"
                            for event in events
                        )
                    )
                    self.assertTrue(
                        any(event["event_type"] == "apply_success" for event in events)
                    )

    def test_approved_apply_then_run_tests_records_integrated_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            repository_tools, patch_id = self._create_saved_patch(repo_path)
            approval_gate = CliApplyPatchApproval(repository_tools)
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, object()),
                repository_tools=repository_tools,
                run_logger=cast(RunLogger, object()),
                tool_runner=approval_gate.run_tool,
            )
            apply_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

            with patch("builtins.input", return_value="yes"):
                with redirect_stdout(io.StringIO()):
                    apply_result = agent._run_tool(apply_call)

            self.assertTrue(apply_result["ok"], apply_result)
            self.assertEqual("new line\n", (repo_path / "sample.txt").read_text())

            with patch("tools.subprocess.run") as mocked_run:
                mocked_run.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="unit ok",
                    stderr="",
                )
                test_result = repository_tools.run_tests("unit")

            self.assertEqual(0, test_result["exit_code"])
            self.assertEqual("unit ok", test_result["stdout"])
            patch_events = self._read_run_events(repository_tools, patch_id)
            patch_event_types = [str(event["event_type"]) for event in patch_events]
            self.assertEqual(
                ["proposed", "apply_confirmation", "apply_start", "apply_success"],
                patch_event_types,
            )
            self.assertTrue(
                any(
                    event["event_type"] == "apply_confirmation"
                    and event["status"] == "approved"
                    and event["details"] == {"approved": True}
                    for event in patch_events
                )
            )
            metadata_path = repository_tools.repopilot_patches_dir / f"{patch_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual("applied", metadata["status"])
            self.assertEqual(["sample.txt"], metadata["paths"])
            self.assertEqual(["sample.txt"], metadata["modified_files"])
            self.assertIn("backup_dir", metadata)

            all_events = self._read_all_run_events(repository_tools)
            event_types = {str(event["event_type"]) for event in all_events}
            self.assertIn("run_tests_start", event_types)
            self.assertIn("run_tests_end", event_types)
            run_test_end_found = False
            for event in all_events:
                details = event.get("details")
                if (
                    event.get("event_type") == "run_tests_end"
                    and event.get("status") == "ok"
                    and isinstance(details, dict)
                    and details.get("command_name") == "unit"
                    and details.get("exit_code") == 0
                ):
                    run_test_end_found = True
            self.assertTrue(run_test_end_found)

    def test_rejection_tokens_return_agent_error_without_mutation(self) -> None:
        for rejection_text in ["no", "", "maybe"]:
            with self.subTest(rejection_text=rejection_text):
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_path = Path(temp_dir)
                    repository_tools, patch_id = self._create_saved_patch(repo_path)
                    approval_gate = CliApplyPatchApproval(repository_tools)
                    agent = CodeAnalysisAgent(
                        llm_client=cast(LlmClient, object()),
                        repository_tools=repository_tools,
                        run_logger=cast(RunLogger, object()),
                        tool_runner=approval_gate.run_tool,
                    )
                    tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

                    with patch("builtins.input", return_value=rejection_text):
                        with redirect_stdout(io.StringIO()):
                            tool_result = agent._run_tool(tool_call)

                    self.assertFalse(tool_result["ok"])
                    self.assertIsInstance(tool_result["error"], str)
                    error_text = cast(str, tool_result["error"])
                    self.assertIn("rejected by user", error_text)
                    self.assertEqual("old line\n", (repo_path / "sample.txt").read_text())
                    events = self._read_run_events(repository_tools, patch_id)
                    self.assertTrue(
                        any(
                            event["event_type"] == "apply_confirmation"
                            and event["status"] == "rejected"
                            for event in events
                        )
                    )
                    self.assertFalse(
                        any(event["event_type"] == "apply_start" for event in events)
                    )

    def test_stale_metadata_paths_reject_before_approval_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            repository_tools, patch_id = self._create_saved_patch(repo_path)
            metadata_path = repository_tools.repopilot_patches_dir / f"{patch_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["paths"] = ["other.txt"]
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            approval_gate = CliApplyPatchApproval(repository_tools)
            tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

            with patch("builtins.input", return_value="yes") as mocked_input:
                with patch.object(repository_tools, "run_tool") as mocked_run_tool:
                    with self.assertRaises(ToolError) as error_context:
                        with redirect_stdout(io.StringIO()):
                            approval_gate.run_tool(tool_call)

            self.assertIn("metadata.paths do not match", str(error_context.exception))
            mocked_input.assert_not_called()
            mocked_run_tool.assert_not_called()
            self.assertEqual("old line\n", (repo_path / "sample.txt").read_text())
            events = self._read_run_events(repository_tools, patch_id)
            event_types = [str(event["event_type"]) for event in events]
            self.assertEqual(["proposed"], event_types)

    def test_auto_for_eval_rejects_repo_without_marker_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            repository_tools, patch_id = self._create_saved_patch(repo_path)
            approval_gate = CliApplyPatchApproval(
                repository_tools,
                approval_mode="auto_for_eval",
                eval_run_id="run-1",
            )
            tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

            with patch("builtins.input") as mocked_input:
                with patch.object(repository_tools, "run_tool") as mocked_run_tool:
                    with self.assertRaises(ToolError) as error_context:
                        approval_gate.run_tool(tool_call)

            self.assertIn("eval temp marker 不存在", str(error_context.exception))
            mocked_input.assert_not_called()
            mocked_run_tool.assert_not_called()
            self.assertEqual("old line\n", (repo_path / "sample.txt").read_text())
            events = self._read_run_events(repository_tools, patch_id)
            event_types = [str(event["event_type"]) for event in events]
            self.assertEqual(["proposed"], event_types)

    def test_auto_for_eval_accepts_valid_marker_without_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo_path = temp_root / "repo"
            repo_path.mkdir()
            repository_tools, patch_id = self._create_saved_patch(repo_path)
            write_eval_temp_marker(
                repo_path=repo_path,
                run_id="run-1",
                case_id="case-1",
                temp_root=temp_root,
            )
            approval_gate = CliApplyPatchApproval(
                repository_tools,
                approval_mode="auto_for_eval",
                eval_run_id="run-1",
            )
            tool_call = {"tool": "apply_patch", "args": {"patch_id": patch_id}}

            with patch("builtins.input") as mocked_input:
                tool_output = approval_gate.run_tool(tool_call)

            mocked_input.assert_not_called()
            self.assertIsInstance(tool_output, dict)
            apply_output = cast(dict[str, object], tool_output)
            self.assertTrue(apply_output["ok"])
            self.assertEqual("new line\n", (repo_path / "sample.txt").read_text())
            events = self._read_run_events(repository_tools, patch_id)
            self.assertTrue(
                any(
                    event["event_type"] == "apply_confirmation"
                    and event["status"] == "approved"
                    and event["details"]
                    == {"approved": True, "approval_mode": "auto_for_eval"}
                    for event in events
                )
            )
            self.assertTrue(
                any(event["event_type"] == "apply_success" for event in events)
            )

    def _create_saved_patch(self, repo_path: Path) -> tuple[RepositoryTools, str]:
        (repo_path / "sample.txt").write_text("old line\n", encoding="utf-8")
        repository_tools = RepositoryTools(repo_path)
        diff_text = "\n".join(
            [
                "diff --git a/sample.txt b/sample.txt",
                "--- a/sample.txt",
                "+++ b/sample.txt",
                "@@ -1,1 +1,1 @@",
                "-old line",
                "+new line",
            ]
        )
        propose_result = repository_tools.propose_patch(
            instruction="update sample text",
            diff=diff_text,
        )
        self.assertTrue(propose_result["ok"])
        patch_id_value = propose_result["patch_id"]
        self.assertIsInstance(patch_id_value, str)
        patch_id = cast(str, patch_id_value)
        return repository_tools, patch_id

    def _read_all_run_events(
        self,
        repository_tools: RepositoryTools,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for run_path in repository_tools.repopilot_runs_dir.glob("*.jsonl"):
            events.extend(
                json.loads(line)
                for line in run_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return events

    def _read_run_events(
        self,
        repository_tools: RepositoryTools,
        patch_id: str,
    ) -> list[dict[str, object]]:
        run_path = repository_tools.repopilot_runs_dir / f"{patch_id}.jsonl"
        return [
            json.loads(line)
            for line in run_path.read_text(encoding="utf-8").splitlines()
        ]


class TestReadmeV03Docs(unittest.TestCase):
    """验证 README 和 eval 用例描述 v0.3 安全修改流程。"""

    def setUp(self) -> None:
        project_root = Path(__file__).resolve().parent
        self.readme_text = (project_root / "README.md").read_text(encoding="utf-8")
        self.eval_text = (project_root / "eval_case.json").read_text(encoding="utf-8")

    def test_documents_safe_modification_lifecycle_and_paths(self) -> None:
        for expected_text in [
            "read_file",
            "read_file_range",
            "propose_patch(instruction, diff)",
            ".repopilot/patches",
            "CLI 展示补丁路径",
            "apply_patch(patch_id)",
            ".repopilot/backups",
            ".repopilot/runs",
            "logs/run_YYYYMMDD_HHMMSS.json",
            "run_tests(command_name)",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_approval_tokens_and_rejection_behavior(self) -> None:
        for approval_token in ["`yes`", "`y`", "`approve`"]:
            with self.subTest(approval_token=approval_token):
                self.assertIn(approval_token, self.readme_text)

        self.assertIn("空输入、`no` 或任意其他文本都会拒绝应用", self.readme_text)
        self.assertIn("目标文件不会被修改", self.readme_text)
        self.assertIn("只有用户明确批准后才会修改文件", self.readme_text)

    def test_documents_tools_and_exact_test_commands(self) -> None:
        for expected_text in [
            "`read_file_range(path, start_line, end_line)`",
            "`propose_patch(instruction, diff)`",
            "`apply_patch(patch_id)`",
            "`run_tests(command_name)`",
            "`unit` 对应 `python -m unittest discover`",
            "`compile` 对应 `python -m compileall .`",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_removes_old_patch_contract_and_keeps_safety_constraints(self) -> None:
        self.assertNotIn("propose_patch(file_path", self.readme_text)
        self.assertNotIn('"replacements"', self.readme_text)
        self.assertNotIn("不写文件", self.readme_text)
        self.assertIn("不执行任意 shell 命令", self.readme_text)
        self.assertIn("不使用 LangChain、LangGraph、LlamaIndex、MCP 或多 agent 框架", self.readme_text)
        self.assertIn("不会在没有人工批准的情况下应用修改", self.readme_text)

    def test_eval_case_does_not_depend_on_legacy_patch_contract(self) -> None:
        self.assertNotIn("propose_patch(file_path", self.eval_text)
        self.assertNotIn("replacements", self.eval_text)
        self.assertNotIn("五", self.eval_text)


if __name__ == "__main__":
    unittest.main()
