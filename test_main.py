"""命令行参数解析单元测试。"""

from __future__ import annotations

import unittest
import io
import json
import subprocess
import tempfile
from contextlib import redirect_stdout
from collections.abc import Callable, Sequence
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

V05_TOOL_DESCRIPTIONS = "\n".join(
    [
        *V03_TOOL_DESCRIPTIONS.splitlines()[:4],
        "- build_repo_index: 构建或读取项目文件索引",
        "- inspect_repo: 生成紧凑项目概览",
        *V03_TOOL_DESCRIPTIONS.splitlines()[4:],
    ]
)


class FakeLlmClient:
    """按顺序返回内容并记录每次收到的 messages。"""

    def __init__(
        self,
        outputs: Sequence[str | Callable[[list[dict[str, str]]], str]],
        prepend_plan: bool = True,
    ) -> None:
        self.prepend_plan = prepend_plan
        self.outputs: list[str | Callable[[list[dict[str, str]]], str]] = (
            [*_default_plan_outputs()] if prepend_plan else []
        )
        self.outputs.extend(outputs)
        self.messages_by_call: list[list[dict[str, str]]] = []
        self._call_count = 0

    @property
    def call_count(self) -> int:
        plan_overhead = len(_default_plan_outputs()) if self.prepend_plan else 0
        return max(0, self._call_count - plan_overhead)

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages_by_call.append([dict(message) for message in messages])
        if self._call_count >= len(self.outputs):
            raise AssertionError("fake LLM 没有更多输出")
        output = self.outputs[self._call_count]
        self._call_count += 1
        if callable(output):
            return output(messages)
        return output


class FakeRunLogger:
    """记录 agent 写入 logger 的原始步骤。"""

    def __init__(self) -> None:
        self.steps: list[dict[str, object]] = []
        self.final_answer: str | None = None
        self.error: str | None = None
        self.context_stats: dict[str, object] | None = None

    def record_step(
        self,
        step_number: int,
        model_output: str,
        tool_call: dict[str, object] | None,
        tool_result: dict[str, object],
    ) -> None:
        self.steps.append(
            {
                "step_number": step_number,
                "model_output": model_output,
                "tool_call": tool_call,
                "tool_result": tool_result,
            }
        )

    def set_final_answer(self, final_answer: str) -> None:
        self.final_answer = final_answer

    def set_error(self, error_message: str) -> None:
        self.error = error_message

    def set_context_stats(self, stats: dict[str, object]) -> None:
        self.context_stats = dict(stats)


def _default_plan_outputs() -> list[str]:
    return [_task_plan_json()]


def _task_plan_json(
    step_id: str = "step-1",
    task_type: str = "analysis",
    requires_patch: bool = False,
    requires_tests: bool = False,
    expected_changed_files: Sequence[str] = (),
) -> str:
    verification: list[dict[str, object]] = []
    if task_type in {"edit", "refactor"}:
        verification = [{"must_contain": [{"path": "sample.py", "strings": ["ok"]}]}]
    return json.dumps(
        {
            "task_type": task_type,
            "risk_level": "low",
            "max_steps": 8,
            "requires_patch": requires_patch,
            "requires_tests": requires_tests,
            "expected_changed_files": list(expected_changed_files),
            "steps": [{"id": step_id, "title": "执行任务", "description": "测试计划"}],
            "verification": verification,
        },
        ensure_ascii=False,
    )


def _agent_tool_call(
    tool: str,
    args: dict[str, object],
    plan_step_id: str | None = "step-1",
) -> str:
    tool_call: dict[str, object] = {
        "thought": f"调用 {tool}",
        "tool": tool,
        "args": args,
    }
    if plan_step_id is not None:
        tool_call["plan_step_id"] = plan_step_id
    return json.dumps(tool_call, ensure_ascii=False)


def _assert_v06_summary(test_case: unittest.TestCase, final_answer: str, prefix: str) -> None:
    test_case.assertTrue(final_answer.startswith(prefix), final_answer)
    for expected_text in [
        "v0.6 summary:",
        "execution steps:",
        "changed files:",
        "tests:",
        "verification:",
        "repair:",
    ]:
        test_case.assertIn(expected_text, final_answer)


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

    def test_v06_protocol_separates_plan_execute_verify_and_final_answer(self) -> None:
        for expected_text in [
            "PLAN",
            "EXECUTE",
            "VERIFY",
            "TaskPlan",
            "严格 TaskPlan JSON",
            "plan_step_id",
            "deterministic program verification",
            "确定性程序验证",
            "repair",
            "plan steps",
            "changed files",
            "tests",
            "verification",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.prompt)


class TestReadmeDocumentation(unittest.TestCase):
    """验证 README 保留关键版本说明。"""

    def setUp(self) -> None:
        readme_path = Path(__file__).resolve().with_name("README.md")
        self.readme_text = readme_path.read_text(encoding="utf-8")

    def test_v06_plan_execute_verify_section_documents_protocol(self) -> None:
        for expected_text in [
            "Plan → Execute → Verify",
            "不是多 agent",
            "PLAN",
            "TaskPlan",
            "EXECUTE",
            "{thought, plan_step_id, tool, args}",
            "VERIFY",
            "plan steps",
            "changed files",
            "tests",
            "verification",
            "repair 最多 1 次",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_v06_readme_keeps_eval_commands_and_v05_section(self) -> None:
        for expected_text in [
            "python -m unittest discover",
            "python run_edit_eval.py --cases eval_cases/edit_cases.json",
            "python run_edit_eval.py --cases eval_cases/context_cases.json",
            "普通 CLI 的 `apply_patch` 仍然要求用户手动确认",
            "`auto_for_eval` 只允许评测 runner 在带 marker 的临时 eval repo 中使用",
            "v0.5 上下文管理与项目索引",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_model_output_examples_include_plan_step_id(self) -> None:
        self.assertIn(
            "`plan_step_id` 必须来自 `TaskPlan.steps`",
            self.readme_text,
        )
        self.assertEqual(6, self.readme_text.count('"plan_step_id": "step-1"'))


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


class TestAgentContextV05(unittest.TestCase):
    """验证 agent loop 的上下文统计与 observation 压缩。"""

    def test_compacts_older_observations_only_in_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            for file_name, file_text in {
                "one.txt": "alpha raw observation\n",
                "two.txt": "bravo raw observation\n",
                "three.txt": "charlie raw observation\n",
                "four.txt": "delta raw observation\n",
            }.items():
                (repo_path / file_name).write_text(file_text, encoding="utf-8")
            fake_llm = FakeLlmClient(
                [
                    _agent_tool_call("read_file", {"path": "one.txt"}),
                    _agent_tool_call("read_file", {"path": "two.txt"}),
                    _agent_tool_call("read_file", {"path": "three.txt"}),
                    _agent_tool_call("read_file", {"path": "four.txt"}),
                    _agent_tool_call("finish", {"answer": "完成"}),
                ]
            )
            fake_logger = FakeRunLogger()
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=5,
            )

            final_answer = agent.answer("读取四个文件后完成")

            _assert_v06_summary(self, final_answer, "完成")
            self.assertEqual(5, fake_llm.call_count)
            last_messages = fake_llm.messages_by_call[-1]
            compact_messages = [
                message["content"]
                for message in last_messages
                if "[compact observation]" in message["content"]
            ]
            self.assertEqual(1, len(compact_messages), last_messages)
            self.assertIn("tool=read_file", compact_messages[0])
            self.assertIn("path=one.txt", compact_messages[0])
            last_payload = "\n".join(message["content"] for message in last_messages)
            self.assertNotIn("alpha raw observation", last_payload)
            self.assertIn("bravo raw observation", last_payload)
            self.assertIn("charlie raw observation", last_payload)
            self.assertIn("delta raw observation", last_payload)

            first_step_result = cast(dict[str, object], fake_logger.steps[0]["tool_result"])
            self.assertTrue(first_step_result["ok"])
            self.assertIn("alpha raw observation", str(first_step_result["output"]))
            self.assertNotIn("[compact observation]", str(first_step_result["output"]))

    def test_context_stats_count_reads_ranges_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            source_dir = repo_path / "pkg"
            source_dir.mkdir()
            (source_dir / "sample.py").write_text(
                "def target():\n    return 'needle'\n",
                encoding="utf-8",
            )
            fake_llm = FakeLlmClient(
                [
                    "不是 JSON",
                    _agent_tool_call("read_file", {"path": "pkg\\sample.py"}),
                    _agent_tool_call(
                        "read_file_range",
                        {"path": "pkg/sample.py", "start_line": 1, "end_line": 2},
                    ),
                    _agent_tool_call(
                        "search_text",
                        {"keyword": "absent", "path_glob": "pkg/*.py"},
                    ),
                    _agent_tool_call("finish", {"answer": "完成"}),
                ]
            )
            fake_logger = FakeRunLogger()
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=5,
            )

            final_answer = agent.answer("统计上下文")

            _assert_v06_summary(self, final_answer, "完成")
            context_stats = agent.context_stats.to_dict()
            self.assertEqual(5, context_stats["steps_used"])
            self.assertEqual(["pkg/sample.py"], context_stats["files_read"])
            self.assertEqual(["pkg/sample.py"], context_stats["full_file_reads"])
            self.assertEqual(
                [{"path": "pkg/sample.py", "start_line": 1, "end_line": 2}],
                context_stats["ranges_read"],
            )
            self.assertEqual(1, context_stats["search_calls"])
            expected_output_chars = sum(
                len(
                    json.dumps(
                        cast(dict[str, object], step["tool_result"]),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                for step in fake_logger.steps
            )
            self.assertEqual(
                expected_output_chars,
                context_stats["total_tool_output_chars"],
            )
            self.assertGreater(cast(int, context_stats["messages_total_chars"]), 0)

    def test_missing_plan_step_id_rejects_before_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            fake_llm = FakeLlmClient(
                [_agent_tool_call("list_dir", {"path": "."}, plan_step_id=None)]
            )
            fake_logger = FakeRunLogger()
            tool_calls: list[dict[str, object]] = []
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=1,
                tool_runner=lambda tool_call: tool_calls.append(tool_call),
            )

            final_answer = agent.answer("缺少 plan_step_id")

            self.assertIn("达到最大循环步数 1", final_answer)
            self.assertEqual([], tool_calls)
            self.assertEqual(1, len(fake_logger.steps))
            tool_result = cast(dict[str, object], fake_logger.steps[0]["tool_result"])
            self.assertFalse(tool_result["ok"])
            self.assertIn("plan_step_id", str(tool_result["error"]))

    def test_planner_error_returns_readable_failure_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            invalid_plan = json.dumps(
                {
                    "task_type": "edit",
                    "risk_level": "low",
                    "max_steps": 4,
                    "requires_patch": True,
                    "requires_tests": False,
                    "expected_changed_files": ["README.md"],
                    "steps": [{"id": "step1", "title": "编辑 README"}],
                    "verification": [{"must_contain": "README"}],
                },
                ensure_ascii=False,
            )
            fake_llm = FakeLlmClient([invalid_plan], prepend_plan=False)
            fake_logger = FakeRunLogger()
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=4,
            )

            final_answer = agent.answer("生成非法 plan")

            self.assertIn("PLAN 阶段失败", final_answer)
            self.assertEqual(final_answer, fake_logger.error)
            self.assertEqual(final_answer, fake_logger.final_answer)
            self.assertEqual([], fake_logger.steps)

    def test_planner_error_records_plan_stage_in_real_logger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            log_dir = repo_path / "logs"
            invalid_plan = json.dumps(
                {
                    "task_type": "edit",
                    "risk_level": "low",
                    "max_steps": 4,
                    "requires_patch": True,
                    "requires_tests": False,
                    "expected_changed_files": ["README.md"],
                    "steps": [{"id": "step1", "title": "编辑 README"}],
                    "verification": [{"must_contain": "README"}],
                },
                ensure_ascii=False,
            )
            run_logger = RunLogger(repo_path, "生成非法 plan", log_dir=log_dir)
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, FakeLlmClient([invalid_plan], prepend_plan=False)),
                repository_tools=RepositoryTools(repo_path),
                run_logger=run_logger,
                max_steps=4,
            )

            final_answer = agent.answer("生成非法 plan")

            self.assertIn("PLAN 阶段失败", final_answer)
            self.assertEqual("PLAN", run_logger.payload["stage"])
            self.assertEqual(["INIT", "PLAN"], run_logger.payload["stage_history"])
            self.assertEqual(final_answer, run_logger.payload["error"])
            self.assertEqual(final_answer, run_logger.payload["final_answer"])

    def test_execute_prompt_lists_actual_plan_step_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            custom_plan = _task_plan_json(step_id="step1")
            fake_llm = FakeLlmClient(
                [
                    custom_plan,
                    _agent_tool_call("finish", {"answer": "完成"}, plan_step_id="step1"),
                ],
                prepend_plan=False,
            )
            fake_logger = FakeRunLogger()
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=4,
            )

            final_answer = agent.answer("说明流程")

            _assert_v06_summary(self, final_answer, "完成")
            execute_messages = fake_llm.messages_by_call[1]
            execute_prompt = "\n".join(message["content"] for message in execute_messages)
            self.assertIn("有效 plan_step_id", execute_prompt)
            self.assertIn("step1", execute_prompt)
            self.assertNotIn("step-1。", execute_prompt)

    def test_unknown_plan_step_id_rejects_before_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            fake_llm = FakeLlmClient(
                [_agent_tool_call("list_dir", {"path": "."}, plan_step_id="missing-step")]
            )
            fake_logger = FakeRunLogger()
            tool_calls: list[dict[str, object]] = []
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=1,
                tool_runner=lambda tool_call: tool_calls.append(tool_call),
            )

            final_answer = agent.answer("未知 plan_step_id")

            self.assertIn("达到最大循环步数 1", final_answer)
            self.assertEqual([], tool_calls)
            tool_result = cast(dict[str, object], fake_logger.steps[0]["tool_result"])
            self.assertFalse(tool_result["ok"])
            error_message = str(tool_result["error"])
            self.assertIn("未知 plan_step_id", error_message)
            self.assertIn("有效 plan_step_id", error_message)
            self.assertIn("step-1", error_message)

    def test_apply_patch_before_successful_propose_rejects_before_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            fake_llm = FakeLlmClient(
                [_agent_tool_call("apply_patch", {"patch_id": "20260629_010203_abcdef123456"})]
            )
            fake_logger = FakeRunLogger()
            tool_calls: list[dict[str, object]] = []
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=1,
                tool_runner=lambda tool_call: tool_calls.append(tool_call),
            )

            final_answer = agent.answer("禁止直接 apply")

            self.assertIn("达到最大循环步数 1", final_answer)
            self.assertEqual([], tool_calls)
            tool_result = cast(dict[str, object], fake_logger.steps[0]["tool_result"])
            self.assertFalse(tool_result["ok"])
            self.assertIn("propose_patch", str(tool_result["error"]))

    def test_logger_writes_context_stats_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "sample.py").write_text("print('ok')\n", encoding="utf-8")
            finish_logger = FakeRunLogger()
            finish_agent = CodeAnalysisAgent(
                llm_client=cast(
                    LlmClient,
                    FakeLlmClient(
                        [
                            _agent_tool_call("read_file", {"path": "sample.py"}),
                            _agent_tool_call("finish", {"answer": "完成"}),
                        ]
                    ),
                ),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, finish_logger),
                max_steps=2,
            )

            final_answer = finish_agent.answer("读取后完成")

            _assert_v06_summary(self, final_answer, "完成")
            self.assertEqual(finish_agent.context_stats.to_dict(), finish_logger.context_stats)
            self.assertEqual(2, cast(dict[str, object], finish_logger.context_stats)["steps_used"])

            max_steps_logger = FakeRunLogger()
            max_steps_agent = CodeAnalysisAgent(
                llm_client=cast(
                    LlmClient,
                    FakeLlmClient([_agent_tool_call("read_file", {"path": "sample.py"})]),
                ),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, max_steps_logger),
                max_steps=1,
            )

            max_steps_answer = max_steps_agent.answer("不调用 finish")

            self.assertIn("达到最大循环步数 1", max_steps_answer)
            self.assertEqual(max_steps_agent.context_stats.to_dict(), max_steps_logger.context_stats)
            self.assertEqual(1, cast(dict[str, object], max_steps_logger.context_stats)["steps_used"])

    def test_payload_records_verify_after_successful_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "sample.py").write_text("print('ok')\n", encoding="utf-8")
            run_logger = RunLogger(repo_path=repo_path, user_task="verify payload", log_dir=repo_path / "logs")
            agent = CodeAnalysisAgent(
                llm_client=cast(
                    LlmClient,
                    FakeLlmClient(
                        [
                            _agent_tool_call("read_file", {"path": "sample.py"}),
                            _agent_tool_call("finish", {"answer": "完成"}),
                        ]
                    ),
                ),
                repository_tools=RepositoryTools(repo_path),
                run_logger=run_logger,
                max_steps=2,
            )

            final_answer = agent.answer("直接完成")

            _assert_v06_summary(self, final_answer, "完成")
            self.assertEqual(["INIT", "PLAN", "EXECUTE", "VERIFY", "FINISH"], run_logger.payload["stage_history"])
            self.assertEqual("FINISH", run_logger.payload["stage"])
            self.assertIsInstance(run_logger.payload["plan"], dict)
            self.assertEqual("step-1", run_logger.payload["plan_step_id"])
            steps = cast(list[dict[str, object]], run_logger.payload["steps"])
            self.assertEqual("step-1", steps[0]["plan_step_id"])
            verify_status = cast(dict[str, object], run_logger.payload["verify_status"])
            self.assertTrue(verify_status["passed"])
            self.assertEqual(0, run_logger.payload["repair_attempt"])
            self.assertEqual(0, run_logger.payload["repair_attempts"])

    def test_tests_failed_verification_triggers_one_repair_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "sample.py").write_text("print('ok')\n", encoding="utf-8")
            fake_llm = FakeLlmClient(
                [
                    _task_plan_json(task_type="edit", requires_tests=True),
                    _agent_tool_call("run_tests", {"command_name": "unit"}),
                    _agent_tool_call("finish", {"answer": "首次完成"}),
                    _agent_tool_call("run_tests", {"command_name": "unit"}),
                    _agent_tool_call("finish", {"answer": "修复后完成"}),
                ],
                prepend_plan=False,
            )
            run_logger = RunLogger(repo_path=repo_path, user_task="repair", log_dir=repo_path / "logs")
            run_tests_calls = 0

            def fake_tool_runner(tool_call: dict[str, object]) -> object:
                nonlocal run_tests_calls
                tool_name = tool_call.get("tool")
                if tool_name == "run_tests":
                    run_tests_calls += 1
                    return {
                        "command_name": "unit",
                        "exit_code": 1 if run_tests_calls == 1 else 0,
                        "stdout": "",
                        "stderr": "failed" if run_tests_calls == 1 else "",
                        "timed_out": False,
                    }
                if tool_name == "finish":
                    tool_args = cast(dict[str, object], tool_call["args"])
                    return tool_args["answer"]
                raise ToolError(f"unexpected tool: {tool_name}")

            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=run_logger,
                max_steps=4,
                tool_runner=fake_tool_runner,
            )

            final_answer = agent.answer("测试失败后修复")

            _assert_v06_summary(self, final_answer, "修复后完成")
            self.assertEqual(2, run_tests_calls)
            self.assertEqual(1, run_logger.payload["repair_attempts"])
            self.assertEqual(1, run_logger.payload["repair_attempt"])
            self.assertEqual(
                ["INIT", "PLAN", "EXECUTE", "VERIFY", "REPAIR", "EXECUTE", "VERIFY", "FINISH"],
                run_logger.payload["stage_history"],
            )
            verify_status = cast(dict[str, object], run_logger.payload["verify_status"])
            self.assertTrue(verify_status["passed"])

    def test_analysis_plan_tests_failed_does_not_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            fake_llm = FakeLlmClient(
                [
                    _task_plan_json(task_type="analysis", requires_tests=True),
                    _agent_tool_call("run_tests", {"command_name": "unit"}),
                    _agent_tool_call("finish", {"answer": "首次完成"}),
                    _agent_tool_call("run_tests", {"command_name": "unit"}),
                    _agent_tool_call("finish", {"answer": "不应修复"}),
                ],
                prepend_plan=False,
            )
            run_logger = RunLogger(repo_path=repo_path, user_task="analysis no repair", log_dir=repo_path / "logs")
            run_tests_calls = 0

            def fake_tool_runner(tool_call: dict[str, object]) -> object:
                nonlocal run_tests_calls
                tool_name = tool_call.get("tool")
                if tool_name == "run_tests":
                    run_tests_calls += 1
                    return {
                        "command_name": "unit",
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "failed",
                        "timed_out": False,
                    }
                if tool_name == "finish":
                    tool_args = cast(dict[str, object], tool_call["args"])
                    return tool_args["answer"]
                raise ToolError(f"unexpected tool: {tool_name}")

            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=run_logger,
                max_steps=4,
                tool_runner=fake_tool_runner,
            )

            final_answer = agent.answer("analysis 测试失败不修复")

            _assert_v06_summary(self, final_answer, "首次完成")
            self.assertEqual(1, run_tests_calls)
            self.assertEqual(0, run_logger.payload["repair_attempts"])
            self.assertEqual(["INIT", "PLAN", "EXECUTE", "VERIFY", "FINISH"], run_logger.payload["stage_history"])
            self.assertIsNotNone(run_logger.payload["error"])
            self.assertIn("tests_failed", str(run_logger.payload["verify_status"]))

    def test_logger_keeps_full_tool_results_after_compaction(self) -> None:
        large_observation = "raw-line\n" * 200
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            for file_name in ["one.txt", "two.txt", "three.txt", "four.txt"]:
                (repo_path / file_name).write_text(large_observation, encoding="utf-8")
            fake_llm = FakeLlmClient(
                [
                    _agent_tool_call("read_file", {"path": "one.txt"}),
                    _agent_tool_call("read_file", {"path": "two.txt"}),
                    _agent_tool_call("read_file", {"path": "three.txt"}),
                    _agent_tool_call("read_file", {"path": "four.txt"}),
                    _agent_tool_call("finish", {"answer": "完成"}),
                ]
            )
            fake_logger = FakeRunLogger()
            agent = CodeAnalysisAgent(
                llm_client=cast(LlmClient, fake_llm),
                repository_tools=RepositoryTools(repo_path),
                run_logger=cast(RunLogger, fake_logger),
                max_steps=5,
            )

            final_answer = agent.answer("读取四个文件后完成")

            _assert_v06_summary(self, final_answer, "完成")
            compacted_llm_payload = "\n".join(
                message["content"] for message in fake_llm.messages_by_call[-1]
            )
            self.assertIn("[compact observation]", compacted_llm_payload)
            self.assertNotIn(large_observation.strip(), compacted_llm_payload)
            first_step_result = cast(dict[str, object], fake_logger.steps[0]["tool_result"])
            self.assertTrue(first_step_result["ok"])
            self.assertIn("1 | raw-line", str(first_step_result["output"]))
            self.assertIn("200 | raw-line", str(first_step_result["output"]))
            self.assertNotIn("[compact observation]", str(first_step_result["output"]))


class TestPromptContextV05(unittest.TestCase):
    """验证 v0.5 上下文管理 prompt 规则。"""

    def setUp(self) -> None:
        self.prompt = build_system_prompt(V05_TOOL_DESCRIPTIONS)

    def test_tool_whitelist_includes_repo_index_and_inspect_repo(self) -> None:
        self.assertIn("build_repo_index", self.prompt)
        self.assertIn("inspect_repo", self.prompt)
        self.assertIn(V05_TOOL_DESCRIPTIONS, self.prompt)
        self.assertIn("tool 只能是", self.prompt)

    def test_prefers_inspect_repo_for_project_analysis_and_modification(self) -> None:
        self.assertIn("项目分析或修改任务应优先调用 inspect_repo", self.prompt)
        self.assertIn("项目分析、定位入口或准备修改代码时，优先使用 inspect_repo", self.prompt)
        self.assertIn("只有需要刷新索引时才调用 build_repo_index", self.prompt)

    def test_requires_ranges_for_large_files_and_evidence_in_final_answer(self) -> None:
        self.assertIn("优先使用 read_file_range", self.prompt)
        self.assertIn("禁止完整读取与任务无关的大文件", self.prompt)
        self.assertIn("文件名、函数名、类名、行号和已读取证据", self.prompt)


class TestRunLoggerV06Protocol(unittest.TestCase):
    """验证 logger payload 的 v0.6 默认字段与旧字段兼容。"""

    def test_payload_keeps_legacy_fields_and_adds_v06_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            run_logger = RunLogger(repo_path=repo_path, user_task="检查日志", log_dir=repo_path / "logs")

            for legacy_field in [
                "started_at",
                "repo_path",
                "user_task",
                "steps",
                "final_answer",
                "error",
                "context_stats",
            ]:
                with self.subTest(legacy_field=legacy_field):
                    self.assertIn(legacy_field, run_logger.payload)

            self.assertEqual("INIT", run_logger.payload["stage"])
            self.assertEqual(["INIT"], run_logger.payload["stage_history"])
            self.assertIsNone(run_logger.payload["plan"])
            self.assertIsNone(run_logger.payload["plan_step_id"])
            self.assertEqual(0, run_logger.payload["repair_attempt"])
            self.assertEqual(0, run_logger.payload["repair_attempts"])
            self.assertIsNone(run_logger.payload["verify_status"])

    def test_record_step_keeps_old_shape_and_copies_plan_step_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            run_logger = RunLogger(repo_path=repo_path, user_task="检查步骤", log_dir=repo_path / "logs")
            tool_call = {
                "thought": "读取文件",
                "plan_step_id": "step-1",
                "tool": "read_file",
                "args": {"path": "sample.py"},
            }

            run_logger.record_step(
                step_number=1,
                model_output=json.dumps(tool_call, ensure_ascii=False),
                tool_call=tool_call,
                tool_result={"ok": True, "output": "ok"},
            )

            steps = cast(list[dict[str, object]], run_logger.payload["steps"])
            self.assertEqual(1, len(steps))
            step_payload = steps[0]
            for old_step_field in ["step", "model_output", "tool_call", "tool_result"]:
                with self.subTest(old_step_field=old_step_field):
                    self.assertIn(old_step_field, step_payload)
            self.assertEqual("step-1", step_payload["plan_step_id"])
            self.assertEqual("step-1", run_logger.payload["plan_step_id"])


if __name__ == "__main__":
    unittest.main()
