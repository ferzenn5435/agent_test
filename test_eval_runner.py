"""edit eval loader 的单元测试。"""

from __future__ import annotations

import difflib
import json
import re
import sys
import unittest
import tempfile
import importlib
import io
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout

from pathlib import Path
from typing import cast
from unittest.mock import patch

from agent import CodeAnalysisAgent
from config import MAX_STEPS
from llm_client import LlmClient
from logger import RunLogger
from main import main as cli_main
from model_provider import LLMResponse, TokenUsage
from tools import RepositoryTools

eval_runner = importlib.import_module("eval_runner")
eval_safety = importlib.import_module("eval_safety")
run_edit_eval_cli = importlib.import_module("run_edit_eval")
EditEvalCase = eval_runner.EditEvalCase
EditEvalConfigError = eval_runner.EditEvalConfigError
EvalSafetyError = eval_safety.EvalSafetyError
MustContainRule = eval_runner.MustContainRule
check_allowed_changed_files = eval_runner.check_allowed_changed_files
check_must_contain = eval_runner.check_must_contain
classify_failure_type = eval_runner.classify_failure_type
compare_snapshots = eval_runner.compare_snapshots
copy_fixture_to_temp = eval_runner.copy_fixture_to_temp
load_edit_cases = eval_runner.load_edit_cases
run_edit_case = eval_runner.run_edit_case
run_edit_eval = eval_runner.run_edit_eval
snapshot_repo_files = eval_runner.snapshot_repo_files
validate_eval_temp_repo = eval_safety.validate_eval_temp_repo
write_eval_temp_marker = eval_safety.write_eval_temp_marker


class FakeLlmClient:
    """按顺序返回工具调用的最小 fake LLM。"""

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
        self._call_count = 0

    @property
    def call_count(self) -> int:
        plan_overhead = len(_default_plan_outputs()) if self.prepend_plan else 0
        return max(0, self._call_count - plan_overhead)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self._call_count >= len(self.outputs):
            raise AssertionError("fake LLM 没有更多输出")
        output = self.outputs[self._call_count]
        self._call_count += 1
        if callable(output):
            return output(messages)
        return output


class StructuredFakeLlmClient(FakeLlmClient):
    """返回 LLMResponse 的 fake LLM，用于验证 usage schema。"""

    def chat_response(self, messages: list[dict[str, str]]) -> LLMResponse:
        content = self.chat(messages)
        return LLMResponse(
            content=content,
            provider="mock",
            model="mock-model",
            profile_name="mock-profile",
            latency_ms=12.5,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                estimated=False,
                estimated_cost=0.25,
            ),
            raw={},
        )


def _default_plan_outputs() -> list[str]:
    """返回默认 plan 阶段输出列表（单个 TaskPlan JSON）。"""
    return [_task_plan_json()]


def _task_plan_json(
    step_id: str = "step-1",
    task_type: str = "analysis",
    requires_patch: bool = False,
    requires_tests: bool = False,
    expected_changed_files: Sequence[str] = (),
) -> str:
    """构造 TaskPlan JSON 字符串，用于 fake LLM 的 plan 阶段返回。"""
    verification: list[dict[str, object]] = []
    if task_type in {"edit", "refactor"}:
        verification = [{"must_contain": [{"path": "README.md", "strings": ["# 简单 Python 项目"]}]}]
    return json.dumps(
        {
            "task_type": task_type,
            "risk_level": "low",
            "max_steps": 16,
            "requires_patch": requires_patch,
            "requires_tests": requires_tests,
            "expected_changed_files": list(expected_changed_files),
            "steps": [{"id": step_id, "title": "执行评测", "description": "fake plan"}],
            "verification": verification,
        },
        ensure_ascii=False,
    )


def _tool_call(tool: str, args: dict[str, object], plan_step_id: str | None = "step-1") -> str:
    """构造 agent 步骤的工具调用 JSON 字符串。"""
    tool_call: dict[str, object] = {
        "thought": f"调用 {tool}",
        "tool": tool,
        "args": args,
    }
    if plan_step_id is not None:
        tool_call["plan_step_id"] = plan_step_id
    return json.dumps(tool_call, ensure_ascii=False)


def _unified_diff(path: str, before: str, after: str) -> str:
    """生成 unified diff 格式字符串，用于构造补丁测试数据。"""
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join((f"diff --git a/{path} b/{path}", *diff_lines, ""))


def _apply_last_patch_call(messages: list[dict[str, str]]) -> str:
    """从最近一次 LLM 响应中提取 patch_id，构造 apply_patch 调用的回调函数。"""
    latest_feedback = messages[-1]["content"]
    patch_id_match = re.search(r'"patch_id"\s*:\s*"([^"]+)"', latest_feedback)
    if patch_id_match is None:
        raise AssertionError(f"未找到 patch_id: {latest_feedback}")
    return _tool_call("apply_patch", {"patch_id": patch_id_match.group(1)})


def _run_compile_tests_call() -> str:
    """返回执行 compile 测试的工具调用 JSON。"""
    return _tool_call("run_tests", {"command_name": "compile"})


def _readme_patch_call() -> str:
    """返回更新 README.md 的 propose_patch 工具调用 JSON。"""
    before = (
        "# 简单 Python 项目\n"
        "\n"
        "这是一个用于编辑能力评估的最小 Python 示例项目，当前提供一个加法函数和对应的单元测试。\n"
    )
    after = f"{before}本项目用于验证 agent 的安全编辑能力。\n"
    return _tool_call(
        "propose_patch",
        {
            "instruction": "更新 README.md 说明 eval 安全编辑能力。",
            "diff": _unified_diff("README.md", before, after),
        },
    )


def _app_patch_call() -> str:
    """返回修改 app.py 返回值的 propose_patch 工具调用 JSON。"""
    before = "def add(a, b):\n    return a + b\n"
    after = "def add(a, b):\n    return 42\n"
    return _tool_call(
        "propose_patch",
        {
            "instruction": "修改 app.py 返回值。",
            "diff": _unified_diff("app.py", before, after),
        },
    )


def _new_file_patch_call(path: str, content: str) -> str:
    """返回新增文件的 propose_patch 工具调用 JSON。"""
    diff_lines = [
        f"diff --git a/{path} b/{path}",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(content.splitlines())} @@",
        *(f"+{line}" for line in content.splitlines()),
    ]
    return _tool_call(
        "propose_patch",
        {
            "instruction": f"新增 {path}。",
            "diff": "\n".join(diff_lines) + "\n",
        },
    )


class TestFakeLlmClient(unittest.TestCase):
    """验证 deterministic fake LLM 可驱动真实 agent 循环。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent

    def _case(self, case_id: str, max_steps: int = 5) -> EditEvalCase:
        return EditEvalCase(
            id=case_id,
            fixture="tests/fixtures/simple_python_project",
            prompt=f"执行 {case_id}",
            max_steps=max_steps,
            allowed_changed_files=("README.md",),
            must_contain=(),
        )

    def test_drives_agent_to_finish(self) -> None:
        fake_llm = FakeLlmClient(
            [
                _tool_call("read_file", {"path": "README.md"}),
                _readme_patch_call(),
                _apply_last_patch_call,
                _run_compile_tests_call(),
                _tool_call("finish", {"answer": "README.md 已更新。"}),
            ]
        )

        eval_result = run_edit_case(
            self._case("fake-finish"),
            self.project_root,
            lambda _case: fake_llm,
        )

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual(5, fake_llm.call_count)
        self.assertEqual(("README.md",), eval_result.changed_files)
        self.assertIsNotNone(eval_result.final_answer)
        assert eval_result.final_answer is not None
        self.assertIsNotNone(eval_result.final_answer)
        assert eval_result.final_answer is not None
        self.assertTrue(eval_result.final_answer.startswith("README.md 已更新。"), eval_result.final_answer)
        self.assertIn("v0.6 summary:", eval_result.final_answer)
        self.assertIsNone(eval_result.error)

    def test_missing_finish_records_failure(self) -> None:
        fake_llm = FakeLlmClient([_tool_call("list_dir", {"path": "."})])

        eval_result = run_edit_case(
            self._case("fake-missing-finish", max_steps=1),
            self.project_root,
            lambda _case: fake_llm,
        )

        self.assertFalse(eval_result.passed)
        self.assertEqual(1, eval_result.steps)
        self.assertEqual(1, fake_llm.call_count)
        self.assertIn("agent 未在 max_steps 内成功调用 finish", eval_result.reasons)
        self.assertIsNotNone(eval_result.error)

    def test_invalid_json_records_failure(self) -> None:
        fake_llm = FakeLlmClient(["这不是 JSON"])

        eval_result = run_edit_case(
            self._case("fake-invalid-json", max_steps=1),
            self.project_root,
            lambda _case: fake_llm,
        )

        self.assertFalse(eval_result.passed)
        self.assertEqual(1, eval_result.steps)
        self.assertEqual(1, fake_llm.call_count)
        self.assertEqual((), eval_result.changed_files)
        self.assertIn("agent 未在 max_steps 内成功调用 finish", eval_result.reasons)
        self.assertIsNotNone(eval_result.error)
        self.assertEqual("invalid_json", eval_result.failure_type)

    def test_eval_prompt_documents_auto_approval_without_changing_case_prompt(self) -> None:
        eval_case = self._case("fake-auto-approval")

        eval_prompt = eval_runner._build_eval_agent_prompt(eval_case)

        self.assertIn(eval_case.prompt, eval_prompt)
        self.assertIn("auto_for_eval 自动批准", eval_prompt)
        self.assertIn("评测临时仓库", eval_prompt)
        self.assertIn("直接调用 apply_patch", eval_prompt)
        self.assertIn("不要额外调用 inspect_repo", eval_prompt)
        self.assertIn("不要重复读取同一文件", eval_prompt)
        self.assertEqual("执行 fake-auto-approval", eval_case.prompt)


class TestEditCaseLoader(unittest.TestCase):
    """验证 edit 评测用例加载与校验规则。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent

    def _write_cases(self, cases: list[dict[str, object]]) -> Path:
        temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temp_directory.cleanup)
        cases_path = Path(temp_directory.name) / "edit_cases.json"
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return cases_path

    def test_loads_valid_file_with_v06_cases(self) -> None:
        cases_path = self.project_root / "eval_cases" / "edit_cases.json"
        cases = load_edit_cases(cases_path)
        cases_by_id = {case.id: case for case in cases}

        self.assertEqual(6, len(cases))
        self.assertIsInstance(cases[0], EditEvalCase)
        self.assertEqual("update-readme", cases[0].id)
        self.assertEqual("tests/fixtures/simple_python_project", cases[0].fixture)
        self.assertEqual("README.md", cases[0].allowed_changed_files[0])
        self.assertEqual("unit", cases[1].test_command)
        self.assertFalse(cases[0].expect_no_business_changes)
        self.assertEqual(7, cases[0].max_steps)
        self.assertEqual(16, cases[1].max_steps)

        forbidden_case = cases[2]
        self.assertEqual("forbidden-outside-file", forbidden_case.id)
        self.assertEqual(tuple(), forbidden_case.must_contain)
        self.assertEqual((), forbidden_case.allowed_changed_files)
        self.assertEqual(4, forbidden_case.max_steps)
        self.assertTrue(forbidden_case.expect_no_business_changes)
        self.assertIsNotNone(forbidden_case.raw_case)
        self.assertEqual(forbidden_case.raw_case.get("prompt"), forbidden_case.prompt)

        planned_case = cases_by_id["planned-add-function-with-tests"]
        self.assertTrue(planned_case.must_have_plan)
        self.assertEqual(("PLAN", "EXECUTE", "VERIFY", "FINISH"), planned_case.required_stages)
        self.assertTrue(planned_case.must_reference_plan_steps)
        self.assertTrue(planned_case.require_verify_after_patch)
        self.assertEqual(1, planned_case.max_repair_attempts)

    def test_normalizes_backslashes_in_allowed_changed_files(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "normalize-backslashes",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "normalize paths",
                    "allowed_changed_files": ["docs\\readme.md", "src\\main.py"],
                    "must_contain": [],
                }
            ]
        )

        cases = load_edit_cases(cases_path)

        self.assertEqual(("docs/readme.md", "src/main.py"), cases[0].allowed_changed_files)

    def test_rejects_duplicate_case_ids(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "duplicate",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "first case",
                    "allowed_changed_files": [],
                    "must_contain": [],
                },
                {
                    "id": "duplicate",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "second case",
                    "allowed_changed_files": [],
                    "must_contain": [],
                },
            ]
        )

        with self.assertRaisesRegex(EditEvalConfigError, "id 重复"):
            load_edit_cases(cases_path)

    def test_rejects_invalid_test_command(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "invalid-command",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "run invalid command",
                    "allowed_changed_files": ["app.py"],
                    "must_contain": [
                        {
                            "path": "app.py",
                            "strings": ["assert"],
                        }
                    ],
                    "test_command": "rm -rf .",
                }
            ]
        )

        with self.assertRaisesRegex(
            EditEvalConfigError,
            "test_command 必须是 unit 或 compile",
        ):
            load_edit_cases(cases_path)

    def test_rejects_invalid_allowed_changed_files(self) -> None:
        for raw_value in ["app.py", ["..", "app.py"], ["/etc/passwd"]]:
            with self.subTest(raw_value=raw_value):
                cases_path = self._write_cases(
                    [
                        {
                            "id": f"bad-allowed-{abs(hash(str(raw_value))) % 10000}",
                            "fixture": "tests/fixtures/simple_python_project",
                            "prompt": "edit file",
                            "allowed_changed_files": raw_value,
                            "must_contain": [
                                {
                                    "path": "README.md",
                                    "strings": ["hello"],
                                }
                            ],
                        }
                    ]
                )

                with self.assertRaises(EditEvalConfigError):
                    load_edit_cases(cases_path)

    def test_rejects_empty_must_contain_strings(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "empty-strings",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "missing strings",
                    "allowed_changed_files": [],
                    "must_contain": [
                        {
                            "path": "app.py",
                            "strings": [],
                        }
                    ],
                }
            ]
        )

        with self.assertRaisesRegex(
            EditEvalConfigError,
            "strings 必须是非空字符串数组",
        ):
            load_edit_cases(cases_path)

    def test_omitted_max_steps_uses_default(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "default-steps",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "no max steps",
                    "allowed_changed_files": [],
                    "must_contain": [],
                }
            ]
        )

        cases = load_edit_cases(cases_path)

        self.assertEqual(1, len(cases))
        self.assertEqual(MAX_STEPS, cases[0].max_steps)

    def test_loads_valid_v06_fields_and_defaults_old_fields(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "v06-fields",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "validate v06 fields",
                    "allowed_changed_files": [],
                    "must_contain": [],
                    "must_have_plan": True,
                    "required_stages": ["PLAN", "EXECUTE", "VERIFY"],
                    "must_reference_plan_steps": True,
                    "require_verify_after_patch": True,
                    "max_repair_attempts": 0,
                },
                {
                    "id": "old-fields-only",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "old fields only",
                    "allowed_changed_files": [],
                    "must_contain": [],
                },
            ]
        )

        cases = load_edit_cases(cases_path)

        self.assertTrue(cases[0].must_have_plan)
        self.assertEqual(("PLAN", "EXECUTE", "VERIFY"), cases[0].required_stages)
        self.assertTrue(cases[0].must_reference_plan_steps)
        self.assertTrue(cases[0].require_verify_after_patch)
        self.assertEqual(0, cases[0].max_repair_attempts)
        self.assertFalse(cases[1].must_have_plan)
        self.assertEqual((), cases[1].required_stages)
        self.assertFalse(cases[1].must_reference_plan_steps)
        self.assertFalse(cases[1].require_verify_after_patch)
        self.assertIsNone(cases[1].max_repair_attempts)

    def test_rejects_invalid_v06_field_types(self) -> None:
        invalid_cases = [
            {"id": "bad-plan", "must_have_plan": "yes"},
            {"id": "bad-stages-type", "required_stages": "VERIFY"},
            {"id": "bad-stages-item", "required_stages": [""]},
            {"id": "bad-reference", "must_reference_plan_steps": 1},
            {"id": "bad-verify", "require_verify_after_patch": "true"},
            {"id": "bad-repair-bool", "max_repair_attempts": False},
            {"id": "bad-repair-negative", "max_repair_attempts": -1},
        ]
        for raw_case in invalid_cases:
            with self.subTest(case_id=raw_case["id"]):
                cases_path = self._write_cases(
                    [
                        {
                            "fixture": "tests/fixtures/simple_python_project",
                            "prompt": "bad v06 field",
                            "allowed_changed_files": [],
                            "must_contain": [],
                            **raw_case,
                        }
                    ]
                )
                with self.assertRaises(EditEvalConfigError):
                    load_edit_cases(cases_path)


class TestEvalTempSafety(unittest.TestCase):
    """验证 eval 临时代码库 marker 与文件变更校验。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name)

    def test_marker_write_validate_accepts_legal_temp_repo(self) -> None:
        repo_path = self.temp_root / "repo"
        repo_path.mkdir()

        marker_path = write_eval_temp_marker(
            repo_path=repo_path,
            run_id="run-123",
            case_id="case-abc",
            temp_root=self.temp_root,
        )
        validate_eval_temp_repo(repo_path, "run-123")

        self.assertTrue(marker_path.is_file())

    def test_marker_validation_rejects_missing_marker_and_wrong_run_id(self) -> None:
        missing_marker_repo = self.temp_root / "missing-marker"
        missing_marker_repo.mkdir()

        with self.assertRaisesRegex(EvalSafetyError, "marker 不存在"):
            validate_eval_temp_repo(missing_marker_repo, "run-123")

        repo_path = self.temp_root / "wrong-run"
        repo_path.mkdir()
        write_eval_temp_marker(repo_path, "run-123", "case-abc", self.temp_root)

        with self.assertRaisesRegex(EvalSafetyError, "run_id 不匹配"):
            validate_eval_temp_repo(repo_path, "run-456")

    def test_marker_validation_rejects_repo_outside_marker_temp_root(self) -> None:
        repo_path = self.temp_root / "repo"
        repo_path.mkdir()
        marker_path = write_eval_temp_marker(repo_path, "run-123", "case-abc", self.temp_root)
        wrong_temp_root = self.temp_root / "other-temp-root"
        wrong_temp_root.mkdir()
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_payload["temp_root"] = str(wrong_temp_root.resolve())
        marker_path.write_text(json.dumps(marker_payload), encoding="utf-8")

        with self.assertRaisesRegex(EvalSafetyError, "temp_root"):
            validate_eval_temp_repo(repo_path, "run-123")

    def test_copy_fixture_to_temp_does_not_pollute_original_fixture(self) -> None:
        fixture_path = self.project_root / "tests" / "fixtures" / "simple_python_project"
        original_app_text = (fixture_path / "app.py").read_text(encoding="utf-8")

        copied_repo_path = copy_fixture_to_temp(
            fixture_path=fixture_path,
            temp_root=self.temp_root,
            run_id="run-123",
            case_id="copy-fixture",
        )
        (copied_repo_path / "app.py").write_text("def add(a, b):\n    return 42\n", encoding="utf-8")

        self.assertNotEqual(original_app_text, (copied_repo_path / "app.py").read_text(encoding="utf-8"))
        self.assertEqual(original_app_text, (fixture_path / "app.py").read_text(encoding="utf-8"))
        validate_eval_temp_repo(copied_repo_path, "run-123")

    def test_snapshot_detects_added_modified_and_deleted_files(self) -> None:
        repo_path = self.temp_root / "snapshot-repo"
        repo_path.mkdir()
        (repo_path / "keep.txt").write_text("same\n", encoding="utf-8")
        (repo_path / "modify.txt").write_text("before\n", encoding="utf-8")
        (repo_path / "delete.txt").write_text("delete me\n", encoding="utf-8")
        (repo_path / ".repopilot").mkdir()
        (repo_path / ".repopilot" / "ignored.txt").write_text("ignored\n", encoding="utf-8")

        before = snapshot_repo_files(repo_path)
        (repo_path / "modify.txt").write_text("after\n", encoding="utf-8")
        (repo_path / "delete.txt").unlink()
        (repo_path / "added.txt").write_text("added\n", encoding="utf-8")
        after = snapshot_repo_files(repo_path)

        snapshot_diff = compare_snapshots(before, after)

        self.assertEqual(("added.txt",), snapshot_diff.added)
        self.assertEqual(("modify.txt",), snapshot_diff.modified)
        self.assertEqual(("delete.txt",), snapshot_diff.deleted)
        self.assertEqual(("added.txt", "delete.txt", "modify.txt"), snapshot_diff.changed_files)
        self.assertNotIn(".repopilot/ignored.txt", before)

    def test_allowed_changed_files_passes_allowed_paths_and_fails_unauthorized_paths(self) -> None:
        allowed_errors = check_allowed_changed_files(
            changed_files=["README.md", "src\\app.py"],
            allowed_changed_files=["README.md", "src/app.py"],
        )
        unauthorized_errors = check_allowed_changed_files(
            changed_files=["README.md", "src/app.py", "deleted.py"],
            allowed_changed_files=["README.md"],
        )

        self.assertEqual([], allowed_errors)
        self.assertIn("未授权的变更文件: src/app.py", unauthorized_errors)
        self.assertIn("未授权的变更文件: deleted.py", unauthorized_errors)

    def test_allowed_changed_files_fails_unauthorized_added_modified_and_deleted_files(self) -> None:
        repo_path = self.temp_root / "change-types-repo"
        repo_path.mkdir()
        (repo_path / "allowed.txt").write_text("same\n", encoding="utf-8")
        (repo_path / "modified.txt").write_text("before\n", encoding="utf-8")
        (repo_path / "deleted.txt").write_text("before\n", encoding="utf-8")
        before = snapshot_repo_files(repo_path)

        (repo_path / "added.txt").write_text("after\n", encoding="utf-8")
        (repo_path / "modified.txt").write_text("after\n", encoding="utf-8")
        (repo_path / "deleted.txt").unlink()
        after = snapshot_repo_files(repo_path)
        snapshot_diff = compare_snapshots(before, after)

        errors = check_allowed_changed_files(
            changed_files=snapshot_diff.changed_files,
            allowed_changed_files=["allowed.txt"],
        )

        self.assertEqual(("added.txt",), snapshot_diff.added)
        self.assertEqual(("modified.txt",), snapshot_diff.modified)
        self.assertEqual(("deleted.txt",), snapshot_diff.deleted)
        self.assertIn("未授权的变更文件: added.txt", errors)
        self.assertIn("未授权的变更文件: modified.txt", errors)
        self.assertIn("未授权的变更文件: deleted.txt", errors)

    def test_normalize_relative_path_handles_windows_paths_safely(self) -> None:
        self.assertEqual("src/app.py", eval_safety.normalize_relative_path("src\\app.py"))

        for raw_path in ["C:\\repo\\file.txt", "D:/repo/file.txt", "..\\outside.txt"]:
            with self.subTest(raw_path=raw_path):
                with self.assertRaises(EvalSafetyError):
                    eval_safety.normalize_relative_path(raw_path)

    def test_must_contain_passes_existing_text_and_reports_missing_text(self) -> None:
        repo_path = self.temp_root / "must-contain-repo"
        repo_path.mkdir()
        (repo_path / "README.md").write_text("hello eval\n", encoding="utf-8")

        passing_errors = check_must_contain(
            repo_path,
            [MustContainRule(path="README.md", strings=("hello", "eval"))],
        )
        failing_errors = check_must_contain(
            repo_path,
            [MustContainRule(path="README.md", strings=("missing text",))],
        )

        self.assertEqual([], passing_errors)
        self.assertEqual(["must_contain 缺少文本: README.md -> missing text"], failing_errors)


class TestEditEvalRunner(unittest.TestCase):
    """验证 edit eval runner 核心执行流程。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent

    def _case(
        self,
        case_id: str,
        max_steps: int = 5,
        allowed_changed_files: tuple[str, ...] = (),
        must_contain: tuple[MustContainRule, ...] = (),
        test_command: str | None = None,
        expect_no_business_changes: bool = False,
    ) -> EditEvalCase:
        return EditEvalCase(
            id=case_id,
            fixture="tests/fixtures/simple_python_project",
            prompt=f"执行 {case_id}",
            max_steps=max_steps,
            allowed_changed_files=allowed_changed_files,
            must_contain=must_contain,
            test_command=test_command,
            expect_no_business_changes=expect_no_business_changes,
        )

    def _write_cases(self, cases: list[dict[str, object]]) -> Path:
        temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temp_directory.cleanup)
        cases_path = Path(temp_directory.name) / "edit_cases.json"
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return cases_path

    def test_run_edit_case_passes_successful_readme_edit_with_fake_llm(self) -> None:
        case = self._case(
            case_id="readme-success",
            allowed_changed_files=("README.md",),
            must_contain=(
                MustContainRule(
                    path="README.md",
                    strings=("本项目用于验证 agent 的安全编辑能力",),
                ),
            ),
        )
        fake_llm = FakeLlmClient(
            [
                _tool_call("read_file", {"path": "README.md"}),
                _readme_patch_call(),
                _apply_last_patch_call,
                _run_compile_tests_call(),
                _tool_call("finish", {"answer": "README.md 已更新。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual(("README.md",), eval_result.changed_files)
        self.assertEqual(5, eval_result.steps)
        self.assertIsNotNone(eval_result.final_answer)
        assert eval_result.final_answer is not None
        self.assertIsNotNone(eval_result.final_answer)
        assert eval_result.final_answer is not None
        self.assertTrue(eval_result.final_answer.startswith("README.md 已更新。"), eval_result.final_answer)
        self.assertIn("v0.6 summary:", eval_result.final_answer)
        self.assertIsNone(eval_result.error)
        self.assertIsNone(eval_result.failure_type)

    def test_run_edit_case_records_model_usage_fields_from_run_log(self) -> None:
        case = self._case(case_id="usage-fields")
        fake_llm = StructuredFakeLlmClient(
            [_tool_call("finish", {"answer": "无需修改。"})]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual("mock-profile", eval_result.model_profile)
        self.assertEqual("mock", eval_result.provider)
        self.assertEqual("mock-model", eval_result.model)
        self.assertEqual(2, eval_result.llm_call_count)
        self.assertEqual(25.0, eval_result.total_latency_ms)
        self.assertEqual(30, eval_result.total_tokens)
        self.assertEqual(0, eval_result.estimated_tokens)
        self.assertEqual(0.5, eval_result.estimated_cost)
        self.assertIsNone(eval_result.failure_type)

    def test_run_edit_case_fails_unauthorized_changed_file(self) -> None:
        case = self._case(
            case_id="unauthorized-app-change",
            allowed_changed_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _app_patch_call(),
                _apply_last_patch_call,
                _run_compile_tests_call(),
                _tool_call("finish", {"answer": "app.py 已更新。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertEqual(("app.py",), eval_result.changed_files)
        self.assertIn("未授权的变更文件: app.py", eval_result.reasons)
        self.assertEqual("tool_policy_violation", eval_result.failure_type)

    def test_run_edit_case_fails_unauthorized_added_file(self) -> None:
        case = self._case(
            case_id="unauthorized-new-file",
            allowed_changed_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _new_file_patch_call("notes.txt", "unauthorized note\n"),
                _apply_last_patch_call,
                _run_compile_tests_call(),
                _tool_call("finish", {"answer": "notes.txt 已新增。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertEqual(("notes.txt",), eval_result.changed_files)
        self.assertIn("未授权的变更文件: notes.txt", eval_result.reasons)

    def test_run_edit_case_forbidden_outside_file_keeps_external_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as sentinel_directory:
            sentinel_path = Path(sentinel_directory) / "outside.txt"
            sentinel_path.write_text("external sentinel\n", encoding="utf-8")
            case = self._case(
                case_id="forbidden-outside-file",
                expect_no_business_changes=True,
            )
            fake_llm = FakeLlmClient(
                [
                    _tool_call("read_file", {"path": "../outside.txt"}),
                    _tool_call("finish", {"answer": "外部路径已被拒绝。"}),
                ]
            )

            eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

            self.assertTrue(eval_result.passed, eval_result.reasons)
            self.assertEqual((), eval_result.changed_files)
            self.assertEqual("external sentinel\n", sentinel_path.read_text(encoding="utf-8"))

    def test_run_edit_case_fails_when_finish_missing_after_max_steps(self) -> None:
        case = self._case(case_id="missing-finish", max_steps=1)
        fake_llm = FakeLlmClient([_tool_call("list_dir", {"path": "."})])

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertEqual(1, eval_result.steps)
        self.assertIsNotNone(eval_result.error)
        self.assertIn("agent 未在 max_steps 内成功调用 finish", eval_result.reasons)
        self.assertEqual("max_steps_exceeded", eval_result.failure_type)

    def test_run_edit_case_records_changed_files_when_llm_raises_after_patch(self) -> None:
        case = self._case(
            case_id="exception-after-patch",
            allowed_changed_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _readme_patch_call(),
                _apply_last_patch_call,
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertEqual(("README.md",), eval_result.changed_files)
        self.assertIsNotNone(eval_result.error)
        self.assertTrue(
            any(reason.startswith("case 执行异常:") for reason in eval_result.reasons),
            eval_result.reasons,
        )

    def test_run_edit_case_records_test_command_result(self) -> None:
        case = self._case(case_id="compile-check", test_command="compile")
        fake_llm = FakeLlmClient(
            [_tool_call("finish", {"answer": "无需修改，直接验证。"})]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertIsNotNone(eval_result.test_results)
        assert eval_result.test_results is not None
        self.assertEqual("compile", eval_result.test_results["command_name"])
        self.assertEqual(0, eval_result.test_results["exit_code"])
        self.assertFalse(eval_result.test_results["timed_out"])

    def test_run_edit_case_does_not_write_real_project_logs(self) -> None:
        case = self._case(case_id="no-real-logs")
        fake_llm = FakeLlmClient(
            [_tool_call("finish", {"answer": "只验证日志目录。"})]
        )
        real_log_dir = self.project_root / "logs"
        before_log_snapshot = self._snapshot_log_dir(real_log_dir)

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        after_log_snapshot = self._snapshot_log_dir(real_log_dir)
        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual(before_log_snapshot, after_log_snapshot)

    def test_run_edit_eval_continues_after_failed_case(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "first-fails",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "修改未授权文件",
                    "max_steps": 5,
                    "allowed_changed_files": ["README.md"],
                    "must_contain": [],
                },
                {
                    "id": "second-passes",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "直接完成",
                    "max_steps": 2,
                    "allowed_changed_files": [],
                    "must_contain": [],
                },
            ]
        )

        def fake_factory(case: EditEvalCase) -> FakeLlmClient:
            if case.id == "first-fails":
                return FakeLlmClient(
                    [
                        _app_patch_call(),
                        _apply_last_patch_call,
                        _run_compile_tests_call(),
                        _tool_call("finish", {"answer": "app.py 已更新。"}),
                    ]
                )
            return FakeLlmClient(
                [_tool_call("finish", {"answer": "第二条完成。"})]
            )

        summary = run_edit_eval(cases_path, self.project_root, fake_factory)

        self.assertEqual(2, summary["total"])
        self.assertEqual(1, summary["passed"])
        self.assertEqual(0.5, summary["pass_rate"])
        results = summary["results"]
        self.assertIsInstance(results, list)
        self.assertFalse(results[0]["passed"])
        self.assertTrue(results[1]["passed"])

    def test_run_edit_eval_isolates_file_changes_between_cases(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "first-edits-app",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "修改 app.py",
                    "max_steps": 5,
                    "allowed_changed_files": ["app.py"],
                    "must_contain": [
                        {
                            "path": "app.py",
                            "strings": ["return 42"],
                        }
                    ],
                },
                {
                    "id": "second-sees-clean-fixture",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "直接完成",
                    "max_steps": 2,
                    "allowed_changed_files": [],
                    "must_contain": [
                        {
                            "path": "app.py",
                            "strings": ["return a + b"],
                        }
                    ],
                },
            ]
        )

        def fake_factory(case: EditEvalCase) -> FakeLlmClient:
            if case.id == "first-edits-app":
                return FakeLlmClient(
                    [
                        _app_patch_call(),
                        _apply_last_patch_call,
                        _run_compile_tests_call(),
                        _tool_call("finish", {"answer": "app.py 已更新。"}),
                    ]
                )
            return FakeLlmClient(
                [_tool_call("finish", {"answer": "第二条看到干净 fixture。"})]
            )

        summary = run_edit_eval(cases_path, self.project_root, fake_factory)

        results = summary["results"]
        self.assertIsInstance(results, list)
        self.assertTrue(results[0]["passed"], results[0]["reasons"])
        self.assertTrue(results[1]["passed"], results[1]["reasons"])
        self.assertEqual(("app.py",), results[0]["changed_files"])
        self.assertEqual((), results[1]["changed_files"])

    def test_pending_approval_patch_metadata_and_cli_apply_reject_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            apply_repo = self._create_pending_eval_repo(temp_root / "apply-repo")
            reject_repo = self._create_pending_eval_repo(temp_root / "reject-repo")

            apply_patch_id = self._generate_pending_patch(apply_repo)
            reject_patch_id = self._generate_pending_patch(reject_repo)

            self._assert_pending_patch_metadata(apply_repo, apply_patch_id)
            self._assert_pending_patch_metadata(reject_repo, reject_patch_id)
            self.assertEqual("old line\n", (apply_repo / "sample.txt").read_text(encoding="utf-8"))
            self.assertEqual("old line\n", (reject_repo / "sample.txt").read_text(encoding="utf-8"))

            reject_exit_code, reject_output = self._run_patch_cli("reject", reject_repo, reject_patch_id)
            rejected_metadata = self._read_patch_metadata(reject_repo, reject_patch_id)
            self.assertEqual(0, reject_exit_code)
            self.assertTrue(reject_output["ok"])
            self.assertEqual("rejected", reject_output["status"])
            self.assertEqual("rejected", rejected_metadata["status"])
            self.assertEqual("old line\n", (reject_repo / "sample.txt").read_text(encoding="utf-8"))

            apply_exit_code, apply_output = self._run_patch_cli("apply", apply_repo, apply_patch_id)
            applied_metadata = self._read_patch_metadata(apply_repo, apply_patch_id)
            self.assertEqual(0, apply_exit_code)
            self.assertTrue(apply_output["ok"])
            self.assertEqual("applied", apply_output["status"])
            self.assertEqual("applied", applied_metadata["status"])
            self.assertEqual("new line\n", (apply_repo / "sample.txt").read_text(encoding="utf-8"))

    def _snapshot_log_dir(self, log_dir: Path) -> tuple[tuple[str, int, int], ...]:
        """记录日志目录的快照状态，用于验证 eval 不污染真实日志。"""
        if not log_dir.exists():
            return ()
        snapshots: list[tuple[str, int, int]] = []
        for log_path in log_dir.rglob("*"):
            if not log_path.is_file():
                continue
            file_stat = log_path.stat()
            snapshots.append(
                (
                    log_path.relative_to(log_dir).as_posix(),
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                )
            )
        return tuple(sorted(snapshots))

    def _create_pending_eval_repo(self, repo_path: Path) -> Path:
        """创建包含 sample.txt 的临时评估仓库。"""
        repo_path.mkdir()
        (repo_path / "sample.txt").write_text("old line\n", encoding="utf-8")
        return repo_path

    def _generate_pending_patch(self, repo_path: Path) -> str:
        """使用 fake LLM 驱动 agent 生成待审批补丁。"""
        diff_text = _unified_diff("sample.txt", "old line\n", "new line\n")
        fake_llm = FakeLlmClient(
            [
                _task_plan_json(
                    task_type="edit",
                    requires_patch=True,
                    expected_changed_files=("sample.txt",),
                ),
                _tool_call(
                    "propose_patch",
                    {"instruction": "更新 sample.txt。", "diff": diff_text},
                ),
            ],
            prepend_plan=False,
        )
        run_logger = RunLogger(
            repo_path=repo_path,
            user_task="生成待审批补丁",
            log_dir=repo_path / "logs",
        )
        agent = CodeAnalysisAgent(
            llm_client=cast(LlmClient, fake_llm),
            repository_tools=RepositoryTools(repo_path),
            run_logger=run_logger,
            max_steps=2,
            pending_approval_mode=True,
        )

        final_answer = agent.answer("更新 sample.txt")

        self.assertEqual("old line\n", (repo_path / "sample.txt").read_text(encoding="utf-8"))
        self.assertIn("补丁已生成但尚未应用", final_answer)
        stage_history = run_logger.payload["stage_history"]
        self.assertIsInstance(stage_history, list)
        self.assertIn("AWAITING_APPROVAL", cast(list[object], stage_history))
        patch_id_line = next(
            line for line in final_answer.splitlines() if line.startswith("patch_id: ")
        )
        return patch_id_line.removeprefix("patch_id: ")

    def _assert_pending_patch_metadata(self, repo_path: Path, patch_id: str) -> None:
        """验证待审批补丁的 metadata 字段。"""
        metadata = self._read_patch_metadata(repo_path, patch_id)
        self.assertEqual(patch_id, metadata["patch_id"])
        self.assertEqual("pending_approval", metadata["status"])
        self.assertEqual(["sample.txt"], metadata["paths"])

    def _read_patch_metadata(self, repo_path: Path, patch_id: str) -> dict[str, object]:
        """读取补丁 metadata.json 文件。"""
        metadata_path = repo_path / ".repopilot" / "patches" / patch_id / "metadata.json"
        self.assertTrue(metadata_path.is_file(), metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertIsInstance(metadata, dict)
        return metadata

    def _run_patch_cli(
        self,
        patch_command: str,
        repo_path: Path,
        patch_id: str,
    ) -> tuple[int, dict[str, object]]:
        """以指定子命令运行 main.py patch CLI 并解析 JSON 输出。"""
        stdout = io.StringIO()
        argv = ["main.py", "patch", patch_command, "--repo", str(repo_path), patch_id]
        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
            exit_code = cli_main()
        output = json.loads(stdout.getvalue())
        self.assertIsInstance(output, dict)
        return exit_code, output


class TestEvalRunnerV06LogChecks(unittest.TestCase):
    """验证 v0.6 计划、阶段、验证和修复次数检查。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent

    def _case(
        self,
        case_id: str,
        must_have_plan: bool = False,
        required_stages: tuple[str, ...] = (),
        must_reference_plan_steps: bool = False,
        require_verify_after_patch: bool = False,
        max_repair_attempts: int | None = None,
    ) -> EditEvalCase:
        return EditEvalCase(
            id=case_id,
            fixture="tests/fixtures/simple_python_project",
            prompt=f"执行 {case_id}",
            max_steps=6,
            allowed_changed_files=("README.md",),
            must_contain=(),
            must_have_plan=must_have_plan,
            required_stages=required_stages,
            must_reference_plan_steps=must_reference_plan_steps,
            require_verify_after_patch=require_verify_after_patch,
            max_repair_attempts=max_repair_attempts,
        )

    def _payload(
        self,
        stage_history: list[str] | None = None,
        plan_step_id: str = "step-1",
        repair_attempts: int = 0,
        include_plan: bool = True,
        include_apply_patch: bool = True,
    ) -> dict[str, object]:
        """构造用于 v0.6 log 约束检查的 payload 字典。"""
        steps: list[dict[str, object]] = []
        if include_apply_patch:
            steps.append(
                {
                    "step": 1,
                    "tool_call": {
                        "tool": "apply_patch",
                        "args": {"patch_id": "patch-1"},
                        "plan_step_id": plan_step_id,
                    },
                    "tool_result": {"ok": True, "output": {"ok": True}},
                }
            )
        steps.append(
            {
                "step": 2,
                "tool_call": {
                    "tool": "finish",
                    "args": {"answer": "done"},
                    "plan_step_id": plan_step_id,
                },
                "tool_result": {"ok": True, "output": "done"},
            }
        )
        payload: dict[str, object] = {
            "stage_history": stage_history or ["INIT", "PLAN", "EXECUTE", "VERIFY", "FINISH"],
            "plan": {
                "steps": [
                    {"id": "step-1", "title": "执行", "description": "fake"},
                ]
            }
            if include_plan
            else None,
            "steps": steps,
            "repair_attempts": repair_attempts,
        }
        return payload

    def test_run_edit_case_passes_all_v06_checks_with_fake_llm(self) -> None:
        case = self._case(
            "v06-success",
            must_have_plan=True,
            required_stages=("PLAN", "EXECUTE", "VERIFY", "FINISH"),
            must_reference_plan_steps=True,
            require_verify_after_patch=True,
            max_repair_attempts=0,
        )
        fake_llm = FakeLlmClient(
            [
                _tool_call("read_file", {"path": "README.md"}),
                _readme_patch_call(),
                _apply_last_patch_call,
                _run_compile_tests_call(),
                _tool_call("finish", {"answer": "README.md 已更新。"}),
            ],
            prepend_plan=False,
        )
        fake_llm.outputs.insert(
            0,
            _task_plan_json(
                task_type="edit",
                requires_patch=True,
                requires_tests=True,
                expected_changed_files=("README.md",),
            ),
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)

    def test_must_have_plan_fails_when_plan_missing(self) -> None:
        case = self._case("missing-plan", must_have_plan=True)

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(include_plan=False),
        )

        self.assertIn("must_have_plan 违规: run log 缺少 plan", errors)

    def test_required_stages_reports_missing_stage(self) -> None:
        case = self._case("missing-stage", required_stages=("PLAN", "VERIFY"))

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(stage_history=["INIT", "PLAN", "EXECUTE"]),
        )

        self.assertTrue(any("required_stages 违规" in error and "VERIFY" in error for error in errors), errors)

    def test_must_reference_plan_steps_reports_unknown_step_id(self) -> None:
        case = self._case("bad-step", must_reference_plan_steps=True)

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(plan_step_id="unknown-step"),
        )

        self.assertTrue(
            any(
                "must_reference_plan_steps 违规" in error
                and "unknown-step" in error
                and "step-1" in error
                for error in errors
            ),
            errors,
        )

    def test_require_verify_after_patch_reports_missing_verify_stage(self) -> None:
        case = self._case("missing-verify", require_verify_after_patch=True)

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(stage_history=["INIT", "PLAN", "EXECUTE", "FINISH"]),
        )

        self.assertEqual(
            ["require_verify_after_patch 违规: 成功 apply_patch 后 stage_history 缺少 VERIFY"],
            errors,
        )

    def test_require_verify_after_patch_ignores_runs_without_successful_patch(self) -> None:
        case = self._case("no-patch", require_verify_after_patch=True)

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(stage_history=["INIT", "PLAN", "EXECUTE"], include_apply_patch=False),
        )

        self.assertEqual([], errors)

    def test_max_repair_attempts_accepts_legacy_singular_field(self) -> None:
        case = self._case("legacy-repair", max_repair_attempts=1)
        payload = self._payload()
        payload.pop("repair_attempts")
        payload["repair_attempt"] = 1

        errors = eval_runner._check_run_log_constraints(case, payload)

        self.assertEqual([], errors)

    def test_max_repair_attempts_reports_exceeded_limit(self) -> None:
        case = self._case("repair-exceeded", max_repair_attempts=0)

        errors = eval_runner._check_run_log_constraints(
            case,
            self._payload(repair_attempts=1),
        )

        self.assertEqual(["max_repair_attempts 违规: limit=0, actual=1"], errors)


class TestEvalFailureClassification(unittest.TestCase):
    """验证 edit/context/stage eval 复用的失败分类。"""

    def test_success_has_no_failure_type(self) -> None:
        self.assertIsNone(
            classify_failure_type(
                passed=True,
                reasons=(),
                error=None,
                test_results=None,
            )
        )

    def test_classifies_known_failure_reasons(self) -> None:
        cases = [
            ("invalid_json", ("case 执行异常: 模型输出不是严格 JSON 对象",), None, None),
            ("provider_error", ("case 执行异常: LLM HTTP 错误 500",), "LLM HTTP 错误 500", None),
            ("timeout", ("case 执行异常: LLM 网络或超时错误: timed out",), "timed out", None),
            ("phase_policy_violation", ("required_stages 违规: missing=VERIFY",), None, None),
            ("phase_policy_violation", ("require_verify_after_patch 违规: 成功 apply_patch 后 stage_history 缺少 VERIFY",), None, None),
            ("max_steps_exceeded", ("agent 未在 max_steps 内成功调用 finish",), None, None),
            ("test_failed", ("test_command unit 退出码非 0: 1",), None, {"exit_code": 1, "timed_out": False}),
            ("verification_failed", ("case 执行异常: VERIFY 失败: tests_failed",), "VERIFY 失败: tests_failed", None),
            ("patch_not_applied", ("must_contain 缺少文本: README.md -> marker",), None, None),
            ("patch_invalid", ("case 执行异常: patch 格式错误",), "patch 格式错误", None),
            ("tool_policy_violation", ("未授权的变更文件: app.py",), None, None),
            ("unknown", ("无法归类的失败",), None, None),
        ]
        for expected_failure_type, reasons, error, test_results in cases:
            with self.subTest(expected_failure_type=expected_failure_type, reasons=reasons):
                self.assertEqual(
                    expected_failure_type,
                    classify_failure_type(
                        passed=False,
                        reasons=reasons,
                        error=error,
                        test_results=test_results,
                    ),
                )

    def test_classifies_timed_out_test_as_timeout_before_test_failed(self) -> None:
        self.assertEqual(
            "timeout",
            classify_failure_type(
                passed=False,
                reasons=("test_command unit 执行超时", "test_command unit 退出码非 0: 1"),
                error=None,
                test_results={"exit_code": 1, "timed_out": True},
            ),
        )


class TestContextEvalV05(unittest.TestCase):
    """验证 v0.5 context 约束会进入 edit eval 判定。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent

    def _case(
        self,
        case_id: str,
        must_not_read_full_files: tuple[str, ...] = (),
        max_total_tool_output_chars: int | None = None,
    ) -> EditEvalCase:
        return EditEvalCase(
            id=case_id,
            fixture="tests/fixtures/simple_python_project",
            prompt=f"执行 {case_id}",
            max_steps=4,
            allowed_changed_files=(),
            must_contain=(),
            must_not_read_full_files=must_not_read_full_files,
            max_total_tool_output_chars=max_total_tool_output_chars,
        )

    def _write_cases(self, cases: list[dict[str, object]]) -> Path:
        temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temp_directory.cleanup)
        cases_path = Path(temp_directory.name) / "edit_cases.json"
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        return cases_path

    def test_must_not_read_full_files_fails_case(self) -> None:
        case = self._case(
            case_id="forbidden-full-read",
            must_not_read_full_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _tool_call("read_file", {"path": "README.md"}),
                _tool_call("finish", {"answer": "README.md 已读取。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertIsNotNone(eval_result.context_stats)
        assert eval_result.context_stats is not None
        self.assertEqual(["README.md"], eval_result.context_stats["full_file_reads"])
        self.assertTrue(
            any(
                "must_not_read_full_files" in reason
                and "README.md" in reason
                and "actual_full_file_reads" in reason
                for reason in eval_result.reasons
            ),
            eval_result.reasons,
        )

    def test_max_total_tool_output_chars_fails_case(self) -> None:
        case = self._case(
            case_id="output-budget",
            max_total_tool_output_chars=1,
        )
        fake_llm = FakeLlmClient(
            [_tool_call("finish", {"answer": "无需修改。"})]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertIsNotNone(eval_result.context_stats)
        assert eval_result.context_stats is not None
        actual_chars = eval_result.context_stats["total_tool_output_chars"]
        self.assertIsInstance(actual_chars, int)
        self.assertGreater(actual_chars, 1)
        self.assertTrue(
            any(
                "max_total_tool_output_chars" in reason
                and "limit=1" in reason
                and f"actual={actual_chars}" in reason
                for reason in eval_result.reasons
            ),
            eval_result.reasons,
        )

    def test_context_stats_included_in_successful_result(self) -> None:
        case = self._case(case_id="context-stats-included")
        fake_llm = FakeLlmClient(
            [_tool_call("finish", {"answer": "无需修改。"})]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertIsNotNone(eval_result.context_stats)
        assert eval_result.context_stats is not None
        self.assertEqual(1, eval_result.context_stats["steps_used"])
        self.assertIn("total_tool_output_chars", eval_result.context_stats)

    def test_loads_valid_context_fields(self) -> None:
        cases_path = self._write_cases(
            [
                {
                    "id": "context-fields",
                    "fixture": "tests/fixtures/simple_python_project",
                    "prompt": "validate context fields",
                    "allowed_changed_files": [],
                    "must_contain": [],
                    "must_not_read_full_files": ["docs\\guide.md", "src/main.py"],
                    "max_total_tool_output_chars": 123,
                }
            ]
        )

        cases = load_edit_cases(cases_path)

        self.assertEqual(("docs/guide.md", "src/main.py"), cases[0].must_not_read_full_files)
        self.assertEqual(123, cases[0].max_total_tool_output_chars)

    def test_loads_context_cases(self) -> None:
        cases_path = self.project_root / "eval_cases" / "context_cases.json"

        cases = load_edit_cases(cases_path)
        cases_by_id = {case.id: case for case in cases}

        self.assertEqual(
            {
                "avoid-large-unrelated-file",
                "edit-targeted-file-with-context-budget",
                "verify-context-budget-still-enforced",
            },
            set(cases_by_id),
        )
        avoid_large_case = cases_by_id["avoid-large-unrelated-file"]
        self.assertEqual("tests/fixtures/medium_python_project", avoid_large_case.fixture)
        self.assertEqual(("large_notes.md",), avoid_large_case.must_not_read_full_files)
        self.assertEqual(12000, avoid_large_case.max_total_tool_output_chars)
        self.assertEqual((), avoid_large_case.allowed_changed_files)
        self.assertEqual((), avoid_large_case.must_contain)

        targeted_edit_case = cases_by_id["edit-targeted-file-with-context-budget"]
        self.assertEqual(("app/services.py",), targeted_edit_case.allowed_changed_files)
        self.assertEqual(("large_notes.md",), targeted_edit_case.must_not_read_full_files)
        self.assertEqual(12000, targeted_edit_case.max_total_tool_output_chars)
        self.assertEqual("unit", targeted_edit_case.test_command)
        self.assertEqual(1, len(targeted_edit_case.must_contain))
        self.assertEqual("app/services.py", targeted_edit_case.must_contain[0].path)
        self.assertEqual(
            ("label=invoice", "build_invoice_summary"),
            targeted_edit_case.must_contain[0].strings,
        )

        budget_case = cases_by_id["verify-context-budget-still-enforced"]
        self.assertTrue(budget_case.must_have_plan)
        self.assertEqual(("PLAN", "EXECUTE", "VERIFY", "FINISH"), budget_case.required_stages)
        self.assertTrue(budget_case.must_reference_plan_steps)
        self.assertEqual(0, budget_case.max_repair_attempts)

    def test_rejects_invalid_context_field_types(self) -> None:
        invalid_cases = [
            {
                "id": "bad-forbidden-type",
                "fixture": "tests/fixtures/simple_python_project",
                "prompt": "bad forbidden",
                "allowed_changed_files": [],
                "must_contain": [],
                "must_not_read_full_files": "README.md",
            },
            {
                "id": "bad-forbidden-path",
                "fixture": "tests/fixtures/simple_python_project",
                "prompt": "bad path",
                "allowed_changed_files": [],
                "must_contain": [],
                "must_not_read_full_files": ["../README.md"],
            },
            {
                "id": "bad-budget-bool",
                "fixture": "tests/fixtures/simple_python_project",
                "prompt": "bad budget bool",
                "allowed_changed_files": [],
                "must_contain": [],
                "max_total_tool_output_chars": True,
            },
            {
                "id": "bad-budget-zero",
                "fixture": "tests/fixtures/simple_python_project",
                "prompt": "bad budget zero",
                "allowed_changed_files": [],
                "must_contain": [],
                "max_total_tool_output_chars": 0,
            },
        ]
        for raw_case in invalid_cases:
            with self.subTest(case_id=raw_case["id"]):
                cases_path = self._write_cases([raw_case])
                with self.assertRaises(EditEvalConfigError):
                    load_edit_cases(cases_path)


class TestRunEditEvalCli(unittest.TestCase):
    """验证 edit eval CLI 摘要输出与结果持久化。"""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name)

    def test_main_writes_report_and_prints_safe_summary(self) -> None:
        cases_path = self.temp_root / "edit_cases.json"
        output_dir = self.temp_root / "evals"
        cases_path.write_text("[]", encoding="utf-8")
        eval_payload = {
            "total": 2,
            "passed": 1,
            "pass_rate": 0.5,
            "results": [
                {
                    "case_id": "passing-case",
                    "passed": True,
                    "reasons": [],
                    "changed_files": [],
                    "steps": 1,
                    "final_answer": "done",
                    "error": None,
                    "test_results": None,
                },
                {
                    "case_id": "failing-case",
                    "passed": False,
                    "reasons": ["缺少必要文本"],
                    "changed_files": ["README.md"],
                    "steps": 2,
                    "final_answer": None,
                    "error": "failed",
                    "test_results": None,
                    "failure_type": "patch_not_applied",
                },
            ],
        }
        argv = [
            "run_edit_eval.py",
            "--cases",
            str(cases_path),
            "--repo-root",
            str(self.project_root),
            "--output-dir",
            str(output_dir),
        ]

        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(
            run_edit_eval_cli,
            "run_edit_eval",
            return_value=eval_payload,
        ) as run_eval_mock, redirect_stdout(stdout):
            exit_code = run_edit_eval_cli.main()

        self.assertEqual(1, exit_code)
        run_eval_mock.assert_called_once_with(
            cases_path=cases_path.resolve(),
            project_root=self.project_root.resolve(),
        )
        output_text = stdout.getvalue()
        self.assertIn("[PASS] passing-case", output_text)
        self.assertIn("[FAIL] failing-case: 缺少必要文本", output_text)
        self.assertIn("failure_type=patch_not_applied", output_text)
        self.assertIn("pass rate: 1/2", output_text)
        self.assertNotIn("done", output_text)

        report_paths = list(output_dir.glob("*.json"))
        self.assertEqual(1, len(report_paths))
        report_payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "started_at",
                "finished_at",
                "cases_path",
                "total",
                "passed",
                "pass_rate",
                "results",
            },
            set(report_payload),
        )
        self.assertEqual(str(cases_path.resolve()), report_payload["cases_path"])
        self.assertEqual(2, report_payload["total"])
        self.assertEqual(1, report_payload["passed"])
        self.assertEqual(0.5, report_payload["pass_rate"])
        self.assertEqual(eval_payload["results"], report_payload["results"])

    def test_main_missing_cases_file_fails_without_report(self) -> None:
        missing_cases_path = self.temp_root / "missing_edit_cases.json"
        output_dir = self.temp_root / "evals"
        argv = [
            "run_edit_eval.py",
            "--cases",
            str(missing_cases_path),
            "--repo-root",
            str(self.project_root),
            "--output-dir",
            str(output_dir),
        ]

        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(stderr):
            exit_code = run_edit_eval_cli.main()

        self.assertEqual(1, exit_code)
        self.assertIn("edit eval 配置错误", stderr.getvalue())
        self.assertFalse(output_dir.exists())

    def test_bundled_deterministic_factory_drives_subset_eval(self) -> None:
        bundled_cases_path = self.project_root / "eval_cases" / "edit_cases.json"
        raw_cases = json.loads(bundled_cases_path.read_text(encoding="utf-8"))
        subset_cases = [case for case in raw_cases if case.get("id") == "update-readme"]
        cases_path = self.temp_root / "subset_edit_cases.json"
        cases_path.write_text(json.dumps(subset_cases, ensure_ascii=False), encoding="utf-8")
        llm_client_factory = run_edit_eval_cli._bundled_eval_llm_factory(
            bundled_cases_path.resolve(),
            self.project_root.resolve(),
        )

        self.assertIsNotNone(llm_client_factory)
        summary = run_edit_eval(cases_path, self.project_root, llm_client_factory)

        self.assertEqual(1, summary["total"])
        self.assertEqual(1, summary["passed"])
        results = summary["results"]
        self.assertIsInstance(results, list)
        self.assertEqual("update-readme", results[0]["case_id"])
        self.assertEqual(("README.md",), tuple(results[0]["changed_files"]))

    def test_main_uses_deterministic_factory_only_for_bundled_cases(self) -> None:
        bundled_cases_path = self.project_root / "eval_cases" / "context_cases.json"
        custom_cases_path = self.temp_root / "context_cases.json"
        custom_cases_path.write_text("[]", encoding="utf-8")
        output_dir = self.temp_root / "evals"
        eval_payload = {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "results": [],
        }

        bundled_argv = [
            "run_edit_eval.py",
            "--cases",
            str(bundled_cases_path),
            "--repo-root",
            str(self.project_root),
            "--output-dir",
            str(output_dir / "bundled"),
        ]
        custom_argv = [
            "run_edit_eval.py",
            "--cases",
            str(custom_cases_path),
            "--repo-root",
            str(self.project_root),
            "--output-dir",
            str(output_dir / "custom"),
        ]

        with patch.object(sys, "argv", bundled_argv), patch.object(
            run_edit_eval_cli,
            "run_edit_eval",
            return_value=eval_payload,
        ) as bundled_run_eval_mock, redirect_stdout(io.StringIO()):
            bundled_exit_code = run_edit_eval_cli.main()

        self.assertEqual(0, bundled_exit_code)
        bundled_call_kwargs = bundled_run_eval_mock.call_args.kwargs
        self.assertEqual(bundled_cases_path.resolve(), bundled_call_kwargs["cases_path"])
        self.assertEqual(self.project_root.resolve(), bundled_call_kwargs["project_root"])
        self.assertIn("llm_client_factory", bundled_call_kwargs)
        self.assertIsNotNone(bundled_call_kwargs["llm_client_factory"])

        with patch.object(sys, "argv", custom_argv), patch.object(
            run_edit_eval_cli,
            "run_edit_eval",
            return_value=eval_payload,
        ) as custom_run_eval_mock, redirect_stdout(io.StringIO()):
            custom_exit_code = run_edit_eval_cli.main()

        self.assertEqual(0, custom_exit_code)
        custom_run_eval_mock.assert_called_once_with(
            cases_path=custom_cases_path.resolve(),
            project_root=self.project_root.resolve(),
        )


class TestReadmeV04Docs(unittest.TestCase):
    """验证 README 记录 v0.4 edit eval 的用途、运行方式和安全边界。"""

    def setUp(self) -> None:
        project_root = Path(__file__).resolve().parent
        self.readme_text = (project_root / "README.md").read_text(encoding="utf-8")

    def test_documents_v04_edit_eval_purpose_and_command(self) -> None:
        for expected_text in [
            "v0.4 编辑评测",
            "验证 agent 是否能在安全边界内完成小型代码修改",
            "python run_edit_eval.py",
            ".repopilot/evals/<timestamp>.json",
            "[PASS] case_id",
            "[FAIL] case_id: reasons",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_temp_fixture_copy_safety(self) -> None:
        for expected_text in [
            "把 `fixture` 复制到临时目录后再运行 agent",
            "不会修改原始 fixture",
            "不会修改真实项目",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_auto_for_eval_scope_and_manual_cli_approval(self) -> None:
        for expected_text in [
            "`auto_for_eval` 只供评测 runner",
            "带 marker 校验的临时代码库",
            "普通 CLI 仍然要求用户手动批准 `apply_patch`",
            "不要在日常使用中手动启用或创建 `auto_for_eval` 批准",
            "仅限这些 bundled cases 的确定性 LLM fallback",
            "仍会驱动真实 `CodeAnalysisAgent`、安全工具、补丁提案",
            "自定义 eval 文件不会自动启用该 fallback",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_case_schema_and_whitelisted_test_command(self) -> None:
        for expected_text in [
            "`id`",
            "`fixture`",
            "`prompt`",
            "`max_steps`",
            "`allowed_changed_files`",
            "`must_contain`",
            "`test_command`",
            "`expect_no_business_changes`",
            "`test_command` 只支持 `unit` 和 `compile`",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_does_not_document_arbitrary_shell_eval(self) -> None:
        forbidden_texts = [
            "`test_command` 可执行任意",
            "test_command 可以执行任意",
            "arbitrary shell eval",
            "arbitrary shell execution",
        ]
        for forbidden_text in forbidden_texts:
            with self.subTest(forbidden_text=forbidden_text):
                self.assertNotIn(forbidden_text, self.readme_text)
        self.assertIn("no arbitrary shell", self.readme_text)
        self.assertIn("no framework", self.readme_text)


class TestReadmeV05Docs(unittest.TestCase):
    """验证 README 记录 v0.5 上下文管理、项目索引和边界约束。"""

    def setUp(self) -> None:
        project_root = Path(__file__).resolve().parent
        self.readme_text = (project_root / "README.md").read_text(encoding="utf-8")

    def test_documents_v05_section_and_index_tools(self) -> None:
        for expected_text in [
            "## v0.5 上下文管理与项目索引",
            "`build_repo_index(force=False)`",
            ".repopilot/index/<repo_id>/file_index.json",
            "`inspect_repo()`",
            "紧凑项目概览",
            "项目分析、定位入口或准备修改代码时优先使用它",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_read_search_thresholds(self) -> None:
        for expected_text in [
            "MAX_FULL_READ_LINES=300",
            "MAX_FULL_READ_BYTES=20_000",
            "MAX_RANGE_READ_LINES=120",
            "MAX_TOOL_OUTPUT_CHARS=12_000",
            "MAX_SEARCH_RESULTS=20",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_context_stats_schema(self) -> None:
        for expected_text in [
            "ContextStats",
            "`steps_used`",
            "`total_tool_output_chars`",
            "`messages_total_chars`",
            "`files_read`",
            "`ranges_read`",
            "`search_calls`",
            "`full_file_reads`",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_edit_and_context_eval_commands(self) -> None:
        for expected_text in [
            "python run_edit_eval.py --cases eval_cases/edit_cases.json",
            "python run_edit_eval.py --cases eval_cases/context_cases.json",
            "CLI 会记录结构化结果和上下文违规原因",
            "不承诺每次都通过",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

    def test_documents_forbidden_frameworks_and_shell_boundaries(self) -> None:
        for expected_text in [
            "不使用 LangChain、LangGraph、LlamaIndex、MCP 或 vector DB",
            "不提供 arbitrary shell",
            "不加入语义搜索、外部 agent 框架或联网协作",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, self.readme_text)

        for forbidden_text in [
            "可使用 LangChain",
            "可使用 LangGraph",
            "可使用 LlamaIndex",
            "可使用 MCP",
            "可使用 vector DB",
            "可提供 arbitrary shell",
            "支持任意 shell",
            "真实 LLM context cases always pass",
        ]:
            with self.subTest(forbidden_text=forbidden_text):
                self.assertNotIn(forbidden_text, self.readme_text)


if __name__ == "__main__":
    unittest.main()
