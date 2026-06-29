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
from unittest.mock import patch

from config import MAX_STEPS

eval_runner = importlib.import_module("eval_runner")
eval_safety = importlib.import_module("eval_safety")
run_edit_eval_cli = importlib.import_module("run_edit_eval")
EditEvalCase = eval_runner.EditEvalCase
EditEvalConfigError = eval_runner.EditEvalConfigError
EvalSafetyError = eval_safety.EvalSafetyError
MustContainRule = eval_runner.MustContainRule
check_allowed_changed_files = eval_runner.check_allowed_changed_files
check_must_contain = eval_runner.check_must_contain
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

    def __init__(self, outputs: Sequence[str | Callable[[list[dict[str, str]]], str]]) -> None:
        self.outputs = list(outputs)
        self.call_count = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self.call_count >= len(self.outputs):
            raise AssertionError("fake LLM 没有更多输出")
        output = self.outputs[self.call_count]
        self.call_count += 1
        if callable(output):
            return output(messages)
        return output


def _tool_call(tool: str, args: dict[str, object]) -> str:
    return json.dumps(
        {
            "thought": f"调用 {tool}",
            "tool": tool,
            "args": args,
        },
        ensure_ascii=False,
    )


def _unified_diff(path: str, before: str, after: str) -> str:
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join((f"diff --git a/{path} b/{path}", *diff_lines, ""))


def _apply_last_patch_call(messages: list[dict[str, str]]) -> str:
    latest_feedback = messages[-1]["content"]
    patch_id_match = re.search(r'"patch_id"\s*:\s*"([^"]+)"', latest_feedback)
    if patch_id_match is None:
        raise AssertionError(f"未找到 patch_id: {latest_feedback}")
    return _tool_call("apply_patch", {"patch_id": patch_id_match.group(1)})


def _readme_patch_call() -> str:
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
                _tool_call("finish", {"answer": "README.md 已更新。"}),
            ]
        )

        eval_result = run_edit_case(
            self._case("fake-finish"),
            self.project_root,
            lambda _case: fake_llm,
        )

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual(4, fake_llm.call_count)
        self.assertEqual(("README.md",), eval_result.changed_files)
        self.assertEqual("README.md 已更新。", eval_result.final_answer)
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

    def test_eval_prompt_documents_auto_approval_without_changing_case_prompt(self) -> None:
        eval_case = self._case("fake-auto-approval")

        eval_prompt = eval_runner._build_eval_agent_prompt(eval_case)

        self.assertIn(eval_case.prompt, eval_prompt)
        self.assertIn("auto_for_eval 自动批准", eval_prompt)
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

    def test_loads_valid_file_with_three_cases(self) -> None:
        cases_path = self.project_root / "eval_cases" / "edit_cases.json"
        cases = load_edit_cases(cases_path)

        self.assertEqual(3, len(cases))
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
                _tool_call("finish", {"answer": "README.md 已更新。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda _case: fake_llm)

        self.assertTrue(eval_result.passed, eval_result.reasons)
        self.assertEqual(("README.md",), eval_result.changed_files)
        self.assertEqual(4, eval_result.steps)
        self.assertEqual("README.md 已更新。", eval_result.final_answer)
        self.assertIsNone(eval_result.error)

    def test_run_edit_case_fails_unauthorized_changed_file(self) -> None:
        case = self._case(
            case_id="unauthorized-app-change",
            allowed_changed_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _app_patch_call(),
                _apply_last_patch_call,
                _tool_call("finish", {"answer": "app.py 已更新。"}),
            ]
        )

        eval_result = run_edit_case(case, self.project_root, lambda: fake_llm)

        self.assertFalse(eval_result.passed)
        self.assertEqual(("app.py",), eval_result.changed_files)
        self.assertIn("未授权的变更文件: app.py", eval_result.reasons)

    def test_run_edit_case_fails_unauthorized_added_file(self) -> None:
        case = self._case(
            case_id="unauthorized-new-file",
            allowed_changed_files=("README.md",),
        )
        fake_llm = FakeLlmClient(
            [
                _new_file_patch_call("notes.txt", "unauthorized note\n"),
                _apply_last_patch_call,
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
                    "max_steps": 4,
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
                    "max_steps": 4,
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

    def _snapshot_log_dir(self, log_dir: Path) -> tuple[tuple[str, int, int], ...]:
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
            {"avoid-large-unrelated-file", "edit-targeted-file-with-context-budget"},
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
