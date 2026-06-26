"""RepositoryTools 基础单元测试。"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import sys
import tools
from unittest import mock
from pathlib import Path
from config import MAX_FILE_BYTES
from tools import RepositoryTools, ToolError


class V03TempRepoHelper:
    """为 v0.3 变更测试提供隔离的临时代码库。"""

    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._temporary_directory.name).resolve()

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()

    def create_text_file(self, relative_path: str, content: str) -> Path:
        target_file = self.repo_root / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content, encoding="utf-8")
        return target_file

    def create_binary_file(self, relative_path: str, content: bytes) -> Path:
        target_file = self.repo_root / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(content)
        return target_file

    def create_repository_tools(self) -> RepositoryTools:
        return RepositoryTools(self.repo_root)

    def read_text_file(self, relative_path: str) -> str:
        return (self.repo_root / relative_path).read_text(encoding="utf-8")

    def repopilot_state_path(self) -> Path:
        return self.repo_root / ".repopilot"

    def repopilot_state_exists(self) -> bool:
        return self.repopilot_state_path().exists()


class TestV03TempRepoHelper(unittest.TestCase):
    """验证 v0.3 测试 helper 不污染真实项目根目录。"""

    real_repo_root = Path(__file__).resolve().parent
    root_artifact_names = (
        ".repopilot",
        ".env",
        "binary_target.bin",
        "proposed.patch",
        "apply.patch",
    )

    def setUp(self) -> None:
        self.root_artifact_state = self._snapshot_root_artifacts()
        self.temp_repo_helper = V03TempRepoHelper()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def _snapshot_root_artifacts(self) -> dict[str, tuple[bool, int | None]]:
        artifact_state: dict[str, tuple[bool, int | None]] = {}
        for artifact_name in self.root_artifact_names:
            artifact_path = self.real_repo_root / artifact_name
            if artifact_path.exists():
                artifact_state[artifact_name] = (True, artifact_path.stat().st_mtime_ns)
            else:
                artifact_state[artifact_name] = (False, None)
        return artifact_state

    def test_helper_creates_isolated_repo_tools_and_reads_state(self) -> None:
        created_file = self.temp_repo_helper.create_text_file(
            "src/example.txt",
            "alpha\nbeta\n",
        )
        repository_tools = self.temp_repo_helper.create_repository_tools()

        self.assertTrue(created_file.is_file())
        self.assertEqual(
            "alpha\nbeta\n",
            self.temp_repo_helper.read_text_file("src/example.txt"),
        )
        self.assertIn("src/", repository_tools.list_dir("."))
        self.assertIn("1 | alpha", repository_tools.read_file("src/example.txt"))
        self.assertEqual(self.temp_repo_helper.repo_root, repository_tools.repo_root)
        self.assertEqual(
            self.temp_repo_helper.repo_root / ".repopilot",
            self.temp_repo_helper.repopilot_state_path(),
        )
        self.assertTrue(self.temp_repo_helper.repopilot_state_exists())
        self.assertTrue((self.temp_repo_helper.repopilot_state_path() / "patches").is_dir())
        self.assertTrue((self.temp_repo_helper.repopilot_state_path() / "backups").is_dir())
        self.assertTrue((self.temp_repo_helper.repopilot_state_path() / "runs").is_dir())
        self.assertEqual(self.root_artifact_state, self._snapshot_root_artifacts())


class RepositoryToolsTest(unittest.TestCase):
    """验证只读代码库工具的基础行为和安全限制。"""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent
        self.repository_tools = RepositoryTools(self.repo_root)
        self.env_file = self.repo_root / ".env"
        self.created_env_file = False

        if not self.env_file.exists():
            self.env_file.write_text("SECRET=value\n", encoding="utf-8")
            self.created_env_file = True

    def tearDown(self) -> None:
        if self.created_env_file and self.env_file.exists():
            self.env_file.unlink()

    def assert_patch_errors_contain(
        self,
        patch_result: dict[str, object],
        expected_error_text: str,
    ) -> None:
        errors_object = patch_result["errors"]
        self.assertIsInstance(errors_object, list)
        if not isinstance(errors_object, list):
            self.fail("errors should be a list")
        self.assertTrue(
            any(
                isinstance(error, str) and expected_error_text in error
                for error in errors_object
            )
        )


    def test_list_dir_root_returns_project_files(self) -> None:
        entries = self.repository_tools.list_dir(".")

        self.assertIn("agent.py", entries)
        self.assertIn("tools.py", entries)
        self.assertIn("README.md", entries)

    def test_read_file_returns_numbered_agent_content(self) -> None:
        file_content = self.repository_tools.read_file("agent.py")

        self.assertIn('1 | """LLM + 工具调用循环。"""', file_content)
        self.assertIn("8 | from config import MAX_STEPS", file_content)

    def test_search_text_finds_max_steps_with_context(self) -> None:
        matches = self.repository_tools.search_text("MAX_STEPS")

        self.assertTrue(matches)
        self.assertTrue(
            any("agent.py:" in match and "MAX_STEPS" in match for match in matches)
        )
        self.assertTrue(any(" | " in match for match in matches))

    def test_read_file_rejects_parent_directory_escape(self) -> None:
        with self.assertRaisesRegex(ToolError, "repo 目录内部"):
            self.repository_tools.read_file("../outside.txt")

    def test_read_file_rejects_env_file(self) -> None:
        with self.assertRaisesRegex(ToolError, "隐藏路径|密钥或环境文件"):
            self.repository_tools.read_file(".env")

    def test_read_file_missing_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ToolError, "path 不是文件: not_exist.py"):
            self.repository_tools.read_file("not_exist.py")

    def test_search_text_unlikely_keyword_returns_empty_list(self) -> None:
        missing_keyword = "unlikely_keyword_" + "123456"
        matches = self.repository_tools.search_text(missing_keyword)

        self.assertEqual([], matches)


class TestRunTestsTool(unittest.TestCase):
    """验证 run_tests 白名单命令工具。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def _latest_run_events(self) -> list[dict[str, object]]:
        run_events = self.temp_repo_helper.repo_root / ".repopilot/runs"
        run_files = list(run_events.glob("*.jsonl"))
        if not run_files:
            return []
        latest_run_file = max(run_files, key=lambda path: path.stat().st_mtime)
        lines = [line.strip() for line in latest_run_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    def test_run_tests_unit_command(self) -> None:
        with mock.patch("tools.subprocess.run") as mocked_run:
            mocked_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "unittest", "discover"],
                returncode=0,
                stdout="unit passed",
                stderr="",
            )

            result = self.repository_tools.run_tests("unit")

            self.assertEqual("unit", result["command_name"])
            self.assertEqual(0, result["exit_code"])
            self.assertFalse(result["timed_out"])
            self.assertEqual("unit passed", result["stdout"])
            self.assertEqual("", result["stderr"])
            mocked_run.assert_called_once_with(
                [sys.executable, "-m", "unittest", "discover"],
                shell=False,
                cwd=self.temp_repo_helper.repo_root,
                text=True,
                capture_output=True,
                timeout=tools.RUN_TEST_TIMEOUT_SECONDS,
            )

            events = self._latest_run_events()
            self.assertTrue(events)
            event_types = {event.get("event_type", "") for event in events}
            self.assertIn("run_tests_start", event_types)
            self.assertIn("run_tests_end", event_types)
            self.assertNotIn("run_tests_truncation", event_types)
            self.assertNotIn("run_tests_timeout", event_types)

    def test_run_tests_compile_command(self) -> None:
        with mock.patch("tools.subprocess.run") as mocked_run:
            mocked_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "compileall", "."],
                returncode=0,
                stdout="compile ok",
                stderr="",
            )

            result = self.repository_tools.run_tests("compile")

            self.assertEqual("compile", result["command_name"])
            self.assertEqual(0, result["exit_code"])
            self.assertFalse(result["timed_out"])
            self.assertEqual("compile ok", result["stdout"])
            mocked_run.assert_called_once_with(
                [sys.executable, "-m", "compileall", "."],
                shell=False,
                cwd=self.temp_repo_helper.repo_root,
                text=True,
                capture_output=True,
                timeout=tools.RUN_TEST_TIMEOUT_SECONDS,
            )

            events = self._latest_run_events()
            self.assertTrue(events)
            event_types = {event.get("event_type", "") for event in events}
            self.assertIn("run_tests_start", event_types)
            self.assertIn("run_tests_end", event_types)

    def test_run_tests_rejects_unknown_or_command_like_input(self) -> None:
        forbidden_commands = [
            "rm -rf .",
            "unit && bad",
            "python -m unittest",
        ]

        with mock.patch("tools.subprocess.run") as mocked_run:
            for command_name in forbidden_commands:
                with self.subTest(command_name=command_name):
                    with self.assertRaisesRegex(ToolError, "command_name"):
                        self.repository_tools.run_tests(command_name)
                    mocked_run.assert_not_called()

            mocked_run.reset_mock()

    def test_run_tests_timeout_normalizes_and_truncates_output(self) -> None:
        timeout_output = b"abc" * 10
        timeout_error = b"err" * 10
        with (
            mock.patch("tools.subprocess.run") as mocked_run,
            mock.patch("tools.RUN_TEST_OUTPUT_MAX_BYTES", new=12),
        ):
            mocked_run.side_effect = subprocess.TimeoutExpired(
                cmd=[sys.executable, "-m", "unittest", "discover"],
                timeout=tools.RUN_TEST_TIMEOUT_SECONDS,
                output=timeout_output,
                stderr=timeout_error,
            )

            result = self.repository_tools.run_tests("unit")

            self.assertEqual("unit", result["command_name"])
            self.assertEqual(-1, result["exit_code"])
            self.assertTrue(result["timed_out"])
            self.assertEqual("abcabcabcabc", result["stdout"])
            self.assertEqual("errerrerrerr", result["stderr"])

            events = self._latest_run_events()
            self.assertTrue(events)
            event_types = {event.get("event_type", "") for event in events}
            self.assertIn("run_tests_start", event_types)
            self.assertIn("run_tests_timeout", event_types)
            self.assertIn("run_tests_end", event_types)
            self.assertIn("run_tests_truncation", event_types)


class TestProposePatchV03(unittest.TestCase):
    """验证 v0.3 提案补丁保存行为。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def test_propose_patch_v03_saves_valid_diff_and_metadata(self) -> None:
        self.temp_repo_helper.create_text_file("file.txt", "old line\n")
        original_target = self.temp_repo_helper.read_text_file("file.txt")
        instruction = "更新文件内容"
        diff_text = """diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old line\n+new line\n"""

        patch_result = self.repository_tools.propose_patch(
            instruction=instruction,
            diff=diff_text,
        )

        self.assertTrue(patch_result["ok"])
        self.assertIsInstance(patch_result["patch_id"], str)
        self.assertTrue(patch_result["patch_id"])
        self.assertNotEqual("", patch_result["patch_path"])
        patch_path = self.temp_repo_helper.repo_root / str(patch_result["patch_path"])
        metadata_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / f"{patch_result['patch_id']}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(diff_text, patch_path.read_text(encoding="utf-8"))
        self.assertEqual("proposed", metadata["status"])
        self.assertEqual(instruction, metadata["instruction"])
        self.assertEqual(["file.txt"], metadata["paths"])
        self.assertIn("需人工确认", "".join(metadata["warnings"]))
        self.assertIn("patch_id", patch_result)
        self.assertEqual([], patch_result["errors"])
        self.assertEqual(["file.txt"], patch_result["paths"])
        self.assertEqual(diff_text, patch_result["diff_preview"])
        self.assertEqual(original_target, self.temp_repo_helper.read_text_file("file.txt"))

    def test_propose_patch_v03_truncates_diff_preview(self) -> None:
        file_lines = [f"line {index + 1}" for index in range(130)]
        added_lines = "\n".join([f"+{line}" for line in file_lines])
        diff_text = (
            "diff --git a/preview.txt b/preview.txt\n"
            "--- /dev/null\n"
            "+++ b/preview.txt\n"
            "@@ -0,0 +1,130 @@\n"
            f"{added_lines}\n"
        )

        patch_result = self.repository_tools.propose_patch(
            instruction="预览截断",
            diff=diff_text,
        )

        self.assertTrue(patch_result["ok"])
        diff_preview = str(patch_result["diff_preview"])
        self.assertIn("...（预览已截断）", diff_preview)
        self.assertLessEqual(len(diff_preview.splitlines()), 121)

    def test_propose_patch_v03_rejects_invalid_diff_without_writing(self) -> None:
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches"
        existing_patch_files = list(patch_dir.glob("*.patch"))

        patch_result = self.repository_tools.propose_patch(
            instruction="无效补丁",
            diff="invalid diff content",
        )

        self.assertFalse(patch_result["ok"])
        self.assertTrue(patch_result["errors"])
        self.assertEqual(existing_patch_files, list(patch_dir.glob("*.patch")))

    def test_propose_patch_v03_rejects_hunk_body_count_mismatch_without_writing(self) -> None:
        self.temp_repo_helper.create_text_file("count_mismatch.txt", "one\ntwo\nthree\n")
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches"
        existing_patch_files = list(patch_dir.glob("*.patch"))
        existing_metadata_files = list(patch_dir.glob("*.json"))
        diff_text = (
            "diff --git a/count_mismatch.txt b/count_mismatch.txt\n"
            "--- a/count_mismatch.txt\n"
            "+++ b/count_mismatch.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-one\n"
            "-two\n"
            "+changed\n"
        )

        patch_result = self.repository_tools.propose_patch(
            instruction="reject malformed hunk count",
            diff=diff_text,
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["patch_id"])
        self.assertEqual("", patch_result["patch_path"])
        errors_object = patch_result["errors"]
        self.assertIsInstance(errors_object, list)
        if not isinstance(errors_object, list):
            self.fail("errors should be a list")
        self.assertTrue(
            any(
                isinstance(error, str) and "hunk body count mismatch" in error
                for error in errors_object
            )
        )
        self.assertEqual(existing_patch_files, list(patch_dir.glob("*.patch")))
        self.assertEqual(existing_metadata_files, list(patch_dir.glob("*.json")))
        self.assertEqual("one\ntwo\nthree\n", self.temp_repo_helper.read_text_file("count_mismatch.txt"))

    def test_propose_patch_v03_rejects_empty_inputs(self) -> None:
        invalid_inputs = [
            {"instruction": "", "diff": "diff --git a/x b/x\n"},
            {"instruction": "修复", "diff": ""},
            {"instruction": "修复", "diff": "   \n"},
        ]

        for invalid_input in invalid_inputs:
            patch_result = self.repository_tools.propose_patch(
                instruction=invalid_input["instruction"],
                diff=invalid_input["diff"],
            )

            self.assertFalse(patch_result["ok"])
            self.assertTrue(patch_result["errors"])

    def test_propose_patch_v03_appends_run_event(self) -> None:
        self.temp_repo_helper.create_text_file("run.txt", "old\n")
        diff_text = """diff --git a/run.txt b/run.txt\n--- a/run.txt\n+++ b/run.txt\n@@ -1 +1 @@\n-old\n+new\n"""

        patch_result = self.repository_tools.propose_patch(
            instruction="记录运行事件",
            diff=diff_text,
        )

        self.assertTrue(patch_result["ok"])
        run_log = self.temp_repo_helper.repo_root / ".repopilot/runs" / f"{patch_result['patch_id']}.jsonl"
        self.assertTrue(run_log.exists())
        event = json.loads(run_log.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual("proposed", event["event_type"])
        self.assertEqual("saved", event["status"])
        self.assertEqual(patch_result["patch_id"], event["patch_id"])

    def test_apply_patch_is_registered_and_dispatch_requires_patch_id(self) -> None:
        self.assertIn("apply_patch", self.repository_tools.tools)

        with self.assertRaises(ToolError):
            self.repository_tools.run_tool({"tool": "apply_patch", "args": {}})


class TestApplyPatchV03(unittest.TestCase):
    """Verify v0.3 saved patch application, backup, rejection, and rollback."""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def _propose_patch(self, diff_text: str, instruction: str = "apply patch") -> str:
        patch_result = self.repository_tools.propose_patch(
            instruction=instruction,
            diff=diff_text,
        )
        self.assertTrue(patch_result["ok"], patch_result)
        return str(patch_result["patch_id"])

    def _metadata(self, patch_id: str) -> dict[str, object]:
        metadata_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / f"{patch_id}.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _run_events(self, patch_id: str) -> list[dict[str, object]]:
        run_path = self.temp_repo_helper.repo_root / ".repopilot/runs" / f"{patch_id}.jsonl"
        return [
            json.loads(line)
            for line in run_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_apply_patch_success_updates_files_backup_and_metadata(self) -> None:
        self.temp_repo_helper.create_text_file("src/existing.txt", "old line\nkeep\n")
        diff_text = (
            "diff --git a/src/existing.txt b/src/existing.txt\n"
            "--- a/src/existing.txt\n"
            "+++ b/src/existing.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            " keep\n"
            "diff --git a/src/created.txt b/src/created.txt\n"
            "--- /dev/null\n"
            "+++ b/src/created.txt\n"
            "@@ -0,0 +1,2 @@\n"
            "+created one\n"
            "+created two\n"
        )
        patch_id = self._propose_patch(diff_text)

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertTrue(apply_result["ok"], apply_result)
        self.assertEqual("applied", apply_result["status"])
        self.assertEqual(["src/existing.txt", "src/created.txt"], apply_result["modified_files"])
        self.assertEqual(["src/created.txt"], apply_result["new_files"])
        self.assertEqual("new line\nkeep\n", self.temp_repo_helper.read_text_file("src/existing.txt"))
        self.assertEqual("created one\ncreated two\n", self.temp_repo_helper.read_text_file("src/created.txt"))
        backup_path = self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id / "src/existing.txt"
        self.assertEqual("old line\nkeep\n", backup_path.read_text(encoding="utf-8"))
        metadata = self._metadata(patch_id)
        self.assertEqual("applied", metadata["status"])
        self.assertIn("applied_at", metadata)
        self.assertEqual(["src/existing.txt", "src/created.txt"], metadata["modified_files"])
        event_types = {event["event_type"] for event in self._run_events(patch_id)}
        self.assertIn("apply_start", event_types)
        self.assertIn("apply_success", event_types)

    def test_apply_patch_rejects_already_applied_patch_before_mutation(self) -> None:
        self.temp_repo_helper.create_text_file("repeat.txt", "old\n")
        diff_text = "diff --git a/repeat.txt b/repeat.txt\n--- a/repeat.txt\n+++ b/repeat.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        first_result = self.repository_tools.apply_patch(patch_id)
        self.assertTrue(first_result["ok"])

        second_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(second_result["ok"])
        self.assertEqual("rejected", second_result["status"])
        self.assertEqual(["patch is already applied"], second_result["errors"])
        self.assertEqual("new\n", self.temp_repo_helper.read_text_file("repeat.txt"))

    def test_apply_patch_rejects_unknown_malformed_and_non_proposed_patch(self) -> None:
        self.temp_repo_helper.create_text_file("state.txt", "old\n")
        malformed_result = self.repository_tools.apply_patch("not-a-patch-id")
        self.assertFalse(malformed_result["ok"])

        unknown_patch_id = self.repository_tools._generate_patch_id("unknown patch")
        unknown_result = self.repository_tools.apply_patch(unknown_patch_id)
        self.assertFalse(unknown_result["ok"])

        diff_text = "diff --git a/state.txt b/state.txt\n--- a/state.txt\n+++ b/state.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        metadata = self._metadata(patch_id)
        metadata["status"] = "accepted"
        self.repository_tools._write_patch_metadata(metadata)
        non_proposed_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(non_proposed_result["ok"])
        self.assertEqual(["only status=proposed patches can be applied"], non_proposed_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("state.txt"))
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_apply_patch_revalidates_saved_diff_and_rejects_raw_diff_args(self) -> None:
        self.temp_repo_helper.create_text_file("safe.txt", "old\n")
        diff_text = "diff --git a/safe.txt b/safe.txt\n--- a/safe.txt\n+++ b/safe.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        patch_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / f"{patch_id}.patch"
        patch_path.write_text("not a unified diff\n", encoding="utf-8")

        invalid_saved_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(invalid_saved_result["ok"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("safe.txt"))
        with self.assertRaises(ToolError):
            self.repository_tools.run_tool(
                {"tool": "apply_patch", "args": {"patch_id": patch_id, "diff": diff_text}}
            )

    def test_apply_patch_reports_hunk_content_mismatch_clearly(self) -> None:
        self.temp_repo_helper.create_text_file("mismatch.txt", "current\n")
        diff_text = "diff --git a/mismatch.txt b/mismatch.txt\n--- a/mismatch.txt\n+++ b/mismatch.txt\n@@ -1 +1 @@\n-current\n+patched\n"
        patch_id = self._propose_patch(diff_text)
        self.temp_repo_helper.create_text_file("mismatch.txt", "changed before apply\n")

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("rejected", apply_result["status"])
        self.assertEqual(["hunk content mismatch: mismatch.txt line 1"], apply_result["errors"])
        self.assertEqual("changed before apply\n", self.temp_repo_helper.read_text_file("mismatch.txt"))

    def test_apply_patch_rolls_back_existing_and_new_files_after_partial_write_failure(self) -> None:
        self.temp_repo_helper.create_text_file("first.txt", "old first\n")
        self.temp_repo_helper.create_text_file("second.txt", "old second\n")
        diff_text = (
            "diff --git a/first.txt b/first.txt\n"
            "--- a/first.txt\n"
            "+++ b/first.txt\n"
            "@@ -1 +1 @@\n"
            "-old first\n"
            "+new first\n"
            "diff --git a/created_after_failure.txt b/created_after_failure.txt\n"
            "--- /dev/null\n"
            "+++ b/created_after_failure.txt\n"
            "@@ -0,0 +1 @@\n"
            "+created\n"
            "diff --git a/second.txt b/second.txt\n"
            "--- a/second.txt\n"
            "+++ b/second.txt\n"
            "@@ -1 +1 @@\n"
            "-old second\n"
            "+new second\n"
        )
        patch_id = self._propose_patch(diff_text)
        original_write_text = Path.write_text

        def fail_on_second_write_text(
            target_path: Path,
            content: str,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            if target_path.name == "second.txt":
                raise OSError("simulated write failure")
            return original_write_text(
                target_path,
                content,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )

        with mock.patch.object(Path, "write_text", new=fail_on_second_write_text):
            apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("failed", apply_result["status"])
        self.assertEqual({"attempted": True, "ok": True, "errors": []}, apply_result["rollback"])
        self.assertEqual("old first\n", self.temp_repo_helper.read_text_file("first.txt"))
        self.assertEqual("old second\n", self.temp_repo_helper.read_text_file("second.txt"))
        self.assertFalse((self.temp_repo_helper.repo_root / "created_after_failure.txt").exists())
        run_events = self._run_events(patch_id)
        event_types = {event["event_type"] for event in run_events}
        self.assertIn("apply_failure", event_types)
        self.assertIn("rollback_success", event_types)
        apply_failure_found = False
        for event in run_events:
            details = event.get("details")
            if (
                event.get("event_type") == "apply_failure"
                and event.get("status") == "failed"
                and isinstance(details, dict)
                and details.get("rollback") == {"attempted": True, "ok": True, "errors": []}
            ):
                apply_failure_found = True
        self.assertTrue(apply_failure_found)
        self.assertTrue(
            any(
                event["event_type"] == "rollback_success"
                and event["status"] == "ok"
                and event["details"] == {"errors": []}
                for event in run_events
            )
        )


class TestRepositoryToolsV03Storage(unittest.TestCase):
    """验证 .repopilot 存储目录和 patch 元数据 helper。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def test_init_creates_repopilot_storage_dirs(self) -> None:
        repopilot_root = self.temp_repo_helper.repo_root / ".repopilot"

        self.assertTrue((repopilot_root / "patches").is_dir())
        self.assertTrue((repopilot_root / "backups").is_dir())
        self.assertTrue((repopilot_root / "runs").is_dir())
        self.assertIn("list_dir", self.repository_tools.tools)
        self.assertIn("read_file", self.repository_tools.tools)
        self.assertIn("search_text", self.repository_tools.tools)
        self.assertIn("finish", self.repository_tools.tools)

    def test_write_and_read_patch_file_and_metadata(self) -> None:
        patch_text = "diff --git a/example.txt b/example.txt\n"
        patch_id = self.repository_tools._generate_patch_id(
            instruction="更新 example.txt",
            patch_text=patch_text,
        )
        metadata = self.repository_tools._create_patch_metadata(
            patch_id=patch_id,
            instruction="更新 example.txt",
            paths=["example.txt"],
            warnings=["人工确认后应用"],
        )

        patch_path = self.repository_tools._write_patch_file(patch_id, patch_text)
        metadata_path = self.repository_tools._write_patch_metadata(metadata)

        self.assertRegex(patch_id, r"^\d{8}_\d{6}_[0-9a-f]{12}$")
        self.assertEqual(patch_path.name, f"{patch_id}.patch")
        self.assertEqual(metadata_path.name, f"{patch_id}.json")
        self.assertEqual(patch_text, self.repository_tools._read_patch_file(patch_id))
        self.assertEqual(metadata, self.repository_tools._read_patch_metadata(patch_id))
        self.assertEqual(patch_id, metadata["patch_id"])
        self.assertEqual("更新 example.txt", metadata["instruction"])
        self.assertEqual("proposed", metadata["status"])
        self.assertEqual(["example.txt"], metadata["paths"])
        self.assertEqual(["人工确认后应用"], metadata["warnings"])
        self.assertIn("created_at", metadata)

    def test_update_metadata_status_and_append_run_events(self) -> None:
        patch_id = self.repository_tools._generate_patch_id("记录运行事件")
        metadata = self.repository_tools._create_patch_metadata(
            patch_id=patch_id,
            instruction="记录运行事件",
            paths=["example.txt"],
            warnings=[],
        )
        self.repository_tools._write_patch_metadata(metadata)

        updated_metadata = self.repository_tools._update_patch_metadata_status(
            patch_id,
            "accepted",
        )
        first_run_path = self.repository_tools._append_run_event(
            patch_id=patch_id,
            event_type="metadata_status",
            status="accepted",
            details={"path_count": 1},
        )
        second_run_path = self.repository_tools._append_run_event(
            patch_id=patch_id,
            event_type="review",
            status="completed",
        )

        self.assertEqual("accepted", updated_metadata["status"])
        self.assertIn("updated_at", updated_metadata)
        self.assertEqual(first_run_path, second_run_path)
        event_lines = first_run_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(2, len(event_lines))
        first_event = json.loads(event_lines[0])
        second_event = json.loads(event_lines[1])
        self.assertEqual(patch_id, first_event["patch_id"])
        self.assertEqual("metadata_status", first_event["event_type"])
        self.assertEqual("accepted", first_event["status"])
        self.assertEqual({"path_count": 1}, first_event["details"])
        self.assertEqual("review", second_event["event_type"])
        self.assertEqual("completed", second_event["status"])


class TestReadFileRange(unittest.TestCase):
    """验证 read_file_range 的行范围读取和安全限制。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repo_root = self.temp_repo_helper.repo_root
        self.temp_repo_helper.create_text_file(
            "agent.py",
            "\n".join(
                [
                    '"""LLM + 工具调用循环。"""',
                    "",
                    "from __future__ import annotations",
                    "",
                    "import json",
                    "from pathlib import Path",
                    "from config import MAX_STEPS",
                ]
            )
            + "\n",
        )
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def create_file(self, relative_path: str, content: str | bytes) -> Path:
        if isinstance(content, bytes):
            return self.temp_repo_helper.create_binary_file(relative_path, content)
        return self.temp_repo_helper.create_text_file(relative_path, content)

    def test_read_file_range_is_registered_and_described(self) -> None:
        self.assertIn("read_file_range", self.repository_tools.tools)

        tool_descriptions = self.repository_tools.tool_descriptions()

        self.assertIn("read_file_range", tool_descriptions)
        self.assertIn("start_line", tool_descriptions)
        self.assertIn("end_line", tool_descriptions)

    def test_read_file_range_returns_exact_inclusive_range(self) -> None:
        file_content = self.repository_tools.read_file_range("agent.py", 1, 3)

        self.assertEqual(
            "\n".join(
                [
                    '1 | """LLM + 工具调用循环。"""',
                    "2 | ",
                    "3 | from __future__ import annotations",
                ]
            ),
            file_content,
        )

    def test_read_file_range_returns_single_full_line(self) -> None:
        file_content = self.repository_tools.read_file_range("agent.py", 7, 7)

        self.assertEqual("7 | from config import MAX_STEPS", file_content)

    def test_read_file_range_start_beyond_eof_returns_empty_string(self) -> None:
        file_content = self.repository_tools.read_file_range("agent.py", 10_000, 10_001)

        self.assertEqual("", file_content)

    def test_read_file_range_rejects_invalid_ranges(self) -> None:
        invalid_calls = [
            ("agent.py", 0, 1, "start_line"),
            ("agent.py", 3, 2, "end_line"),
            ("agent.py", "1", 2, "start_line"),
            ("agent.py", 1, "2", "end_line"),
            ("agent.py", True, 2, "start_line"),
            ("agent.py", 1, False, "end_line"),
        ]

        for path, start_line, end_line, expected_error in invalid_calls:
            with self.subTest(start_line=start_line, end_line=end_line):
                with self.assertRaisesRegex(ToolError, expected_error):
                    self.repository_tools.read_file_range(path, start_line, end_line)

    def test_read_file_range_rejects_parent_directory_escape(self) -> None:
        with self.assertRaisesRegex(ToolError, "repo 目录内部"):
            self.repository_tools.read_file_range("../outside.txt", 1, 1)

    def test_read_file_range_rejects_hidden_path(self) -> None:
        self.create_file(".range_hidden.txt", "hidden\n")

        with self.assertRaisesRegex(ToolError, "隐藏路径"):
            self.repository_tools.read_file_range(".range_hidden.txt", 1, 1)

    def test_read_file_range_rejects_sensitive_file(self) -> None:
        self.create_file("range_secret.pem", "SECRET\n")

        with self.assertRaisesRegex(ToolError, "密钥或环境文件"):
            self.repository_tools.read_file_range("range_secret.pem", 1, 1)

    def test_read_file_range_rejects_binary_file(self) -> None:
        self.create_file("range_binary.bin", b"abc\x00def")

        with self.assertRaisesRegex(ToolError, "二进制文件"):
            self.repository_tools.read_file_range("range_binary.bin", 1, 1)

    def test_read_file_range_rejects_invalid_utf8_file(self) -> None:
        self.create_file("range_invalid_utf8.txt", b"\xff\xfe")

        with self.assertRaisesRegex(ToolError, "UTF-8"):
            self.repository_tools.read_file_range("range_invalid_utf8.txt", 1, 1)

    def test_read_file_range_rejects_oversized_file(self) -> None:
        self.create_file("range_oversized.txt", "a" * (MAX_FILE_BYTES + 1))

        with self.assertRaisesRegex(ToolError, "20KB"):
            self.repository_tools.read_file_range("range_oversized.txt", 1, 1)


class TestUnifiedDiffValidation(unittest.TestCase):
    """验证 unified diff 严格校验和路径安全限制。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repo_root = self.temp_repo_helper.repo_root
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def create_text_file(self, relative_path: str, content: str) -> Path:
        return self.temp_repo_helper.create_text_file(relative_path, content)

    def assert_diff_rejected(self, diff_text: str, expected_error_text: str) -> None:
        with self.assertRaisesRegex(ToolError, expected_error_text):
            self.repository_tools.validate_unified_diff(diff_text)

    def test_valid_existing_text_file_modification_returns_touched_path(self) -> None:
        self.create_text_file("diff_existing.txt", "old line\n")
        diff_text = """diff --git a/diff_existing.txt b/diff_existing.txt\n--- a/diff_existing.txt\n+++ b/diff_existing.txt\n@@ -1 +1 @@\n-old line\n+new line\n"""

        touched_paths = self.repository_tools.validate_unified_diff(diff_text)

        self.assertEqual(["diff_existing.txt"], touched_paths)

    def test_valid_new_text_file_creation_returns_touched_path(self) -> None:
        target_path = self.repo_root / "diff_new_file.txt"
        if target_path.exists():
            target_path.unlink()
        diff_text = """diff --git a/diff_new_file.txt b/diff_new_file.txt\n--- /dev/null\n+++ b/diff_new_file.txt\n@@ -0,0 +1 @@\n+created line\n"""

        touched_paths = self.repository_tools.validate_unified_diff(diff_text)

        self.assertEqual(["diff_new_file.txt"], touched_paths)

    def test_rejects_absolute_diff_git_path(self) -> None:
        diff_text = """diff --git E:/outside.txt b/outside.txt\n--- a/outside.txt\n+++ b/outside.txt\n@@ -1 +1 @@\n-old\n+new\n"""

        self.assert_diff_rejected(diff_text, "a/")

    def test_rejects_parent_directory_escape(self) -> None:
        diff_text = """diff --git a/../outside.txt b/../outside.txt\n--- a/../outside.txt\n+++ b/../outside.txt\n@@ -1 +1 @@\n-old\n+new\n"""

        self.assert_diff_rejected(diff_text, r"\.\.")

    def test_rejects_hidden_and_secret_targets(self) -> None:
        hidden_diff = """diff --git a/.hidden.txt b/.hidden.txt\n--- a/.hidden.txt\n+++ b/.hidden.txt\n@@ -1 +1 @@\n-old\n+new\n"""
        secret_diff = """diff --git a/config.pem b/config.pem\n--- /dev/null\n+++ b/config.pem\n@@ -0,0 +1 @@\n+secret\n"""

        self.assert_diff_rejected(hidden_diff, "隐藏路径")
        self.assert_diff_rejected(secret_diff, "密钥或环境文件")

    def test_rejects_git_target(self) -> None:
        diff_text = """diff --git a/.git/config b/.git/config\n--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-old\n+new\n"""

        self.assert_diff_rejected(diff_text, r"隐藏路径|\.git")

    def test_rejects_deletion(self) -> None:
        self.create_text_file("delete_me.txt", "old line\n")
        diff_text = """diff --git a/delete_me.txt b/delete_me.txt\n--- a/delete_me.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-old line\n"""

        self.assert_diff_rejected(diff_text, "删除")

    def test_rejects_rename_copy_and_binary_markers(self) -> None:
        rename_diff = """diff --git a/old.txt b/new.txt\nrename from old.txt\nrename to new.txt\n"""
        copy_diff = """diff --git a/old.txt b/new.txt\ncopy from old.txt\ncopy to new.txt\n"""
        binary_diff = """diff --git a/image.bin b/image.bin\nBinary files a/image.bin and b/image.bin differ\n"""
        git_binary_diff = """diff --git a/image.bin b/image.bin\nGIT binary patch\nliteral 0\n"""

        self.assert_diff_rejected(rename_diff, "rename|copy|跨路径")
        self.assert_diff_rejected(copy_diff, "rename|copy|跨路径")
        self.assert_diff_rejected(binary_diff, "二进制")
        self.assert_diff_rejected(git_binary_diff, "二进制")

    def test_rejects_malformed_hunk_headers_and_lines(self) -> None:
        self.create_text_file("bad_hunk.txt", "old line\n")
        bad_header_diff = """diff --git a/bad_hunk.txt b/bad_hunk.txt\n--- a/bad_hunk.txt\n+++ b/bad_hunk.txt\n@@ broken @@\n-old line\n+new line\n"""
        bad_body_diff = """diff --git a/bad_hunk.txt b/bad_hunk.txt\n--- a/bad_hunk.txt\n+++ b/bad_hunk.txt\n@@ -1 +1 @@\n?bad line\n"""

        self.assert_diff_rejected(bad_header_diff, "hunk")
        self.assert_diff_rejected(bad_body_diff, "hunk 内容")

    def test_rejects_existing_binary_target_file(self) -> None:
        self.temp_repo_helper.create_binary_file("binary_diff_target.bin", b"abc\x00def")
        diff_text = """diff --git a/binary_diff_target.bin b/binary_diff_target.bin\n--- a/binary_diff_target.bin\n+++ b/binary_diff_target.bin\n@@ -1 +1 @@\n-abc\n+def\n"""

        self.assert_diff_rejected(diff_text, "二进制")

    def test_rejects_symlink_traversal_when_supported(self) -> None:
        target_dir = self.repo_root / "symlink_outside"
        link_path = self.repo_root / "diff_symlink"
        try:
            target_dir.mkdir()
            link_path.symlink_to(target_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("当前环境不支持创建符号链接")

        diff_text = """diff --git a/diff_symlink/file.txt b/diff_symlink/file.txt\n--- /dev/null\n+++ b/diff_symlink/file.txt\n@@ -0,0 +1 @@\n+created line\n"""

        self.assert_diff_rejected(diff_text, "符号链接")


if __name__ == "__main__":
    unittest.main()
