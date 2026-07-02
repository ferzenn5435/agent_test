"""RepositoryTools 基础单元测试。"""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
import sys
import tools
from unittest import mock
from pathlib import Path
from config import (
    MAX_FILE_BYTES,
    MAX_FULL_READ_BYTES,
    MAX_FULL_READ_LINES,
    MAX_RANGE_READ_LINES,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_OUTPUT_CHARS,
)
from context_stats import ContextStats
from tools import RepositoryTools, ToolError


class TestContextConstantsV05(unittest.TestCase):
    """验证 v0.5 上下文管理常量。"""

    def test_context_limit_constants_match_v05_schema(self) -> None:
        self.assertEqual(300, MAX_FULL_READ_LINES)
        self.assertEqual(20_000, MAX_FULL_READ_BYTES)
        self.assertEqual(120, MAX_RANGE_READ_LINES)
        self.assertEqual(12_000, MAX_TOOL_OUTPUT_CHARS)
        self.assertEqual(20, MAX_SEARCH_RESULTS)


class TestContextStatsV05(unittest.TestCase):
    """验证 v0.5 ContextStats 的稳定序列化 schema。"""

    expected_schema_keys = [
        "steps_used",
        "total_tool_output_chars",
        "messages_total_chars",
        "files_read",
        "ranges_read",
        "search_calls",
        "full_file_reads",
    ]

    def test_to_dict_returns_fixed_json_serializable_schema(self) -> None:
        context_stats = ContextStats(
            steps_used=3,
            total_tool_output_chars=456,
            messages_total_chars=789,
            files_read=("src\\agent.py", "README.md"),
            ranges_read=(
                {"path": "src\\tools.py", "start_line": 10, "end_line": 20},
            ),
            search_calls=2,
            full_file_reads=("tests\\test_tools.py",),
        )

        stats_dict = context_stats.to_dict()

        self.assertEqual(self.expected_schema_keys, list(stats_dict.keys()))
        self.assertEqual(3, stats_dict["steps_used"])
        self.assertEqual(456, stats_dict["total_tool_output_chars"])
        self.assertEqual(789, stats_dict["messages_total_chars"])
        self.assertEqual(["src/agent.py", "README.md"], stats_dict["files_read"])
        self.assertEqual(
            [{"path": "src/tools.py", "start_line": 10, "end_line": 20}],
            stats_dict["ranges_read"],
        )
        self.assertEqual(2, stats_dict["search_calls"])
        self.assertEqual(["tests/test_tools.py"], stats_dict["full_file_reads"])
        json.dumps(stats_dict)

    def test_default_constructor_to_dict_returns_empty_schema(self) -> None:
        context_stats = ContextStats()

        stats_dict = context_stats.to_dict()

        self.assertEqual(self.expected_schema_keys, list(stats_dict.keys()))
        self.assertEqual(0, stats_dict["steps_used"])
        self.assertEqual(0, stats_dict["total_tool_output_chars"])
        self.assertEqual(0, stats_dict["messages_total_chars"])
        self.assertEqual([], stats_dict["files_read"])
        self.assertEqual([], stats_dict["ranges_read"])
        self.assertEqual(0, stats_dict["search_calls"])
        self.assertEqual([], stats_dict["full_file_reads"])
        json.dumps(stats_dict)


class TestReadSearchContextV05(unittest.TestCase):
    """验证 v0.5 读取和搜索上下文控制。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def test_read_file_rejects_large_file(self) -> None:
        self.temp_repo_helper.create_text_file(
            "large_bytes.txt",
            "a" * (MAX_FULL_READ_BYTES + 1),
        )

        with self.assertRaisesRegex(ToolError, "read_file_range") as error_context:
            self.repository_tools.read_file("large_bytes.txt")

        error_message = str(error_context.exception)
        self.assertIn(str(MAX_FULL_READ_BYTES), error_message)
        self.assertIn(str(MAX_FULL_READ_LINES), error_message)

    def test_read_file_rejects_too_many_lines(self) -> None:
        file_text = "".join("x\n" for _ in range(MAX_FULL_READ_LINES + 1))
        self.temp_repo_helper.create_text_file("many_lines.txt", file_text)

        with self.assertRaisesRegex(ToolError, "read_file_range") as error_context:
            self.repository_tools.read_file("many_lines.txt")

        error_message = str(error_context.exception)
        self.assertIn(str(MAX_FULL_READ_LINES), error_message)
        self.assertIn(str(MAX_FULL_READ_BYTES), error_message)

    def test_read_file_keeps_numbered_format_for_small_file(self) -> None:
        self.temp_repo_helper.create_text_file("small.txt", "alpha\nbeta\n")

        file_content = self.repository_tools.read_file("small.txt")

        self.assertEqual("1 | alpha\n2 | beta", file_content)

    def test_read_file_range_reads_large_file_with_original_line_numbers(self) -> None:
        large_lines = [f"line {line_number}" for line_number in range(1, MAX_FULL_READ_LINES + 20)]
        self.temp_repo_helper.create_text_file("large_range.txt", "\n".join(large_lines) + "\n")

        file_content = self.repository_tools.read_file_range("large_range.txt", 301, 303)

        self.assertEqual(
            "\n".join([
                "301 | line 301",
                "302 | line 302",
                "303 | line 303",
            ]),
            file_content,
        )

    def test_read_file_range_rejects_invalid_range_and_beyond_eof(self) -> None:
        self.temp_repo_helper.create_text_file("range.txt", "one\ntwo\nthree\n")

        invalid_calls = [
            (0, 1, "start_line"),
            (3, 2, "end_line"),
            (1, MAX_RANGE_READ_LINES + 1, str(MAX_RANGE_READ_LINES)),
            (1, 4, "超过文件总行数"),
        ]
        for start_line, end_line, expected_error in invalid_calls:
            with self.subTest(start_line=start_line, end_line=end_line):
                with self.assertRaisesRegex(ToolError, expected_error):
                    self.repository_tools.read_file_range("range.txt", start_line, end_line)

    def test_search_text_path_glob_and_max_results(self) -> None:
        self.temp_repo_helper.create_text_file("src/one.py", "needle one\n")
        self.temp_repo_helper.create_text_file("src/two.py", "needle two\n")
        self.temp_repo_helper.create_text_file("docs/ignored.md", "needle ignored\n")

        matches = self.repository_tools.search_text(
            "needle",
            path_glob="src/*.py",
            max_results=1,
            context_lines=0,
        )

        self.assertEqual(1, len(matches))
        self.assertIn("src/one.py:1", matches[0])
        self.assertIn("1 | needle one", matches[0])
        self.assertNotIn("docs/ignored.md", "\n".join(matches))

    def test_search_text_context_lines(self) -> None:
        self.temp_repo_helper.create_text_file(
            "context.txt",
            "before far\nbefore near\nneedle hit\nafter near\nafter far\n",
        )

        matches = self.repository_tools.search_text("needle", context_lines=1)

        self.assertEqual(1, len(matches))
        self.assertIn("2 | before near", matches[0])
        self.assertIn("3 | needle hit", matches[0])
        self.assertIn("4 | after near", matches[0])
        self.assertNotIn("1 | before far", matches[0])
        self.assertNotIn("5 | after far", matches[0])

    def test_search_text_output_truncation_marker(self) -> None:
        self.temp_repo_helper.create_text_file("long.txt", "needle " + "x" * 200 + "\n")

        with mock.patch("tools.MAX_TOOL_OUTPUT_CHARS", new=30):
            matches = self.repository_tools.search_text("needle", context_lines=0)

        self.assertEqual(["...（结果已截断）"], matches)

    def test_search_text_rejects_invalid_controls(self) -> None:
        invalid_calls = [
            {"max_results": 0},
            {"max_results": MAX_SEARCH_RESULTS + 1},
            {"context_lines": -1},
            {"path_glob": "../*.py"},
        ]

        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ToolError):
                    self.repository_tools.search_text("needle", **kwargs)


class TestRepoIndexV05(unittest.TestCase):
    """验证 v0.5 项目文件索引工具。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def _read_index(self, index_path: str) -> dict[str, object]:
        """辅助方法：读取索引 JSON 文件。"""
        full_index_path = self.temp_repo_helper.repo_root / index_path
        return json.loads(full_index_path.read_text(encoding="utf-8"))

    def _file_records_by_path(self, repo_index: dict[str, object]) -> dict[str, dict[str, object]]:
        """将索引文件列表转换为 path -> record 字典。"""
        files = repo_index["files"]
        self.assertIsInstance(files, list)
        if not isinstance(files, list):
            self.fail("files should be a list")
        return {
            str(file_record["path"]): file_record
            for file_record in files
            if isinstance(file_record, dict)
        }

    def test_build_repo_index_extracts_symbols(self) -> None:
        self.temp_repo_helper.create_text_file(
            "pkg/example.py",
            "class Service:\n"
            "    def run(self):\n"
            "        return 1\n"
            "    async def arun(self):\n"
            "        return 2\n"
            "\n"
            "def helper():\n"
            "    return 3\n"
            "\n"
            "async def async_helper():\n"
            "    return 4\n",
        )

        build_result = self.repository_tools.build_repo_index(force=True)
        repo_index = self._read_index(str(build_result["index_path"]))
        records_by_path = self._file_records_by_path(repo_index)
        example_record = records_by_path["pkg/example.py"]

        self.assertIn("build_repo_index", self.repository_tools.tools)
        self.assertEqual("built", build_result["status"])
        self.assertEqual(".py", example_record["extension"])
        self.assertEqual(11, example_record["line_count"])
        self.assertEqual(
            [
                {"name": "Service", "kind": "class", "line": 1},
                {"name": "run", "kind": "method", "line": 2},
                {"name": "arun", "kind": "async_method", "line": 4},
                {"name": "helper", "kind": "function", "line": 7},
                {"name": "async_helper", "kind": "async_function", "line": 10},
            ],
            example_record["symbols"],
        )

    def test_build_repo_index_skips_excluded_dirs(self) -> None:
        self.temp_repo_helper.create_text_file("src/kept.py", "def kept():\n    return True\n")
        for skipped_dir_name in tools.REPO_INDEX_SKIPPED_DIR_NAMES:
            self.temp_repo_helper.create_text_file(
                f"{skipped_dir_name}/ignored.py",
                "def ignored():\n    return False\n",
            )

        build_result = self.repository_tools.build_repo_index(force=True)
        repo_index = self._read_index(str(build_result["index_path"]))
        records_by_path = self._file_records_by_path(repo_index)

        self.assertIn("src/kept.py", records_by_path)
        for skipped_dir_name in tools.REPO_INDEX_SKIPPED_DIR_NAMES:
            self.assertNotIn(f"{skipped_dir_name}/ignored.py", records_by_path)
        self.assertTrue(
            all(not path.startswith(".repopilot/index/") for path in records_by_path)
        )

    def test_build_repo_index_records_syntax_error_file(self) -> None:
        self.temp_repo_helper.create_text_file("broken.py", "def broken(:\n    pass\n")

        build_result = self.repository_tools.build_repo_index(force=True)
        repo_index = self._read_index(str(build_result["index_path"]))
        records_by_path = self._file_records_by_path(repo_index)
        broken_record = records_by_path["broken.py"]

        self.assertEqual([], broken_record["symbols"])
        self.assertIn("syntax error", str(broken_record["error"]))

    def test_build_repo_index_reuses_cache_when_force_false(self) -> None:
        self.temp_repo_helper.create_text_file("src/original.py", "def original():\n    return 1\n")
        first_result = self.repository_tools.build_repo_index(force=True)
        self.temp_repo_helper.create_text_file("src/new_file.py", "def new_file():\n    return 2\n")

        cached_result = self.repository_tools.build_repo_index()
        repo_index = self._read_index(str(cached_result["index_path"]))
        records_by_path = self._file_records_by_path(repo_index)

        self.assertEqual("cached", cached_result["status"])
        self.assertEqual(first_result["index_path"], cached_result["index_path"])
        self.assertIn("src/original.py", records_by_path)
        self.assertNotIn("src/new_file.py", records_by_path)

    def test_build_repo_index_creates_output_file(self) -> None:
        self.temp_repo_helper.create_text_file("README.md", "hello\n")

        build_result = self.repository_tools.build_repo_index(force=True)
        index_path = self.temp_repo_helper.repo_root / str(build_result["index_path"])
        repo_index = json.loads(index_path.read_text(encoding="utf-8"))

        self.assertTrue(index_path.is_file())
        self.assertEqual(build_result["repo_id"], repo_index["repo_id"])
        self.assertEqual("README.md", repo_index["files"][0]["path"])

    def test_build_repo_index_serializes_only_relative_posix_paths(self) -> None:
        self.temp_repo_helper.create_text_file("src/main.py", "def main():\n    return 0\n")
        self.temp_repo_helper.create_text_file(
            ".repopilot/index/ignored/file_index.json",
            '{"path": "ignored.py"}\n',
        )

        build_result = self.repository_tools.build_repo_index(force=True)
        index_path = self.temp_repo_helper.repo_root / str(build_result["index_path"])
        serialized_index = index_path.read_text(encoding="utf-8")
        repo_index = json.loads(serialized_index)
        records_by_path = self._file_records_by_path(repo_index)

        self.assertNotIn(str(self.temp_repo_helper.repo_root), serialized_index)
        self.assertNotIn("\\", serialized_index)
        self.assertIn("src/main.py", records_by_path)
        self.assertTrue(
            all(
                not path.startswith("/")
                and ":" not in path
                and "\\" not in path
                and not path.startswith(".repopilot/index/")
                for path in records_by_path
            )
        )


class TestInspectRepoV05(unittest.TestCase):
    """验证 v0.5 紧凑项目概览工具。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def test_inspect_repo_contains_core_modules(self) -> None:
        self.temp_repo_helper.create_text_file(
            "src/core.py",
            "class Service:\n"
            "    def run(self):\n"
            "        return 1\n"
            "\n"
            "def helper():\n"
            "    return 2\n",
        )
        self.temp_repo_helper.create_text_file(
            "main.py",
            "def main():\n"
            "    return 0\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n",
        )
        self.temp_repo_helper.create_text_file(
            "tests/test_core.py",
            "def test_service():\n"
            "    assert True\n",
        )
        self.temp_repo_helper.create_text_file("README.md", "hello\n")

        inspect_result = self.repository_tools.inspect_repo()

        self.assertIn("inspect_repo", self.repository_tools.tools)
        self.assertEqual(True, inspect_result["ok"])
        self.assertEqual("built", inspect_result["index_status"])
        self.assertEqual(4, inspect_result["file_count"])
        self.assertEqual(3, inspect_result["python_file_count"])
        self.assertTrue((self.temp_repo_helper.repo_root / str(inspect_result["index_path"])).is_file())

        main_python_modules = inspect_result["main_python_modules"]
        test_files = inspect_result["test_files"]
        entrypoint_candidates = inspect_result["entrypoint_candidates"]
        if not isinstance(main_python_modules, list):
            self.fail("main_python_modules should be a list")
        if not isinstance(test_files, list):
            self.fail("test_files should be a list")
        if not isinstance(entrypoint_candidates, list):
            self.fail("entrypoint_candidates should be a list")

        main_module_paths = [
            module["path"]
            for module in main_python_modules
            if isinstance(module, dict)
        ]
        test_file_paths = [
            test_file["path"]
            for test_file in test_files
            if isinstance(test_file, dict)
        ]
        entrypoint_paths = [
            entrypoint["path"]
            for entrypoint in entrypoint_candidates
            if isinstance(entrypoint, dict)
        ]

        self.assertEqual("src/core.py", main_module_paths[0])
        self.assertIn("main.py", entrypoint_paths)
        self.assertIn("tests/test_core.py", test_file_paths)
        self.assertTrue(
            all(not path.startswith(".repopilot/index/") for path in main_module_paths)
        )

        cached_result = self.repository_tools.inspect_repo()

        self.assertEqual("cached", cached_result["index_status"])

    def test_inspect_repo_output_is_compact(self) -> None:
        large_file_body = "large-body-marker\n" * (MAX_FULL_READ_LINES + 1)
        self.temp_repo_helper.create_text_file("notes/large_notes.md", large_file_body)
        self.temp_repo_helper.create_text_file(
            "app.py",
            "def main():\n"
            "    return \"ok\"\n",
        )
        self.temp_repo_helper.create_text_file(
            "pkg/worker.py",
            "def work():\n"
            "    return 1\n",
        )
        self.temp_repo_helper.create_text_file(
            "tests/test_worker.py",
            "def test_work():\n"
            "    assert True\n",
        )

        inspect_result = self.repository_tools.inspect_repo()
        serialized_result = json.dumps(inspect_result, ensure_ascii=False, sort_keys=True)
        large_files = inspect_result["large_files"]
        if not isinstance(large_files, list):
            self.fail("large_files should be a list")
        large_file_records = [
            file_record
            for file_record in large_files
            if isinstance(file_record, dict) and file_record.get("path") == "notes/large_notes.md"
        ]

        self.assertLessEqual(len(serialized_result), MAX_TOOL_OUTPUT_CHARS)
        self.assertNotIn("large-body-marker", serialized_result)
        self.assertEqual(1, len(large_file_records))
        self.assertTrue(large_file_records[0]["exceeds_full_read_threshold"])
        self.assertTrue(large_file_records[0]["exceeds_lines_threshold"])
        self.assertIn("app.py", serialized_result)
        self.assertIn("tests/test_worker.py", serialized_result)

    def test_inspect_repo_respects_small_output_budget(self) -> None:
        for file_number in range(20):
            self.temp_repo_helper.create_text_file(
                f"src/package_with_long_name_{file_number}/module_with_long_name_{file_number}.py",
                "class Service:\n"
                "    def run(self):\n"
                "        return 1\n"
                "\n"
                "def helper():\n"
                "    return 2\n",
            )
        required_keys = {
            "ok",
            "index_status",
            "index_path",
            "file_count",
            "python_file_count",
            "main_python_modules",
            "test_files",
            "entrypoint_candidates",
            "large_files",
        }

        with mock.patch("tools.MAX_TOOL_OUTPUT_CHARS", new=500):
            inspect_result = self.repository_tools.inspect_repo()
            serialized_result = json.dumps(inspect_result, ensure_ascii=False, sort_keys=True)

        self.assertLessEqual(len(serialized_result), 500)
        self.assertTrue(required_keys.issubset(inspect_result.keys()))
        self.assertEqual(True, inspect_result["truncated"])


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

    real_repo_root = Path(__file__).resolve().parents[1]
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
        self.repo_root = Path(__file__).resolve().parents[1]
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
        """断言 patch_result.errors 列表包含预期的错误文本。"""
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

    def test_read_file_returns_numbered_project_content(self) -> None:
        file_content = self.repository_tools.read_file("context_stats.py")

        self.assertIn('1 | """上下文统计 schema。"""', file_content)
        self.assertIn("13 | class ContextStats:", file_content)

    def test_search_text_finds_max_steps_with_context(self) -> None:
        matches = self.repository_tools.search_text("MAX_STEPS", path_glob="config.py")

        self.assertTrue(matches)
        self.assertTrue(
            any("config.py:" in match and "MAX_STEPS" in match for match in matches)
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
        """返回最近一次运行的事件日志列表。"""
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
        original_file_path = self.temp_repo_helper.repo_root / "file.txt"
        original_sha256 = hashlib.sha256(original_file_path.read_bytes()).hexdigest()
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
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches" / str(patch_result["patch_id"])
        metadata_path = patch_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(patch_dir / "patch.diff", patch_path)
        self.assertEqual(diff_text, patch_path.read_text(encoding="utf-8"))
        self.assertTrue((patch_dir / "patch.diff").is_file())
        self.assertTrue(metadata_path.is_file())
        self.assertEqual("pending_approval", metadata["status"])
        self.assertEqual(instruction, metadata["instruction"])
        self.assertEqual("", metadata["run_id"])
        self.assertEqual("", metadata["task"])
        self.assertEqual("", metadata["summary"])
        self.assertEqual("unknown", metadata["risk_level"])
        self.assertIsNone(metadata["approved_at"])
        self.assertIsNone(metadata["applied_at"])
        self.assertIsNone(metadata["rejected_at"])
        self.assertIsNone(metadata["plan_snapshot"])
        self.assertEqual(["file.txt"], metadata["paths"])
        self.assertEqual(f".repopilot/patches/{patch_result['patch_id']}/patch.diff", metadata["diff_path"])
        self.assertEqual(hashlib.sha256(diff_text.encode("utf-8")).hexdigest(), metadata["diff_sha256"])
        self.assertEqual(
            [
                {
                    "path": "file.txt",
                    "existed_before": True,
                    "sha256_before": original_sha256,
                    "operation": "modify",
                }
            ],
            metadata["target_files"],
        )
        self.assertIn("需人工确认", "".join(metadata["warnings"]))
        self.assertIn("patch_id", patch_result)
        self.assertEqual([], patch_result["errors"])
        self.assertEqual(["file.txt"], patch_result["paths"])
        self.assertEqual(metadata["target_files"], patch_result["target_files"])
        self.assertTrue(patch_result["next_commands"])
        self.assertEqual(diff_text, patch_result["diff_preview"])
        self.assertEqual(original_target, self.temp_repo_helper.read_text_file("file.txt"))

    def test_propose_patch_v03_accepts_optional_metadata_context(self) -> None:
        diff_text = (
            "diff --git a/new_file.txt b/new_file.txt\n"
            "--- /dev/null\n"
            "+++ b/new_file.txt\n"
            "@@ -0,0 +1 @@\n"
            "+created\n"
        )

        patch_result = self.repository_tools.propose_patch(
            instruction="创建文件",
            diff=diff_text,
            run_id="run-1",
            task="task-2",
            summary="创建 new_file.txt",
            plan_snapshot={"step": "保存补丁"},
            risk_level="low",
        )

        self.assertTrue(patch_result["ok"])
        metadata_path = (
            self.temp_repo_helper.repo_root
            / ".repopilot/patches"
            / str(patch_result["patch_id"])
            / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual("run-1", metadata["run_id"])
        self.assertEqual("task-2", metadata["task"])
        self.assertEqual("创建 new_file.txt", metadata["summary"])
        self.assertEqual({"step": "保存补丁"}, metadata["plan_snapshot"])
        self.assertEqual("low", metadata["risk_level"])
        self.assertEqual(
            [
                {
                    "path": "new_file.txt",
                    "existed_before": False,
                    "sha256_before": None,
                    "operation": "create",
                }
            ],
            metadata["target_files"],
        )
        self.assertFalse((self.temp_repo_helper.repo_root / "new_file.txt").exists())

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
        existing_patch_dirs = [path for path in patch_dir.iterdir() if path.is_dir()]

        patch_result = self.repository_tools.propose_patch(
            instruction="无效补丁",
            diff="invalid diff content",
        )

        self.assertFalse(patch_result["ok"])
        self.assertTrue(patch_result["errors"])
        self.assertEqual(existing_patch_dirs, [path for path in patch_dir.iterdir() if path.is_dir()])

    def test_propose_patch_v03_rejects_hunk_body_count_mismatch_without_writing(self) -> None:
        self.temp_repo_helper.create_text_file("count_mismatch.txt", "one\ntwo\nthree\n")
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches"
        existing_patch_dirs = [path for path in patch_dir.iterdir() if path.is_dir()]
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
        self.assertEqual(existing_patch_dirs, [path for path in patch_dir.iterdir() if path.is_dir()])
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

    def test_propose_patch_rejects_empty_diff_without_storage_mutation(self) -> None:
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches"
        existing_patch_dirs = [path for path in patch_dir.iterdir() if path.is_dir()]

        patch_result = self.repository_tools.propose_patch(
            instruction="空 diff",
            diff="\n\t  ",
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual(["diff must not be empty"], patch_result["errors"])
        self.assertEqual(existing_patch_dirs, [path for path in patch_dir.iterdir() if path.is_dir()])

    def test_propose_patch_rejects_paths_with_spaces_without_storage_mutation(self) -> None:
        patch_dir = self.temp_repo_helper.repo_root / ".repopilot/patches"
        existing_patch_dirs = [path for path in patch_dir.iterdir() if path.is_dir()]
        diff_text = "diff --git a/space path.txt b/space path.txt\n--- a/space path.txt\n+++ b/space path.txt\n@@ -0,0 +1 @@\n+new\n"

        patch_result = self.repository_tools.propose_patch(
            instruction="path with spaces",
            diff=diff_text,
        )

        self.assertFalse(patch_result["ok"])
        errors_object = patch_result["errors"]
        self.assertIsInstance(errors_object, list)
        if not isinstance(errors_object, list):
            self.fail("errors should be a list")
        self.assertTrue(any("diff 缺少合法文件头" in str(error) for error in errors_object))
        self.assertEqual(existing_patch_dirs, [path for path in patch_dir.iterdir() if path.is_dir()])

    def test_propose_patch_normalizes_windows_separators_in_targets(self) -> None:
        self.temp_repo_helper.create_text_file("win/path.txt", "old\n")
        diff_text = "diff --git a/win\\path.txt b/win\\path.txt\n--- a/win\\path.txt\n+++ b/win\\path.txt\n@@ -1 +1 @@\n-old\n+new\n"

        patch_result = self.repository_tools.propose_patch(
            instruction="windows separators",
            diff=diff_text,
        )

        self.assertTrue(patch_result["ok"], patch_result)
        self.assertEqual(["win/path.txt"], patch_result["paths"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("win/path.txt"))

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
    """验证 v0.3 保存补丁的应用、备份、拒绝和回滚行为。"""

    def setUp(self) -> None:
        self.temp_repo_helper = V03TempRepoHelper()
        self.repository_tools = self.temp_repo_helper.create_repository_tools()

    def tearDown(self) -> None:
        self.temp_repo_helper.cleanup()

    def _propose_patch(self, diff_text: str, instruction: str = "apply patch") -> str:
        """辅助方法：提交补丁提案并返回 patch_id。"""
        patch_result = self.repository_tools.propose_patch(
            instruction=instruction,
            diff=diff_text,
        )
        self.assertTrue(patch_result["ok"], patch_result)
        return str(patch_result["patch_id"])

    def _metadata(self, patch_id: str) -> dict[str, object]:
        """辅助方法：读取指定补丁的 metadata.json。"""
        metadata_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / patch_id / "metadata.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _run_events(self, patch_id: str) -> list[dict[str, object]]:
        """辅助方法：读取指定补丁的运行事件日志。"""
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
        self.assertEqual(
            [{"command_name": None, "status": "skipped", "reason": "no test_commands"}],
            metadata["verification_results"],
        )
        self.assertEqual(f".repopilot/backups/{patch_id}", metadata["backup_dir"])
        self.assertEqual(["src/existing.txt", "src/created.txt"], metadata["modified_files"])
        manifest_path = self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(patch_id, manifest["patch_id"])
        self.assertEqual(metadata["diff_sha256"], manifest["diff_sha256"])
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
        metadata["status"] = "proposed"
        self.repository_tools._write_patch_metadata(metadata)
        non_proposed_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(non_proposed_result["ok"])
        self.assertEqual(["only status=pending_approval patches can be applied or rejected"], non_proposed_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("state.txt"))
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_show_patch_lists_preview_and_full_diff(self) -> None:
        self.temp_repo_helper.create_text_file("show.txt", "old\n")
        diff_text = "diff --git a/show.txt b/show.txt\n--- a/show.txt\n+++ b/show.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)

        list_result = self.repository_tools.list_patches()
        preview_result = self.repository_tools.show_patch(patch_id)
        full_result = self.repository_tools.show_patch(patch_id, full=True)

        self.assertTrue(list_result["ok"])
        patches = list_result["patches"]
        self.assertIsInstance(patches, list)
        if not isinstance(patches, list):
            self.fail("patches should be a list")
        self.assertEqual([patch_id], [patch["patch_id"] for patch in patches if isinstance(patch, dict)])
        self.assertTrue(preview_result["ok"])
        self.assertEqual(diff_text, preview_result["diff_preview"])
        self.assertEqual("", preview_result["diff"])
        self.assertTrue(full_result["ok"])
        self.assertEqual(diff_text, full_result["diff"])
        self.assertEqual("", full_result["diff_preview"])

    def test_apply_patch_rejects_sha256_tamper_without_metadata_mutation(self) -> None:
        self.temp_repo_helper.create_text_file("tamper.txt", "old\n")
        diff_text = "diff --git a/tamper.txt b/tamper.txt\n--- a/tamper.txt\n+++ b/tamper.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        patch_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / patch_id / "patch.diff"
        patch_path.write_text(
            "diff --git a/tamper.txt b/tamper.txt\n--- a/tamper.txt\n+++ b/tamper.txt\n@@ -1 +1 @@\n-old\n+evil\n",
            encoding="utf-8",
        )

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual(["patch.diff sha256 does not match metadata.diff_sha256"], apply_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("tamper.txt"))
        self.assertEqual("pending_approval", self._metadata(patch_id)["status"])
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_apply_patch_rejects_metadata_target_mismatch_before_write(self) -> None:
        self.temp_repo_helper.create_text_file("target.txt", "old\n")
        diff_text = "diff --git a/target.txt b/target.txt\n--- a/target.txt\n+++ b/target.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        metadata = self._metadata(patch_id)
        metadata["target_files"] = [
            {"path": "other.txt", "existed_before": False, "sha256_before": None, "operation": "create"}
        ]
        self.repository_tools._write_patch_metadata(metadata)

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual(["metadata.target_files paths do not match patch.diff targets"], apply_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("target.txt"))
        self.assertEqual("pending_approval", self._metadata(patch_id)["status"])

    def test_apply_patch_rejects_stale_modify_sha_before_backup_or_write(self) -> None:
        self.temp_repo_helper.create_text_file("stale.txt", "old\n")
        diff_text = "diff --git a/stale.txt b/stale.txt\n--- a/stale.txt\n+++ b/stale.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        self.temp_repo_helper.create_text_file("stale.txt", "old but changed\n")

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("rejected", apply_result["status"])
        self.assertEqual(
            ["metadata target state mismatch: stale.txt sha256_before does not match current file"],
            apply_result["errors"],
        )
        self.assertEqual("old but changed\n", self.temp_repo_helper.read_text_file("stale.txt"))
        self.assertEqual("pending_approval", self._metadata(patch_id)["status"])
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_apply_patch_rejects_create_target_that_now_exists_before_backup_or_write(self) -> None:
        diff_text = (
            "diff --git a/new_existing.txt b/new_existing.txt\n"
            "--- /dev/null\n"
            "+++ b/new_existing.txt\n"
            "@@ -0,0 +1 @@\n"
            "+created\n"
        )
        patch_id = self._propose_patch(diff_text)
        self.temp_repo_helper.create_text_file("new_existing.txt", "already here\n")

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("rejected", apply_result["status"])
        self.assertTrue(apply_result["errors"])
        self.assertEqual("already here\n", self.temp_repo_helper.read_text_file("new_existing.txt"))
        self.assertEqual("pending_approval", self._metadata(patch_id)["status"])
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_apply_patch_rejects_missing_modify_target_before_backup_or_write(self) -> None:
        diff_text = (
            "diff --git a/missing_modify.txt b/missing_modify.txt\n"
            "--- /dev/null\n"
            "+++ b/missing_modify.txt\n"
            "@@ -0,0 +1 @@\n"
            "+created\n"
        )
        patch_id = self._propose_patch(diff_text)
        metadata = self._metadata(patch_id)
        metadata["target_files"] = [
            {
                "path": "missing_modify.txt",
                "existed_before": True,
                "sha256_before": "0" * 64,
                "operation": "modify",
            }
        ]
        self.repository_tools._write_patch_metadata(metadata)

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("rejected", apply_result["status"])
        self.assertEqual(
            ["metadata target state mismatch: missing_modify.txt modify target is missing"],
            apply_result["errors"],
        )
        self.assertFalse((self.temp_repo_helper.repo_root / "missing_modify.txt").exists())
        self.assertEqual("pending_approval", self._metadata(patch_id)["status"])
        self.assertFalse((self.temp_repo_helper.repo_root / ".repopilot/backups" / patch_id).exists())

    def test_reject_patch_blocks_later_apply_without_business_file_changes(self) -> None:
        self.temp_repo_helper.create_text_file("reject_me.txt", "old\n")
        diff_text = "diff --git a/reject_me.txt b/reject_me.txt\n--- a/reject_me.txt\n+++ b/reject_me.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)

        reject_result = self.repository_tools.reject_patch(patch_id)
        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertTrue(reject_result["ok"])
        self.assertEqual("rejected", reject_result["status"])
        metadata = self._metadata(patch_id)
        self.assertEqual("rejected", metadata["status"])
        self.assertIsNotNone(metadata["rejected_at"])
        self.assertFalse(apply_result["ok"])
        self.assertEqual(["only status=pending_approval patches can be applied or rejected"], apply_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("reject_me.txt"))

    def test_apply_patch_revalidates_saved_diff_and_rejects_raw_diff_args(self) -> None:
        self.temp_repo_helper.create_text_file("safe.txt", "old\n")
        diff_text = "diff --git a/safe.txt b/safe.txt\n--- a/safe.txt\n+++ b/safe.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        patch_path = self.temp_repo_helper.repo_root / ".repopilot/patches" / patch_id / "patch.diff"
        patch_path.write_text("not a unified diff\n", encoding="utf-8")

        invalid_saved_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(invalid_saved_result["ok"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("safe.txt"))
        with self.assertRaises(ToolError):
            self.repository_tools.run_tool(
                {"tool": "apply_patch", "args": {"patch_id": patch_id, "diff": diff_text}}
            )

    def test_apply_patch_reports_current_state_mismatch_clearly(self) -> None:
        self.temp_repo_helper.create_text_file("mismatch.txt", "current\n")
        diff_text = "diff --git a/mismatch.txt b/mismatch.txt\n--- a/mismatch.txt\n+++ b/mismatch.txt\n@@ -1 +1 @@\n-current\n+patched\n"
        patch_id = self._propose_patch(diff_text)
        self.temp_repo_helper.create_text_file("mismatch.txt", "changed before apply\n")

        apply_result = self.repository_tools.apply_patch(patch_id)

        self.assertFalse(apply_result["ok"])
        self.assertEqual("rejected", apply_result["status"])
        self.assertEqual(
            ["metadata target state mismatch: mismatch.txt sha256_before does not match current file"],
            apply_result["errors"],
        )
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
        rollback_result = apply_result["rollback"]
        self.assertIsInstance(rollback_result, dict)
        if not isinstance(rollback_result, dict):
            self.fail("rollback should be a dict")
        self.assertEqual(True, rollback_result["attempted"])
        self.assertEqual(True, rollback_result["ok"])
        self.assertEqual([], rollback_result["errors"])
        self.assertTrue(rollback_result["files"])
        self.assertEqual("old first\n", self.temp_repo_helper.read_text_file("first.txt"))
        self.assertEqual("old second\n", self.temp_repo_helper.read_text_file("second.txt"))
        self.assertFalse((self.temp_repo_helper.repo_root / "created_after_failure.txt").exists())
        metadata = self._metadata(patch_id)
        self.assertEqual("failed", metadata["status"])
        self.assertEqual(rollback_result, metadata["rollback"])
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
                and isinstance(details.get("rollback"), dict)
                and details["rollback"].get("attempted") is True
                and details["rollback"].get("ok") is True
            ):
                apply_failure_found = True
        self.assertTrue(apply_failure_found)
        rollback_success_found = False
        for event in run_events:
            details = event.get("details")
            if not isinstance(details, dict):
                continue
            if (
                event["event_type"] == "rollback_success"
                and event["status"] == "ok"
                and details.get("errors") == []
                and isinstance(details.get("files"), list)
            ):
                rollback_success_found = True
        self.assertTrue(rollback_success_found)

    def test_apply_patch_rolls_back_when_whitelist_test_fails(self) -> None:
        self.temp_repo_helper.create_text_file("verify.txt", "old\n")
        diff_text = "diff --git a/verify.txt b/verify.txt\n--- a/verify.txt\n+++ b/verify.txt\n@@ -1 +1 @@\n-old\n+new\n"
        patch_id = self._propose_patch(diff_text)
        metadata = self._metadata(patch_id)
        metadata["plan_snapshot"] = {"test_commands": [{"command_name": "unit"}]}
        self.repository_tools._write_patch_metadata(metadata)

        with mock.patch.object(
            self.repository_tools,
            "run_tests",
            return_value={"command_name": "unit", "exit_code": 1, "stdout": "fail", "stderr": "", "timed_out": False},
        ) as mocked_run_tests:
            apply_result = self.repository_tools.apply_patch(patch_id)

        mocked_run_tests.assert_called_once_with("unit")
        self.assertFalse(apply_result["ok"])
        self.assertEqual("failed", apply_result["status"])
        self.assertEqual(["patch verification failed"], apply_result["errors"])
        self.assertEqual("old\n", self.temp_repo_helper.read_text_file("verify.txt"))
        self.assertEqual(
            [{"command_name": "unit", "status": "failed", "exit_code": 1, "timed_out": False, "stdout": "fail", "stderr": ""}],
            apply_result["verification_results"],
        )
        failed_metadata = self._metadata(patch_id)
        self.assertEqual("failed", failed_metadata["status"])
        self.assertEqual(apply_result["verification_results"], failed_metadata["verification_results"])


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
        self.assertEqual(patch_path.name, "patch.diff")
        self.assertEqual(patch_path.parent.name, patch_id)
        self.assertEqual(metadata_path.name, "metadata.json")
        self.assertEqual(metadata_path.parent.name, patch_id)
        self.assertTrue((self.temp_repo_helper.repo_root / ".repopilot/patches" / patch_id / "patch.diff").is_file())
        self.assertTrue((self.temp_repo_helper.repo_root / ".repopilot/patches" / patch_id / "metadata.json").is_file())
        self.assertEqual(patch_text, self.repository_tools._read_patch_file(patch_id))
        self.assertEqual(metadata, self.repository_tools._read_patch_metadata(patch_id))
        self.assertEqual(patch_id, metadata["patch_id"])
        self.assertEqual("更新 example.txt", metadata["instruction"])
        self.assertEqual("pending_approval", metadata["status"])
        self.assertEqual(["example.txt"], metadata["paths"])
        self.assertEqual([], metadata["target_files"])
        self.assertEqual("", metadata["diff_path"])
        self.assertEqual("", metadata["diff_sha256"])
        self.assertIsNone(metadata["approved_at"])
        self.assertIsNone(metadata["applied_at"])
        self.assertIsNone(metadata["rejected_at"])
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

    def test_read_file_range_start_beyond_eof_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolError, "超过文件总行数"):
            self.repository_tools.read_file_range("agent.py", 10_000, 10_001)

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

    def test_read_file_range_allows_oversized_file(self) -> None:
        self.create_file("range_oversized.txt", "a" * (MAX_FILE_BYTES + 1))

        file_content = self.repository_tools.read_file_range("range_oversized.txt", 1, 1)

        self.assertEqual(f"1 | {'a' * (MAX_FILE_BYTES + 1)}", file_content)


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
