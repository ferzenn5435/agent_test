"""RepositoryTools 基础单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools import RepositoryTools, ToolError


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
        self.assertIn("7 | from config import MAX_STEPS", file_content)

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

    def test_propose_patch_single_replacement_returns_diff(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")
        plan = "更新模块说明"

        patch_result = self.repository_tools.propose_patch(
            file_path="agent.py",
            plan=plan,
            replacements=[
                {
                    "old_text": '"""LLM + 工具调用循环。"""',
                    "new_text": '"""LLM 工具调用循环。"""',
                }
            ],
        )

        self.assertTrue(patch_result["ok"])
        self.assertEqual("agent.py", patch_result["file_path"])
        self.assertEqual(plan, patch_result["plan"])
        self.assertEqual([], patch_result["errors"])
        diff_object = patch_result["diff"]
        self.assertIsInstance(diff_object, str)
        if not isinstance(diff_object, str):
            self.fail("diff 应为字符串")
        diff = diff_object
        self.assertIn("--- a/agent.py", diff)
        self.assertIn("+++ b/agent.py", diff)
        self.assertIn("@@", diff)
        self.assertIn('-"""LLM + 工具调用循环。"""', diff)
        self.assertIn('+"""LLM 工具调用循环。"""', diff)
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_multi_block_returns_single_file_diff(self) -> None:
        target_file = self.repo_root / "tools.py"
        original_content = target_file.read_text(encoding="utf-8")
        plan = "调整工具说明文本"

        patch_result = self.repository_tools.propose_patch(
            file_path="tools.py",
            plan=plan,
            replacements=[
                {
                    "old_text": '"""只读代码库工具集合。"""',
                    "new_text": '"""代码库工具集合。"""',
                },
                {
                    "old_text": "class ToolError(ValueError):",
                    "new_text": "class RepositoryToolError(ValueError):",
                },
            ],
        )

        self.assertTrue(patch_result["ok"])
        self.assertEqual("tools.py", patch_result["file_path"])
        self.assertEqual(plan, patch_result["plan"])
        self.assertEqual([], patch_result["errors"])
        diff_object = patch_result["diff"]
        self.assertIsInstance(diff_object, str)
        if not isinstance(diff_object, str):
            self.fail("diff 应为字符串")
        diff = diff_object
        self.assertIn("--- a/tools.py", diff)
        self.assertIn("+++ b/tools.py", diff)
        self.assertIn("@@", diff)
        self.assertIn('-"""只读代码库工具集合。"""', diff)
        self.assertIn('+"""代码库工具集合。"""', diff)
        self.assertIn("-class ToolError(ValueError):", diff)
        self.assertIn("+class RepositoryToolError(ValueError):", diff)
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_does_not_write_when_replacement_fails(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="agent.py",
            plan="尝试替换不存在文本",
            replacements=[
                {
                    "old_text": "不存在的唯一文本片段",
                    "new_text": "不会写入的文本片段",
                }
            ],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assertTrue(patch_result["errors"])
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))


    def test_propose_patch_rejects_multiple_match_old_text_without_writing(self) -> None:
        target_file = self.repo_root / "tools.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="tools.py",
            plan="reject multiple matches",
            replacements=[{"old_text": "        ", "new_text": "    "}],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "matched multiple locations")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_empty_replacements_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="agent.py",
            plan="reject empty replacements",
            replacements=[],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "replacements must not be empty")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_empty_replacement_text_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="agent.py",
            plan="reject empty replacement text",
            replacements=[{"old_text": "from __future__ import annotations", "new_text": ""}],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "new_text must not be empty")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_duplicate_replacement_ranges_without_writing(self) -> None:
        target_file = self.repo_root / "tools.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="tools.py",
            plan="reject duplicate ranges",
            replacements=[
                {
                    "old_text": "class ToolError(ValueError):",
                    "new_text": "class RepositoryToolError(ValueError):",
                },
                {
                    "old_text": "class ToolError(ValueError):",
                    "new_text": "class LocalToolError(ValueError):",
                },
            ],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "duplicates")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_overlapping_ranges_without_writing(self) -> None:
        target_file = self.repo_root / "tools.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="tools.py",
            plan="reject overlapping ranges",
            replacements=[
                {
                    "old_text": "class ToolError(ValueError):\n    \"\"\"",
                    "new_text": "class ToolError(RuntimeError):\n    \"\"\"",
                },
                {
                    "old_text": "ToolError(ValueError)",
                    "new_text": "ToolError(RuntimeError)",
                },
            ],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "overlapping")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_path_escape_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="../outside.txt",
            plan="reject path escape",
            replacements=[{"old_text": "outside", "new_text": "inside"}],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "repo")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_absolute_path_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path=str(target_file),
            plan="reject absolute path",
            replacements=[
                {
                    "old_text": '"""LLM + 工具调用循环。"""',
                    "new_text": '"""LLM 工具调用循环。"""',
                }
            ],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "相对路径")
        self.assert_patch_errors_contain(patch_result, "绝对路径")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_directory_target_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path=".",
            plan="reject directory target",
            replacements=[{"old_text": "agent.py", "new_text": "main.py"}],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "path")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_missing_file_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="not_exist.py",
            plan="reject missing file",
            replacements=[{"old_text": "missing", "new_text": "present"}],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "not_exist.py")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_propose_patch_rejects_binary_target_without_writing(self) -> None:
        target_file = self.repo_root / "binary_target.bin"
        target_file.write_bytes(b"abc\x00def")
        original_content = target_file.read_bytes()

        try:
            patch_result = self.repository_tools.propose_patch(
                file_path="binary_target.bin",
                plan="reject binary target",
                replacements=[{"old_text": "abc", "new_text": "xyz"}],
            )

            self.assertFalse(patch_result["ok"])
            self.assertEqual("", patch_result["diff"])
            self.assert_patch_errors_contain(patch_result, "binary_target.bin")
            self.assertEqual(original_content, target_file.read_bytes())
        finally:
            if target_file.exists():
                target_file.unlink()

    def test_propose_patch_rejects_no_op_replacement_without_writing(self) -> None:
        target_file = self.repo_root / "agent.py"
        original_content = target_file.read_text(encoding="utf-8")

        patch_result = self.repository_tools.propose_patch(
            file_path="agent.py",
            plan="reject no-op replacement",
            replacements=[
                {
                    "old_text": "from __future__ import annotations",
                    "new_text": "from __future__ import annotations",
                }
            ],
        )

        self.assertFalse(patch_result["ok"])
        self.assertEqual("", patch_result["diff"])
        self.assert_patch_errors_contain(patch_result, "produces no change")
        self.assertEqual(original_content, target_file.read_text(encoding="utf-8"))

    def test_apply_patch_is_not_registered_and_dispatch_rejects_it(self) -> None:
        self.assertNotIn("apply_patch", self.repository_tools.tools)

        with self.assertRaisesRegex(ToolError, "apply_patch"):
            self.repository_tools.run_tool({"tool": "apply_patch", "args": {}})


if __name__ == "__main__":
    unittest.main()
