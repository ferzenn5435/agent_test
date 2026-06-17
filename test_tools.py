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


if __name__ == "__main__":
    unittest.main()
