"""命令行参数解析单元测试。"""

from __future__ import annotations

import unittest

from config import MAX_STEPS
from main import parse_args


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


if __name__ == "__main__":
    unittest.main()
