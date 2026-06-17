"""只读代码库工具集合。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import MAX_FILE_BYTES


SKIPPED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
}
SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}
SECRET_FILE_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
}


class ToolError(ValueError):
    """工具参数或执行结果不符合安全规则。"""


@dataclass(frozen=True)
class ToolSpec:
    """工具注册信息。"""

    name: str
    description: str
    function: Callable[..., object]


class RepositoryTools:
    """面向单个目标代码库的只读工具。"""

    def __init__(self, repo_root: str | Path) -> None:
        resolved_repo_root = Path(repo_root).expanduser().resolve()
        if not resolved_repo_root.is_dir():
            raise ToolError(f"repo 路径不是目录: {repo_root}")

        self.repo_root = resolved_repo_root
        self.tools: dict[str, ToolSpec] = {
            "list_dir": ToolSpec(
                name="list_dir",
                description='列出 repo 内目录条目，参数: {"path": "相对路径"}',
                function=self.list_dir,
            ),
            "read_file": ToolSpec(
                name="read_file",
                description='读取 repo 内 UTF-8 文本文件并返回带行号内容，最大 20KB，参数: {"path": "相对路径"}',
                function=self.read_file,
            ),
            "search_text": ToolSpec(
                name="search_text",
                description='在 repo 内搜索文本关键字，返回文件名、行号和带行号上下文，参数: {"keyword": "关键字"}',
                function=self.search_text,
            ),
            "finish": ToolSpec(
                name="finish",
                description='完成任务并输出最终答案，尽量引用文件名、函数名和行号，参数: {"answer": "最终答案"}',
                function=self.finish,
            ),
        }

    def tool_descriptions(self) -> str:
        """返回给模型阅读的工具说明。"""

        return "\n".join(
            f"- {tool_spec.name}: {tool_spec.description}"
            for tool_spec in self.tools.values()
        )

    def list_dir(self, path: str) -> list[str]:
        """列出目录下的非隐藏文件和文件夹。

        Args:
            path: repo 内目录路径。

        Returns:
            文件和文件夹名称列表，目录名以 "/" 结尾。
        """

        target_dir = self._resolve_repo_path(path)
        if not target_dir.is_dir():
            raise ToolError(f"path 不是目录: {path}")

        entries: list[str] = []
        for child_path in sorted(target_dir.iterdir(), key=lambda item: item.name.lower()):
            if child_path.name.startswith("."):
                continue
            suffix = "/" if child_path.is_dir() else ""
            entries.append(f"{child_path.name}{suffix}")

        return entries

    def read_file(self, path: str) -> str:
        """读取 repo 内 UTF-8 文本文件。

        Args:
            path: repo 内文件路径。

        Returns:
            带行号的文件文本内容。
        """

        target_file = self._resolve_repo_path(path)
        if not target_file.is_file():
            raise ToolError(f"path 不是文件: {self._format_repo_path(target_file)}")

        file_size = target_file.stat().st_size
        if file_size > MAX_FILE_BYTES:
            raise ToolError(f"文件超过 20KB 限制: {path}")

        self._validate_readable_file(target_file)

        try:
            file_text = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(f"文件不是有效的 UTF-8 文本: {path}") from error

        return "\n".join(self._format_numbered_lines(file_text.splitlines(), start_line=1))

    def search_text(self, keyword: str) -> list[str]:
        """在 repo 内搜索关键字。

        Args:
            keyword: 要搜索的关键字。

        Returns:
            命中列表，包含文件名、行号和附近带行号代码片段。
        """

        normalized_keyword = keyword.strip()
        if not normalized_keyword:
            raise ToolError("keyword 不能为空")

        matches: list[str] = []
        for file_path in self._iter_searchable_files():
            lines = self._read_text_lines(file_path)
            if lines is None:
                continue

            for line_number, line_text in enumerate(lines, start=1):
                if normalized_keyword not in line_text:
                    continue

                snippet_start = max(line_number - 2, 1)
                snippet_end = min(line_number + 2, len(lines))
                snippet_lines = lines[snippet_start - 1:snippet_end]
                formatted_snippet = self._format_numbered_lines(
                    snippet_lines,
                    start_line=snippet_start,
                )
                matches.append(
                    "\n".join(
                        [
                            f"{self._format_repo_path(file_path)}:{line_number}",
                            *formatted_snippet,
                        ]
                    )
                )

        return matches

    def finish(self, answer: str) -> str:
        """输出最终答案。"""

        normalized_answer = answer.strip()
        if not normalized_answer:
            raise ToolError("answer 不能为空")

        return normalized_answer

    def run_tool(self, tool_call: dict[str, object]) -> object:
        """执行模型产生的工具调用对象。"""

        tool_name = tool_call.get("tool")
        if not isinstance(tool_name, str):
            raise ToolError("tool 必须是字符串")

        tool_args = tool_call.get("args", {})
        if not isinstance(tool_args, dict):
            raise ToolError("args 必须是对象")

        tool_spec = self.tools.get(tool_name)
        if tool_spec is None:
            raise ToolError(f"未知工具: {tool_name}")

        try:
            return tool_spec.function(**tool_args)
        except TypeError as error:
            raise ToolError(f"工具参数不匹配: {tool_name}") from error

    def run_tool_json(self, tool_call_json: str) -> str:
        """执行 JSON 字符串形式的工具调用。"""

        try:
            tool_call = json.loads(tool_call_json)
        except json.JSONDecodeError as error:
            raise ToolError("工具调用不是有效 JSON") from error

        if not isinstance(tool_call, dict):
            raise ToolError("工具调用必须是 JSON 对象")

        tool_output = self.run_tool(tool_call)
        return json.dumps({"ok": True, "output": tool_output}, ensure_ascii=False, indent=2)

    def _resolve_repo_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path 必须是非空字符串")

        resolved_path = (self.repo_root / path).resolve()
        try:
            resolved_path.relative_to(self.repo_root)
        except ValueError as error:
            raise ToolError(f"path 必须位于 repo 目录内部: {path}") from error

        return resolved_path

    def _validate_readable_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise ToolError(f"path 不是文件: {self._format_repo_path(file_path)}")
        if self._has_hidden_path_part(file_path):
            raise ToolError(f"禁止读取隐藏路径: {self._format_repo_path(file_path)}")
        if self._is_secret_file(file_path):
            raise ToolError(f"禁止读取密钥或环境文件: {self._format_repo_path(file_path)}")
        if self._is_binary_file(file_path):
            raise ToolError(f"禁止读取二进制文件: {self._format_repo_path(file_path)}")

    def _iter_searchable_files(self) -> list[Path]:
        searchable_files: list[Path] = []
        for current_dir, dir_names, file_names in os.walk(self.repo_root):
            dir_names[:] = [
                dir_name
                for dir_name in dir_names
                if not self._should_skip_dir(dir_name)
            ]

            current_dir_path = Path(current_dir)
            for file_name in file_names:
                file_path = current_dir_path / file_name
                if self._should_skip_file(file_path):
                    continue
                searchable_files.append(file_path)

        return sorted(searchable_files, key=lambda item: self._format_repo_path(item).lower())

    def _should_skip_dir(self, dir_name: str) -> bool:
        return dir_name.startswith(".") or dir_name in SKIPPED_DIR_NAMES

    def _should_skip_file(self, file_path: Path) -> bool:
        try:
            resolved_file_path = file_path.resolve()
            resolved_file_path.relative_to(self.repo_root)
        except (OSError, ValueError):
            return True

        if self._has_hidden_path_part(file_path):
            return True
        if self._is_secret_file(file_path):
            return True
        try:
            file_size = file_path.stat().st_size
        except OSError:
            return True
        if file_size > MAX_FILE_BYTES:
            return True
        return self._is_binary_file(file_path)

    def _read_text_lines(self, file_path: Path) -> list[str] | None:
        try:
            return file_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return None

    def _format_numbered_lines(self, lines: list[str], start_line: int) -> list[str]:
        return [
            f"{line_number} | {line_text}"
            for line_number, line_text in enumerate(lines, start=start_line)
        ]

    def _has_hidden_path_part(self, file_path: Path) -> bool:
        relative_parts = file_path.relative_to(self.repo_root).parts
        return any(part.startswith(".") for part in relative_parts)

    def _is_secret_file(self, file_path: Path) -> bool:
        file_name = file_path.name.lower()
        return file_name in SECRET_FILE_NAMES or file_path.suffix.lower() in SECRET_FILE_SUFFIXES

    def _is_binary_file(self, file_path: Path) -> bool:
        try:
            with file_path.open("rb") as file_stream:
                sample_bytes = file_stream.read(4096)
        except OSError:
            return True

        return b"\x00" in sample_bytes

    def _format_repo_path(self, file_path: Path) -> str:
        return file_path.relative_to(self.repo_root).as_posix()
