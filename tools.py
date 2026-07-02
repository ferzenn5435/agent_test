"""只读代码库工具集合。

核心目标：在 repo_root 内提供可审计的只读/受控写操作。
约束原则如下：
1. 路径沙箱：所有路径先解析为 repo_root 下路径，拒绝越界。
2. 工具边界：默认不读隐藏目录/文件、敏感文件与二进制。
3. 补丁边界：只允许 apply pending 的统一差异补丁，带完整元数据与 hash 校验。
4. 测试边界：仅允许白名单测试命令 unit / compile。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import (
    MAX_FILE_BYTES,
    MAX_FULL_READ_BYTES,
    MAX_FULL_READ_LINES,
    MAX_RANGE_READ_LINES,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_OUTPUT_CHARS,
)
import config

RUN_TEST_OUTPUT_MAX_BYTES = getattr(config, "RUN_TEST_OUTPUT_MAX_BYTES", 20 * 1024)
RUN_TEST_TIMEOUT_SECONDS = getattr(config, "RUN_TEST_TIMEOUT_SECONDS", 60)


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
# search/search_text 的目录跳过集合：.git/缓存/构建等目录会被直接过滤，避免噪音与性能退化。
REPO_INDEX_SKIPPED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "logs",
    ".repopilot",
    "node_modules",
    "dist",
    "build",
}
# repo 索引扫描的目录跳过集合：与 search 不同，.venv 与日志目录等也会排除。
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
# 名称命中敏感文件（如 .env、SSH key）一律不允许读取或修改。
SECRET_FILE_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
}
DIFF_GIT_HEADER_PATTERN = re.compile(r"^diff --git (\S+) (\S+)$")
DIFF_FILE_HEADER_PATTERN = re.compile(r"^(---|\+\+\+) (\S+)(?:\t.*)?$")
DIFF_HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
BINARY_PATCH_MARKERS = ("Binary files ", "GIT binary patch")
UNSUPPORTED_DIFF_HEADERS = ("rename from ", "rename to ", "copy from ", "copy to ")


class ToolError(ValueError):
    """工具参数或执行结果不符合安全规则。"""


@dataclass(frozen=True)
class ToolSpec:
    """工具注册信息。"""

    name: str
    description: str
    function: Callable[..., object]


class RepositoryTools:
    """面向单个目标代码库的工具集合。

包含路径安全、搜索边界、补丁元数据、备份回滚、测试白名单等基础安全机制。
"""

    # .repopilot 存放 patch/metadata/backups/runs 日志：
    # 目的不是“执行权限”，而是审计、恢复和验证证据的保真落盘。

    def __init__(self, repo_root: str | Path) -> None:
        """初始化仓库边界与工具注册清单。"""
        resolved_repo_root = Path(repo_root).expanduser().resolve()
        if not resolved_repo_root.is_dir():
            raise ToolError(f"repo 路径不是目录: {repo_root}")

        self.repo_root = resolved_repo_root
        self.repopilot_root = self.repo_root / ".repopilot"
        self.repopilot_patches_dir = self.repopilot_root / "patches"
        self.repopilot_backups_dir = self.repopilot_root / "backups"
        self.repopilot_runs_dir = self.repopilot_root / "runs"
        self._ensure_repopilot_storage()
        self.tools: dict[str, ToolSpec] = {
            "list_dir": ToolSpec(
                name="list_dir",
                description='列出 repo 内目录条目，参数: {"path": "相对路径"}',
                function=self.list_dir,
            ),
            "read_file": ToolSpec(
                name="read_file",
                description=(
                    '读取 repo 内 UTF-8 文本文件并返回带行号内容，超过完整读取阈值时改用 '
                    'read_file_range，参数: {"path": "相对路径"}'
                ),
                function=self.read_file,
            ),
            "read_file_range": ToolSpec(
                name="read_file_range",
                description=(
                    '读取 repo 内 UTF-8 文本文件的闭区间行范围并返回带行号内容，'
                    '参数: {"path": "相对路径", "start_line": 1, "end_line": 10}'
                ),
                function=self.read_file_range,
            ),
            "search_text": ToolSpec(
                name="search_text",
                description=(
                    '在 repo 内按字面量搜索文本关键字，返回文件名、行号和带行号上下文，参数: '
                    '{"keyword": "关键字", "path_glob": "src/*.py", "max_results": 20, "context_lines": 2}'
                ),
                function=self.search_text,
            ),
            "build_repo_index": ToolSpec(
                name="build_repo_index",
                description='构建或读取项目文件索引，参数: {"force": false}，force 为 true 时重新扫描',
                function=self.build_repo_index,
            ),
            "inspect_repo": ToolSpec(
                name="inspect_repo",
                description="生成紧凑项目概览，无需参数；自动读取或构建项目索引",
                function=self.inspect_repo,
            ),
            "propose_patch": ToolSpec(
                name="propose_patch",
                description=(
                    '提交统一差异补丁草稿，参数: {"instruction": "修改说明", "diff": "unified diff"}，'
                    "保存为 pending_approval，返回字段包括 ok、patch_id、patch_path、diff_preview、"
                    "warnings、paths、target_files、next_commands、errors"
                ),
                function=self.propose_patch,
            ),
            "apply_patch": ToolSpec(
                name="apply_patch",
                description='按 patch_id 确定性应用已保存的 pending_approval 补丁，参数: {"patch_id": "patch ID"}',
                function=self.apply_patch,
            ),
            "list_patches": ToolSpec(
                name="list_patches",
                description="List deterministic pending patch records; no arguments",
                function=self.list_patches,
            ),
            "show_patch": ToolSpec(
                name="show_patch",
                description='Show saved patch metadata and diff preview, args: {"patch_id": "patch ID", "full": false}',
                function=self.show_patch,
            ),
            "reject_patch": ToolSpec(
                name="reject_patch",
                description='Reject a pending patch without modifying files, args: {"patch_id": "patch ID"}',
                function=self.reject_patch,
            ),
            "run_tests": ToolSpec(
                name="run_tests",
                description='执行白名单测试命令，参数: {"command_name": "unit" 或 "compile"}',
                function=self.run_tests,
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
        if file_size > MAX_FULL_READ_BYTES:
            raise ToolError(
                "文件超过 read_file 完整读取字节阈值，"
                f"请改用 read_file_range；当前 {file_size} bytes，"
                f"阈值 {MAX_FULL_READ_BYTES} bytes，行阈值 {MAX_FULL_READ_LINES} 行"
            )

        self._validate_readable_file(target_file)

        try:
            file_text = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(f"文件不是有效的 UTF-8 文本: {path}") from error

        file_lines = file_text.splitlines()
        if len(file_lines) > MAX_FULL_READ_LINES:
            raise ToolError(
                "文件超过 read_file 完整读取行数阈值，"
                f"请改用 read_file_range；当前 {len(file_lines)} 行，"
                f"行阈值 {MAX_FULL_READ_LINES} 行，字节阈值 {MAX_FULL_READ_BYTES} bytes"
            )

        return "\n".join(self._format_numbered_lines(file_lines, start_line=1))

    def read_file_range(self, path: str, start_line: int, end_line: int) -> str:
        """读取 repo 内 UTF-8 文本文件的闭区间行范围。

        Args:
            path: repo 内文件路径。
            start_line: 起始行号，从 1 开始。
            end_line: 结束行号，包含该行。

        Returns:
            带行号的指定范围文本内容。起始行超过文件末尾时返回空字符串。
        """

        if not isinstance(start_line, int) or isinstance(start_line, bool):
            raise ToolError("start_line 必须是整数")
        if not isinstance(end_line, int) or isinstance(end_line, bool):
            raise ToolError("end_line 必须是整数")
        if start_line < 1:
            raise ToolError("start_line 必须大于等于 1")
        if end_line < start_line:
            raise ToolError("end_line 必须大于等于 start_line")
        requested_line_count = end_line - start_line + 1
        if requested_line_count > MAX_RANGE_READ_LINES:
            raise ToolError(
                f"read_file_range 单次最多读取 {MAX_RANGE_READ_LINES} 行，"
                f"当前请求 {requested_line_count} 行"
            )

        target_file = self._resolve_repo_path(path)
        if not target_file.is_file():
            raise ToolError(f"path 不是文件: {self._format_repo_path(target_file)}")

        self._validate_readable_file(target_file)

        try:
            selected_lines: list[str] = []
            last_line_number = 0
            with target_file.open("r", encoding="utf-8") as file_stream:
                for line_number, raw_line in enumerate(file_stream, start=1):
                    last_line_number = line_number
                    if line_number < start_line:
                        continue
                    if line_number > end_line:
                        break
                    selected_lines.append(raw_line.rstrip("\r\n"))
        except UnicodeDecodeError as error:
            raise ToolError(f"文件不是有效的 UTF-8 文本: {path}") from error

        if end_line > last_line_number:
            raise ToolError(
                f"end_line 超过文件总行数: {end_line} > {last_line_number}"
            )

        return "\n".join(
            self._format_numbered_lines(selected_lines, start_line=start_line)
        )

    def search_text(
        self,
        keyword: str,
        path_glob: str | None = None,
        max_results: int = MAX_SEARCH_RESULTS,
        context_lines: int = 2,
    ) -> list[str]:
        """在 repo 内搜索关键字。

        Args:
            keyword: 要搜索的关键字。
            path_glob: 可选 repo 相对路径 glob。
            max_results: 最大返回命中数。
            context_lines: 命中行上下文行数。

        Returns:
            命中列表，包含文件名、行号和附近带行号代码片段。
        """

        normalized_keyword = keyword.strip()
        if not normalized_keyword:
            raise ToolError("keyword 不能为空")
        normalized_path_glob = self._normalize_search_path_glob(path_glob)
        normalized_max_results = self._normalize_search_max_results(max_results)
        normalized_context_lines = self._normalize_search_context_lines(context_lines)

        matches: list[str] = []
        output_chars = 0
        for file_path in self._iter_searchable_files():
            if not self._matches_search_path_glob(file_path, normalized_path_glob):
                continue
            lines = self._read_text_lines(file_path)
            if lines is None:
                continue

            for line_number, line_text in enumerate(lines, start=1):
                if normalized_keyword not in line_text:
                    continue

                snippet_start = max(line_number - normalized_context_lines, 1)
                snippet_end = min(line_number + normalized_context_lines, len(lines))
                snippet_lines = lines[snippet_start - 1:snippet_end]
                formatted_snippet = self._format_numbered_lines(
                    snippet_lines,
                    start_line=snippet_start,
                )
                formatted_match = "\n".join(
                    [
                        f"{self._format_repo_path(file_path)}:{line_number}",
                        *formatted_snippet,
                    ]
                )
                next_output_chars = output_chars + len(formatted_match)
                if next_output_chars > MAX_TOOL_OUTPUT_CHARS:
                    self._append_search_truncation_marker(matches)
                    return matches
                matches.append(formatted_match)
                output_chars = next_output_chars
                if len(matches) >= normalized_max_results:
                    return matches

        return matches

    def build_repo_index(self, force: bool = False) -> dict[str, object]:
        """构建或读取 repo 文件索引。"""

        if not isinstance(force, bool):
            raise ToolError("force 必须是布尔值")

        repo_id = hashlib.sha256(str(self.repo_root.resolve()).encode("utf-8")).hexdigest()[:16]
        index_path = self.repopilot_root / "index" / repo_id / "file_index.json"
        if not force and index_path.is_file():
            cached_index = self._read_repo_index(index_path)
            files = cached_index.get("files", [])
            file_count = len(files) if isinstance(files, list) else 0
            return {
                "ok": True,
                "status": "cached",
                "repo_id": repo_id,
                "index_path": self._format_repo_path(index_path),
                "file_count": file_count,
            }

        file_records = [
            self._build_repo_index_file_record(file_path)
            for file_path in self._iter_repo_index_files()
        ]
        repo_index = {
            "repo_id": repo_id,
            "files": file_records,
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(repo_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "status": "built",
            "repo_id": repo_id,
            "index_path": self._format_repo_path(index_path),
            "file_count": len(file_records),
        }

    def inspect_repo(self) -> dict[str, object]:
        """返回基于项目索引的紧凑项目概览。"""

        index_result = self.build_repo_index(force=False)
        index_path = self.repo_root / str(index_result["index_path"])
        repo_index = self._read_repo_index(index_path)
        file_records = self._get_repo_index_file_records(repo_index)
        python_file_records = [
            file_record
            for file_record in file_records
            if file_record.get("extension") == ".py"
        ]

        overview: dict[str, object] = {
            "ok": True,
            "index_status": index_result["status"],
            "index_path": index_result["index_path"],
            "file_count": len(file_records),
            "python_file_count": len(python_file_records),
            "main_python_modules": self._select_main_python_modules(python_file_records),
            "test_files": self._select_test_files(file_records),
            "entrypoint_candidates": self._select_entrypoint_candidates(python_file_records),
            "large_files": self._select_large_files(file_records),
        }
        return self._fit_inspect_repo_output(overview)

    def propose_patch(
        self,
        instruction: str,
        diff: str,
        run_id: str = "",
        task: str = "",
        summary: str = "",
        plan_snapshot: object | None = None,
        risk_level: str = "unknown",
    ) -> dict[str, object]:
        """提交统一差异补丁草稿，不修改目标文件。"""

        errors = self._validate_patch_contract_v03(instruction, diff)
        response: dict[str, object] = {
            "ok": False,
            "patch_id": "",
            "patch_path": "",
            "diff_preview": "",
            "warnings": [],
            "paths": [],
            "target_files": [],
            "next_commands": [],
            "errors": errors,
        }
        if errors:
            return response

        try:
            touched_paths = self.validate_unified_diff(diff)
        except ToolError as error:
            response["errors"] = [str(error)]
            return response

        warnings = [
            "该补丁已保存为文件，需人工确认后再手动应用。",
            "建议先在非生产环境验证补丁行为后再落盘。",
        ]
        patch_preview = self._generate_diff_preview(diff)
        patch_id = self._generate_patch_id(instruction=instruction, patch_text=diff)
        target_files = self._build_patch_target_files(touched_paths)
        diff_path = self._format_repo_path(self._get_new_patch_file_path(patch_id))
        diff_bytes = diff.encode("utf-8")
        patch_metadata = self._create_patch_metadata(
            patch_id=patch_id,
            instruction=instruction,
            paths=touched_paths,
            target_files=target_files,
            diff_path=diff_path,
            diff_sha256=hashlib.sha256(diff_bytes).hexdigest(),
            run_id=run_id,
            task=task,
            summary=summary,
            plan_snapshot=plan_snapshot,
            risk_level=risk_level,
            warnings=warnings,
        )

        patch_path: Path | None = None
        try:
            patch_path = self._write_patch_file(patch_id=patch_id, patch_text=diff)
            self._write_patch_metadata(patch_metadata)
            self._append_run_event(
                patch_id=patch_id,
                event_type="proposed",
                status="saved",
                details={"paths": touched_paths, "target_files": target_files},
            )
        except (OSError, ValueError) as error:
            self._remove_patch_storage(patch_id)
            response["errors"] = [f"保存补丁失败: {error}"]
            return response

        return {
            "ok": True,
            "patch_id": patch_id,
            "patch_path": self._format_repo_path(patch_path),
            "diff_preview": patch_preview,
            "warnings": warnings,
            "paths": touched_paths,
            "target_files": target_files,
            "next_commands": [
                f"inspect patch {patch_id}",
                f"approve patch {patch_id}",
                f"apply patch {patch_id}",
            ],
            "errors": [],
        }

    def validate_unified_diff(self, diff_text: str) -> list[str]:
        """校验受支持的 unified diff，并返回被触及的 repo 相对路径。"""

        if not isinstance(diff_text, str) or not diff_text.strip():
            raise ToolError("diff_text 必须是非空字符串")

        touched_paths: list[str] = []
        seen_paths: set[str] = set()
        lines = diff_text.splitlines()
        line_index = 0
        while line_index < len(lines):
            # 每个文件段都必须从 `diff --git a/... b/...` 开始；这样后续 ---/+++
            # 与 hunk 校验都能绑定到同一个目标路径，避免“头部指向 A、内容修改 B”的补丁。
            current_line = lines[line_index]
            self._reject_unsupported_diff_line(current_line)
            diff_header_match = DIFF_GIT_HEADER_PATTERN.match(current_line)
            if diff_header_match is None:
                raise ToolError(f"diff 缺少合法文件头: line {line_index + 1}")

            old_git_path = self._parse_prefixed_diff_path(
                diff_header_match.group(1),
                "a",
                line_index + 1,
            )
            new_git_path = self._parse_prefixed_diff_path(
                diff_header_match.group(2),
                "b",
                line_index + 1,
            )
            if old_git_path != new_git_path:
                raise ToolError("不支持 rename/copy 或跨路径补丁")
            self._validate_diff_target_path(old_git_path, must_exist=False)
            self._validate_diff_target_path(new_git_path, must_exist=False)

            line_index += 1
            if line_index < len(lines):
                self._reject_unsupported_diff_line(lines[line_index])
            if line_index + 1 >= len(lines):
                raise ToolError("diff 文件头后缺少 ---/+++ 头")

            old_file_path = self._parse_file_diff_header(lines[line_index], "---", line_index + 1)
            line_index += 1
            new_file_path = self._parse_file_diff_header(lines[line_index], "+++", line_index + 1)
            line_index += 1

            if new_file_path is None:
                raise ToolError("不支持删除文件")
            if old_file_path is None:
                if new_file_path != new_git_path:
                    raise ToolError("新文件路径必须与 diff --git 目标一致")
                normalized_path = self._validate_diff_target_path(new_file_path, must_exist=False)
                if normalized_path.is_file():
                    raise ToolError(f"新文件目标已存在: {new_file_path}")
            else:
                if old_file_path != old_git_path or new_file_path != new_git_path:
                    raise ToolError("---/+++ 路径必须与 diff --git 路径一致")
                if old_file_path != new_file_path:
                    raise ToolError("不支持 rename/copy 或跨路径补丁")
                normalized_path = self._validate_diff_target_path(new_file_path, must_exist=True)

            hunk_count = 0
            while line_index < len(lines):
                hunk_line = lines[line_index]
                self._reject_unsupported_diff_line(hunk_line)
                if hunk_line.startswith("diff --git "):
                    break
                hunk_header_match = DIFF_HUNK_HEADER_PATTERN.match(hunk_line)
                if hunk_header_match is None:
                    raise ToolError(f"hunk 头格式非法: line {line_index + 1}")
                expected_old_count = int(hunk_header_match.group(2) or "1")
                expected_new_count = int(hunk_header_match.group(4) or "1")
                hunk_count += 1
                line_index += 1
                actual_old_count = 0
                actual_new_count = 0

                # hunk body 的 old/new 计数是补丁可信度的第一道内容校验：
                # 只接受上下文、删除、新增和 “\ No newline” 标记，拒绝任何未定义行类型。
                while line_index < len(lines):
                    body_line = lines[line_index]
                    self._reject_unsupported_diff_line(body_line)
                    if body_line.startswith("diff --git ") or body_line.startswith("@@ "):
                        break
                    if not body_line.startswith((" ", "+", "-", "\\")):
                        raise ToolError(f"hunk 内容行格式非法: line {line_index + 1}")
                    if body_line.startswith(" "):
                        actual_old_count += 1
                        actual_new_count += 1
                    elif body_line.startswith("-"):
                        actual_old_count += 1
                    elif body_line.startswith("+"):
                        actual_new_count += 1
                    line_index += 1

                if actual_old_count != expected_old_count or actual_new_count != expected_new_count:
                    raise ToolError(f"hunk body count mismatch: line {line_index + 1}")

            if hunk_count == 0:
                raise ToolError("diff 文件段缺少 hunk")

            repo_path = self._format_repo_path(normalized_path)
            if repo_path not in seen_paths:
                touched_paths.append(repo_path)
                seen_paths.add(repo_path)

        return touched_paths

    def list_patches(self) -> dict[str, object]:
        """读取 patches 目录中的元数据清单（仅审计视图）。"""

        patches: list[dict[str, object]] = []
        for patch_dir in sorted(self.repopilot_patches_dir.iterdir(), key=lambda item: item.name):
            if not patch_dir.is_dir():
                continue
            try:
                self._validate_patch_id(patch_dir.name)
                metadata = self._read_new_patch_metadata(patch_dir.name)
            except (OSError, ToolError):
                continue
            patches.append(
                {
                    "patch_id": metadata.get("patch_id", patch_dir.name),
                    "status": metadata.get("status", "unknown"),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("updated_at", ""),
                    "summary": metadata.get("summary", ""),
                    "paths": metadata.get("paths", []),
                    "target_files": metadata.get("target_files", []),
                    "diff_sha256": metadata.get("diff_sha256", ""),
                }
            )
        return {"ok": True, "patches": patches, "errors": []}

    def show_patch(self, patch_id: str, full: bool = False) -> dict[str, object]:
        """读取单个 patch 的 metadata 与 diff（可选 full）。"""

        response: dict[str, object] = {
            "ok": False,
            "patch_id": patch_id if isinstance(patch_id, str) else "",
            "metadata": {},
            "diff_preview": "",
            "diff": "",
            "full": bool(full),
            "errors": [],
        }
        try:
            self._validate_patch_id(patch_id)
            metadata = self._read_new_patch_metadata(patch_id)
            patch_text = self._read_new_patch_file(patch_id)
            self.validate_unified_diff(patch_text)
        except ToolError as error:
            response["errors"] = [str(error)]
            return response

        response["ok"] = True
        response["metadata"] = metadata
        if full:
            response["diff"] = patch_text
        else:
            response["diff_preview"] = self._generate_diff_preview(patch_text)
        return response

    def reject_patch(self, patch_id: str) -> dict[str, object]:
        """拒绝未应用 patch，不触碰业务文件。"""

        response: dict[str, object] = {
            "ok": False,
            "patch_id": patch_id if isinstance(patch_id, str) else "",
            "status": "rejected",
            "errors": [],
        }
        try:
            self._validate_patch_id(patch_id)
            metadata = self._read_new_patch_metadata(patch_id)
            self._validate_pending_patch_status(metadata)
        except ToolError as error:
            response["errors"] = [str(error)]
            return response

        rejected_at = datetime.now().astimezone().isoformat()
        metadata["status"] = "rejected"
        metadata["rejected_at"] = rejected_at
        metadata["updated_at"] = rejected_at
        self._write_patch_metadata(metadata)
        self._append_run_event(patch_id, "reject", "rejected", {})
        response["ok"] = True
        return response

    def apply_patch(self, patch_id: str) -> dict[str, object]:
        """应用待审批补丁。

执行顺序固定为：元数据校验 -> patch 校验 -> 备份 -> 写入 -> run_tests 验证 ->
失败时回滚。状态与回滚结果都写入 metadata 和 runs 事件日志。
"""

        response: dict[str, object] = {
            "ok": False,
            "patch_id": patch_id if isinstance(patch_id, str) else "",
            "status": "rejected",
            "modified_files": [],
            "backup_dir": "",
            "new_files": [],
            "verification_results": [],
            "rollback": {"attempted": False, "ok": None, "errors": [], "files": []},
            "errors": [],
        }

        try:
            # 写文件前集中完成全部“可提前判断”的校验：ID、状态、diff hash、
            # 目标路径集合和当前文件 hash。任何一项失败都保持业务文件零改动。
            self._validate_patch_id(patch_id)
            metadata = self._read_new_patch_metadata(patch_id)
            self._validate_apply_metadata(patch_id, metadata)
            patch_text = self._read_new_patch_file(patch_id)
            self._validate_patch_diff_sha256(patch_text, metadata)
            touched_paths = self.validate_unified_diff(patch_text)
            self._validate_apply_target_files(metadata, touched_paths)
            self._validate_apply_target_file_state(metadata)
            patched_files = self._build_patched_files(patch_text)
        except ToolError as error:
            response["errors"] = [str(error)]
            return response

        backup_dir = self.repopilot_backups_dir / patch_id
        existing_files: list[Path] = []
        new_file_paths: list[Path] = []
        for patched_file in patched_files:
            target_path = self._patched_target_path(patched_file)
            if patched_file["existed_before"]:
                existing_files.append(target_path)
            else:
                new_file_paths.append(target_path)
        new_files = [self._format_repo_path(target_path) for target_path in new_file_paths]
        response["backup_dir"] = self._format_repo_path(backup_dir)
        response["new_files"] = new_files
        response["modified_files"] = touched_paths

        self._append_run_event(patch_id, "apply_start", "started", {"paths": touched_paths})
        written_new_files: list[Path] = []
        written_existing_files: list[Path] = []
        verification_results: list[dict[str, object]] = []
        try:
            self._backup_existing_files(patch_id, existing_files)
            self._write_backup_manifest(patch_id, patched_files, metadata)
            for patched_file in patched_files:
                target_path = self._patched_target_path(patched_file)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if patched_file["existed_before"]:
                    written_existing_files.append(target_path)
                else:
                    written_new_files.append(target_path)
                target_path.write_text(str(patched_file["content"]), encoding="utf-8")
            verification_results = self._run_apply_verification(metadata)
            failed_verifications = [
                verification_result
                for verification_result in verification_results
                if verification_result.get("status") == "failed"
            ]
            if failed_verifications:
                # 验证失败视为 apply 整体失败：即使文件已经写入，也必须回滚到写入前状态。
                raise ToolError("patch verification failed")
        except (OSError, ToolError) as error:
            rollback_result = self._rollback_patch_apply(patch_id, written_existing_files, written_new_files)
            response["status"] = "failed"
            response["rollback"] = rollback_result
            response["errors"] = [str(error)]
            response["verification_results"] = verification_results
            self._mark_patch_apply_failed(patch_id, metadata, touched_paths, new_files, rollback_result, response["verification_results"], str(error))
            self._append_run_event(patch_id, "apply_failure", "failed", {"error": str(error), "rollback": rollback_result})
            return response

        applied_at = datetime.now().astimezone().isoformat()
        metadata["status"] = "applied"
        metadata["applied_at"] = applied_at
        metadata["updated_at"] = applied_at
        metadata["modified_files"] = touched_paths
        metadata["backup_dir"] = self._format_repo_path(backup_dir)
        metadata["new_files"] = new_files
        metadata["verification_results"] = verification_results
        self._write_patch_metadata(metadata)
        self._append_run_event(
            patch_id,
            "apply_success",
            "applied",
            {"paths": touched_paths, "backup_dir": self._format_repo_path(backup_dir), "new_files": new_files},
        )
        response["ok"] = True
        response["status"] = "applied"
        response["verification_results"] = verification_results
        response["rollback"] = {"attempted": False, "ok": None, "errors": [], "files": []}
        return response

    def _validate_apply_metadata(self, patch_id: str, metadata: dict[str, object]) -> None:
        """检查 metadata 与 patch_id 的绑定关系，防止替换/篡改。

必须满足 metadata.patch_id 一致、状态 pending_approval、diff 路径匹配。
"""
        if metadata.get("patch_id") != patch_id:
            raise ToolError("metadata.patch_id does not match patch_id")
        self._validate_pending_patch_status(metadata)
        expected_diff_path = self._format_repo_path(self._get_new_patch_file_path(patch_id))
        if metadata.get("diff_path") != expected_diff_path:
            raise ToolError("metadata.diff_path does not match patch_id storage path")
        target_files = metadata.get("target_files")
        has_target_files = isinstance(target_files, list) and all(
            isinstance(target_file, dict) and isinstance(target_file.get("path"), str)
            for target_file in target_files
        )
        if not has_target_files:
            raise ToolError("metadata.target_files must describe patch targets")

    def _validate_pending_patch_status(self, metadata: dict[str, object]) -> None:
        status = metadata.get("status")
        if status == "applied":
            raise ToolError("patch is already applied")
        if status != "pending_approval":
            raise ToolError("only status=pending_approval patches can be applied or rejected")

    def _validate_patch_diff_sha256(self, patch_text: str, metadata: dict[str, object]) -> None:
        expected_diff_sha256 = metadata.get("diff_sha256")
        if not isinstance(expected_diff_sha256, str) or not expected_diff_sha256:
            raise ToolError("metadata.diff_sha256 must be a non-empty string")
        actual_diff_sha256 = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        if actual_diff_sha256 != expected_diff_sha256:
            raise ToolError("patch.diff sha256 does not match metadata.diff_sha256")

    def _validate_apply_target_files(self, metadata: dict[str, object], touched_paths: list[str]) -> None:
        target_files = metadata.get("target_files")
        if not isinstance(target_files, list):
            raise ToolError("metadata.target_files must be a list")
        metadata_paths: list[str] = []
        for target_file in target_files:
            if not isinstance(target_file, dict) or not isinstance(target_file.get("path"), str):
                raise ToolError("metadata.target_files must contain path strings")
            metadata_paths.append(str(target_file["path"]))
        if set(metadata_paths) != set(touched_paths) or len(metadata_paths) != len(touched_paths):
            raise ToolError("metadata.target_files paths do not match patch.diff targets")

    def _validate_apply_target_file_state(self, metadata: dict[str, object]) -> None:
        """校验每个 target_file 的操作类型与当前文件状态匹配。"""
        target_files = metadata.get("target_files")
        if not isinstance(target_files, list):
            raise ToolError("metadata.target_files must be a list")

        for target_file in target_files:
            if not isinstance(target_file, dict):
                raise ToolError("metadata.target_files entries must be objects")
            target_path_text = target_file.get("path")
            if not isinstance(target_path_text, str) or not target_path_text.strip():
                raise ToolError("metadata.target_files entries must include path")
            operation = target_file.get("operation")
            existed_before = target_file.get("existed_before")
            sha256_before = target_file.get("sha256_before")
            target_path = self._validate_diff_target_path(target_path_text, must_exist=False)

            if operation == "modify":
                # modify 必须命中“创建补丁时看到的同一个文件版本”；hash 不一致说明用户
                # 或其他进程已改过目标文件，继续写入会覆盖未知变更。
                if existed_before is not True:
                    raise ToolError(f"metadata target state mismatch: {target_path_text} modify must have existed_before=true")
                if not isinstance(sha256_before, str) or not sha256_before:
                    raise ToolError(f"metadata target state mismatch: {target_path_text} modify requires sha256_before")
                if not target_path.is_file():
                    raise ToolError(f"metadata target state mismatch: {target_path_text} modify target is missing")
                current_sha256 = self._sha256_file(target_path)
                if current_sha256 != sha256_before:
                    raise ToolError(f"metadata target state mismatch: {target_path_text} sha256_before does not match current file")
            elif operation == "create":
                # create 只允许目标完全不存在，避免把“新增文件补丁”变成覆盖已有文件。
                if existed_before is not False:
                    raise ToolError(f"metadata target state mismatch: {target_path_text} create must have existed_before=false")
                if sha256_before is not None:
                    raise ToolError(f"metadata target state mismatch: {target_path_text} create requires sha256_before=null")
                if target_path.exists():
                    raise ToolError(f"metadata target state mismatch: {target_path_text} create target already exists")
            else:
                raise ToolError(f"metadata target state mismatch: {target_path_text} unsupported operation")

    def _patched_target_path(self, patched_file: dict[str, object]) -> Path:
        target_path = patched_file.get("target_path")
        if not isinstance(target_path, Path):
            raise ToolError("invalid patched target")
        return target_path

    def _run_apply_verification(self, metadata: dict[str, object]) -> list[dict[str, object]]:
        command_names = self._extract_plan_test_command_names(metadata.get("plan_snapshot"))
        if not command_names:
            # 没有随补丁保存测试计划时显式记录 skipped，避免调用方把“无验证”误读成通过。
            return [{"command_name": None, "status": "skipped", "reason": "no test_commands"}]

        verification_results: list[dict[str, object]] = []
        for command_name in command_names:
            run_result = self.run_tests(command_name)
            exit_code = run_result.get("exit_code")
            timed_out = run_result.get("timed_out")
            verification_status = "ok" if exit_code == 0 and timed_out is False else "failed"
            verification_results.append(
                {
                    "command_name": command_name,
                    "status": verification_status,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "stdout": run_result.get("stdout", ""),
                    "stderr": run_result.get("stderr", ""),
                }
            )
        return verification_results

    def _extract_plan_test_command_names(self, plan_snapshot: object) -> list[str]:
        if not isinstance(plan_snapshot, dict):
            return []
        test_commands = plan_snapshot.get("test_commands")
        if not isinstance(test_commands, list):
            return []

        command_names: list[str] = []
        for test_command in test_commands:
            if isinstance(test_command, str):
                command_name = test_command
            elif isinstance(test_command, dict):
                raw_command_name = test_command.get("command_name", test_command.get("name"))
                if not isinstance(raw_command_name, str):
                    raise ToolError("plan_snapshot.test_commands entries must include command_name")
                command_name = raw_command_name
            else:
                raise ToolError("plan_snapshot.test_commands entries must be strings or objects")
            command_names.append(self._validate_run_command_name(command_name))
        return command_names

    def _mark_patch_apply_failed(
        self,
        patch_id: str,
        metadata: dict[str, object],
        touched_paths: list[str],
        new_files: list[str],
        rollback_result: dict[str, object],
        verification_results: object,
        error_message: str,
    ) -> None:
        failed_at = datetime.now().astimezone().isoformat()
        metadata["status"] = "failed"
        metadata["updated_at"] = failed_at
        metadata["failed_at"] = failed_at
        metadata["modified_files"] = touched_paths
        metadata["backup_dir"] = self._format_repo_path(self.repopilot_backups_dir / patch_id)
        metadata["new_files"] = new_files
        metadata["rollback"] = rollback_result
        metadata["verification_results"] = verification_results if isinstance(verification_results, list) else []
        metadata["failure_error"] = error_message
        self._write_patch_metadata(metadata)

    def _build_patched_files(self, patch_text: str) -> list[dict[str, object]]:
        diff_lines = patch_text.splitlines()
        line_index = 0
        patched_files: list[dict[str, object]] = []
        while line_index < len(diff_lines):
            # 这里不再只做格式校验，而是按 hunk 游标真正重建目标文件内容；
            # 后续写盘只使用该重建结果，确保 apply 与 validate 使用同一套路径/内容规则。
            diff_header_match = DIFF_GIT_HEADER_PATTERN.match(diff_lines[line_index])
            if diff_header_match is None:
                raise ToolError(f"diff missing valid file header: line {line_index + 1}")
            new_git_path = self._parse_prefixed_diff_path(diff_header_match.group(2), "b", line_index + 1)
            line_index += 1
            old_file_path = self._parse_file_diff_header(diff_lines[line_index], "---", line_index + 1)
            line_index += 1
            new_file_path = self._parse_file_diff_header(diff_lines[line_index], "+++", line_index + 1)
            line_index += 1
            if new_file_path is None:
                raise ToolError("file deletion is not supported")
            if new_file_path != new_git_path:
                raise ToolError("+++ path must match diff --git target")
            target_path = self._validate_diff_target_path(new_file_path, must_exist=old_file_path is not None)
            existed_before = old_file_path is not None
            trailing_newline = True
            if existed_before:
                original_text = target_path.read_text(encoding="utf-8")
                original_lines = original_text.splitlines()
                trailing_newline = original_text.endswith("\n")
            else:
                original_lines = []
            patched_lines: list[str] = []
            source_cursor = 0
            while line_index < len(diff_lines):
                hunk_header_match = DIFF_HUNK_HEADER_PATTERN.match(diff_lines[line_index])
                if hunk_header_match is None:
                    break
                old_start = int(hunk_header_match.group(1))
                old_count = int(hunk_header_match.group(2) or "1")
                new_count = int(hunk_header_match.group(4) or "1")
                hunk_source_index = 0 if old_start == 0 else old_start - 1
                if hunk_source_index < source_cursor:
                    raise ToolError(f"hunk order overlaps: {new_file_path}")
                patched_lines.extend(original_lines[source_cursor:hunk_source_index])
                source_cursor = hunk_source_index
                line_index += 1
                removed_count = 0
                added_count = 0
                while line_index < len(diff_lines):
                    body_line = diff_lines[line_index]
                    if body_line.startswith("diff --git ") or body_line.startswith("@@ "):
                        break
                    if body_line.startswith("\\"):
                        line_index += 1
                        continue
                    marker = body_line[:1]
                    body_text = body_line[1:]
                    if marker == " ":
                        # 上下文行必须逐字匹配当前文件；这比只相信 hunk 行号更安全，
                        # 能阻止过期补丁在相同行号但不同内容上被误应用。
                        self._assert_hunk_source_line(original_lines, source_cursor, body_text, new_file_path)
                        patched_lines.append(body_text)
                        source_cursor += 1
                        removed_count += 1
                        added_count += 1
                    elif marker == "-":
                        # 删除行也必须逐字匹配，确保删除的是补丁作者实际看到的文本。
                        self._assert_hunk_source_line(original_lines, source_cursor, body_text, new_file_path)
                        source_cursor += 1
                        removed_count += 1
                    elif marker == "+":
                        patched_lines.append(body_text)
                        added_count += 1
                    else:
                        raise ToolError(f"invalid hunk body line: {new_file_path}")
                    line_index += 1
                if removed_count != old_count:
                    raise ToolError(f"hunk line count mismatch: {new_file_path}")
                if added_count != new_count:
                    raise ToolError(f"hunk line count mismatch: {new_file_path}")
            patched_lines.extend(original_lines[source_cursor:])
            patched_text = "\n".join(patched_lines)
            if patched_lines and trailing_newline:
                patched_text += "\n"
            patched_files.append({"target_path": target_path, "content": patched_text, "existed_before": existed_before})
        return patched_files

    def _assert_hunk_source_line(self, original_lines: list[str], source_index: int, expected_line: str, repo_path: str) -> None:
        if source_index >= len(original_lines):
            raise ToolError(f"hunk source line is out of range: {repo_path}")
        actual_line = original_lines[source_index]
        if actual_line != expected_line:
            raise ToolError(f"hunk content mismatch: {repo_path} line {source_index + 1}")

    def _backup_existing_files(self, patch_id: str, existing_files: list[Path]) -> Path:
        """写入现有目标文件的备份副本：

备份目录为 .repopilot/backups/<patch_id>，路径保持 repo 相对结构。
"""
        backup_dir = self.repopilot_backups_dir / patch_id
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=False)
        for source_path in existing_files:
            backup_path = backup_dir / source_path.relative_to(self.repo_root)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_path)
        return backup_dir

    def _write_backup_manifest(
        self,
        patch_id: str,
        patched_files: list[dict[str, object]],
        metadata: dict[str, object],
    ) -> Path:
        """记录本次 apply 的目标清单（含新建/修改标记）用于回滚审计。"""
        backup_dir = self.repopilot_backups_dir / patch_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        manifest_files: list[dict[str, object]] = []
        for patched_file in patched_files:
            target_path = self._patched_target_path(patched_file)
            manifest_files.append(
                {
                    "path": self._format_repo_path(target_path),
                    "existed_before": bool(patched_file.get("existed_before")),
                }
            )
        manifest = {
            "patch_id": patch_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "diff_sha256": metadata.get("diff_sha256", ""),
            "files": manifest_files,
        }
        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def _rollback_patch_apply(self, patch_id: str, existing_files: list[Path], new_files: list[Path]) -> dict[str, object]:
        """回滚 apply：恢复备份文件 + 删除新建文件。"""
        rollback_errors: list[str] = []
        rollback_files: list[dict[str, object]] = []
        backup_dir = self.repopilot_backups_dir / patch_id
        # 先恢复原有文件，再删除新增文件：这样即使删除新增文件失败，原有业务文件也已尽量复原。
        for target_path in existing_files:
            backup_path = backup_dir / target_path.relative_to(self.repo_root)
            try:
                if not backup_path.is_file():
                    raise ToolError(f"backup file is missing: {self._format_repo_path(target_path)}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, target_path)
                rollback_files.append({"path": self._format_repo_path(target_path), "action": "restored", "ok": True})
            except (OSError, ToolError) as error:
                error_message = str(error)
                rollback_errors.append(error_message)
                rollback_files.append(
                    {"path": self._format_repo_path(target_path), "action": "restore", "ok": False, "error": error_message}
                )
        # 新建文件按写入逆序删除，便于随后从叶子到根清理空目录。
        for target_path in reversed(new_files):
            try:
                if target_path.exists():
                    target_path.unlink()
                self._remove_empty_parent_dirs(target_path.parent)
                rollback_files.append({"path": self._format_repo_path(target_path), "action": "removed", "ok": True})
            except OSError as error:
                error_message = str(error)
                rollback_errors.append(error_message)
                rollback_files.append(
                    {"path": self._format_repo_path(target_path), "action": "remove", "ok": False, "error": error_message}
                )
        rollback_ok = not rollback_errors
        self._append_run_event(
            patch_id,
            "rollback_success" if rollback_ok else "rollback_failure",
            "ok" if rollback_ok else "failed",
            {"errors": rollback_errors, "files": rollback_files},
        )
        return {"attempted": True, "ok": rollback_ok, "errors": rollback_errors, "files": rollback_files}

    def _remove_empty_parent_dirs(self, start_dir: Path) -> None:
        current_dir = start_dir
        while current_dir != self.repo_root and current_dir.exists():
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent

    def finish(self, answer: str) -> str:
        """输出最终答案。"""

        normalized_answer = answer.strip()
        if not normalized_answer:
            raise ToolError("answer 不能为空")

        return normalized_answer

    def run_tests(self, command_name: str) -> dict[str, object]:
        """执行白名单测试。

明确只允许：unit（python -m unittest discover）、compile（python -m compileall .）。
不接受任意 shell 命令字符串，避免任意命令执行风险。
"""

        normalized_command_name = self._validate_run_command_name(command_name)
        command_map = {
            "unit": [sys.executable, "-m", "unittest", "discover"],
            "compile": [sys.executable, "-m", "compileall", "."],
        }

        command = command_map[normalized_command_name]
        run_id = self._generate_patch_id(f"run_tests:{normalized_command_name}")

        # 每次测试运行也写入 runs 事件日志，便于把补丁应用、验证和回滚证据串起来。
        self._append_run_event(
            patch_id=run_id,
            event_type="run_tests_start",
            status="started",
            details={
                "command_name": normalized_command_name,
                "command": command,
            },
        )

        timed_out = False
        exit_code = -1
        stdout = ""
        stderr = ""

        try:
            completed = subprocess.run(
                command,
                shell=False,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=RUN_TEST_TIMEOUT_SECONDS,
            )
            exit_code = completed.returncode
            stdout = self._ensure_text_output(completed.stdout)
            stderr = self._ensure_text_output(completed.stderr)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = self._ensure_text_output(error.stdout)
            stderr = self._ensure_text_output(error.stderr)

            self._append_run_event(
                patch_id=run_id,
                event_type="run_tests_timeout",
                status="timeout",
                details={
                    "command_name": normalized_command_name,
                    "timeout_seconds": RUN_TEST_TIMEOUT_SECONDS,
                },
            )
        except OSError as error:
            self._append_run_event(
                patch_id=run_id,
                event_type="run_tests_end",
                status="failed",
                details={
                    "command_name": normalized_command_name,
                    "error": str(error),
                },
            )
            raise

        stdout, stdout_truncated = self._truncate_output(
            stdout,
            RUN_TEST_OUTPUT_MAX_BYTES,
        )
        stderr, stderr_truncated = self._truncate_output(
            stderr,
            RUN_TEST_OUTPUT_MAX_BYTES,
        )

        if stdout_truncated or stderr_truncated:
            self._append_run_event(
                patch_id=run_id,
                event_type="run_tests_truncation",
                status="truncated",
                details={
                    "command_name": normalized_command_name,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "max_length": RUN_TEST_OUTPUT_MAX_BYTES,
                },
            )

        end_status = "timeout" if timed_out else ("ok" if exit_code == 0 else "failed")
        self._append_run_event(
            patch_id=run_id,
            event_type="run_tests_end",
            status=end_status,
            details={
                "command_name": normalized_command_name,
                "exit_code": exit_code,
                "timed_out": timed_out,
            },
        )

        return {
            "command_name": normalized_command_name,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }

    def _validate_run_command_name(self, command_name: str) -> str:
        """只接受 unit / compile 两种命令名。"""
        if not isinstance(command_name, str) or not command_name.strip():
            raise ToolError("command_name 必须是非空字符串")

        normalized_command_name = command_name.strip()
        if normalized_command_name not in {"unit", "compile"}:
            raise ToolError("command_name 只支持: unit, compile")

        return normalized_command_name

    def _ensure_text_output(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return str(value)

    def _truncate_output(self, text: str, max_length: int) -> tuple[str, bool]:
        if max_length <= 0:
            return "", True if text else False

        if len(text) <= max_length:
            return text, False

        return text[:max_length], True

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
        """将用户输入路径解析到 repo_root 内（路径沙箱）。"""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path 必须是非空字符串")

        resolved_path = (self.repo_root / path).resolve()
        try:
            resolved_path.relative_to(self.repo_root)
        except ValueError as error:
            raise ToolError(f"path 必须位于 repo 目录内部: {path}") from error

        return resolved_path

    def _ensure_repopilot_storage(self) -> None:
        for storage_dir in (
            self.repopilot_patches_dir,
            self.repopilot_backups_dir,
            self.repopilot_runs_dir,
        ):
            storage_dir.mkdir(parents=True, exist_ok=True)

    def _generate_patch_id(
        self,
        instruction: str,
        patch_text: str = "",
        created_at: datetime | None = None,
    ) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolError("instruction 必须是非空字符串")

        timestamp = created_at or datetime.now().astimezone()
        timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")
        digest_source = f"{timestamp.isoformat()}\n{instruction}\n{patch_text}"
        patch_digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
        return f"{timestamp_text}_{patch_digest}"

    def _create_patch_metadata(
        self,
        patch_id: str,
        instruction: str,
        paths: list[str],
        target_files: list[dict[str, object]] | None = None,
        diff_path: str = "",
        diff_sha256: str = "",
        run_id: str = "",
        task: str = "",
        summary: str = "",
        plan_snapshot: object | None = None,
        risk_level: str = "unknown",
        warnings: list[str] | None = None,
        status: str = "pending_approval",
        created_at: datetime | None = None,
    ) -> dict[str, object]:
        """构建 patch 元数据，作为 apply/回滚/审计的权威输入。"""
        self._validate_patch_id(patch_id)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolError("instruction 必须是非空字符串")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ToolError("paths 必须是字符串列表")
        if warnings is not None and not all(isinstance(warning, str) for warning in warnings):
            raise ToolError("warnings 必须是字符串列表")
        if not isinstance(status, str) or not status.strip():
            raise ToolError("status 必须是非空字符串")
        if target_files is not None and not all(isinstance(target_file, dict) for target_file in target_files):
            raise ToolError("target_files 必须是对象列表")
        if not isinstance(diff_path, str):
            raise ToolError("diff_path 必须是字符串")
        if not isinstance(diff_sha256, str):
            raise ToolError("diff_sha256 必须是字符串")
        if not all(isinstance(value, str) for value in (run_id, task, summary, risk_level)):
            raise ToolError("run_id、task、summary、risk_level 必须是字符串")

        timestamp = created_at or datetime.now().astimezone()
        normalized_target_files = target_files if target_files is not None else []
        return {
            "patch_id": patch_id,
            "created_at": timestamp.isoformat(),
            "instruction": instruction,
            "run_id": run_id,
            "task": task,
            "summary": summary,
            "status": status,
            "paths": paths,
            "target_files": normalized_target_files,
            "diff_path": diff_path,
            "diff_sha256": diff_sha256,
            "approved_at": None,
            "applied_at": None,
            "rejected_at": None,
            "plan_snapshot": plan_snapshot,
            "risk_level": risk_level,
            "warnings": warnings or [],
        }

    def _write_patch_file(self, patch_id: str, patch_text: str) -> Path:
        """持久化统一差异到 .repopilot/patches/<patch_id>/patch.diff。"""
        self._validate_patch_id(patch_id)
        if not isinstance(patch_text, str):
            raise ToolError("patch_text 必须是字符串")

        patch_path = self._get_new_patch_file_path(patch_id)
        patch_path.parent.mkdir(parents=True, exist_ok=False)
        patch_path.write_bytes(patch_text.encode("utf-8"))
        return patch_path

    def _read_patch_file(self, patch_id: str) -> str:
        patch_path = self._get_new_patch_file_path(patch_id)
        if not patch_path.is_file():
            raise ToolError(f"patch 文件不存在: {patch_id}")

        return patch_path.read_text(encoding="utf-8")

    def _read_new_patch_file(self, patch_id: str) -> str:
        patch_path = self._get_new_patch_file_path(patch_id)
        if not patch_path.is_file():
            raise ToolError(f"patch.diff 文件不存在: {patch_id}")
        return patch_path.read_text(encoding="utf-8")

    def _write_patch_metadata(self, metadata: dict[str, object]) -> Path:
        """持久化 patch metadata.json，并采用可审计 JSON 格式。"""
        patch_id = metadata.get("patch_id")
        if not isinstance(patch_id, str):
            raise ToolError("metadata.patch_id 必须是字符串")
        self._validate_patch_id(patch_id)

        metadata_path = self._get_new_patch_metadata_path(patch_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_metadata = json.dumps(metadata, ensure_ascii=False, indent=2)
        metadata_path.write_text(serialized_metadata, encoding="utf-8")
        return metadata_path

    def _read_patch_metadata(self, patch_id: str) -> dict[str, object]:
        metadata_path = self._get_new_patch_metadata_path(patch_id)
        if not metadata_path.is_file():
            raise ToolError(f"metadata 文件不存在: {patch_id}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ToolError(f"metadata 不是有效 JSON: {patch_id}") from error
        if not isinstance(metadata, dict):
            raise ToolError(f"metadata 必须是 JSON 对象: {patch_id}")

        return metadata

    def _read_new_patch_metadata(self, patch_id: str) -> dict[str, object]:
        metadata_path = self._get_new_patch_metadata_path(patch_id)
        if not metadata_path.is_file():
            raise ToolError(f"metadata.json 文件不存在: {patch_id}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ToolError(f"metadata.json 不是有效 JSON: {patch_id}") from error
        if not isinstance(metadata, dict):
            raise ToolError(f"metadata.json 必须是 JSON 对象: {patch_id}")

        return metadata

    def _update_patch_metadata_status(self, patch_id: str, status: str) -> dict[str, object]:
        if not isinstance(status, str) or not status.strip():
            raise ToolError("status 必须是非空字符串")

        metadata = self._read_new_patch_metadata(patch_id)
        metadata["status"] = status
        metadata["updated_at"] = datetime.now().astimezone().isoformat()
        self._write_patch_metadata(metadata)
        return metadata

    def _append_run_event(
        self,
        patch_id: str,
        event_type: str,
        status: str,
        details: dict[str, object] | None = None,
    ) -> Path:
        self._validate_patch_id(patch_id)
        if not isinstance(event_type, str) or not event_type.strip():
            raise ToolError("event_type 必须是非空字符串")
        if not isinstance(status, str) or not status.strip():
            raise ToolError("status 必须是非空字符串")
        if details is not None and not isinstance(details, dict):
            raise ToolError("details 必须是对象")

        event = {
            "patch_id": patch_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "event_type": event_type,
            "status": status,
            "details": details or {},
        }
        run_path = self.repopilot_runs_dir / f"{patch_id}.jsonl"
        with run_path.open("a", encoding="utf-8") as run_file:
            run_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return run_path

    def _get_new_patch_file_path(self, patch_id: str) -> Path:
        self._validate_patch_id(patch_id)
        return self.repopilot_patches_dir / patch_id / "patch.diff"

    def _get_new_patch_metadata_path(self, patch_id: str) -> Path:
        self._validate_patch_id(patch_id)
        return self.repopilot_patches_dir / patch_id / "metadata.json"

    def _remove_patch_storage(self, patch_id: str) -> None:
        self._validate_patch_id(patch_id)
        patch_dir = self.repopilot_patches_dir / patch_id
        if patch_dir.exists():
            shutil.rmtree(patch_dir)

    def _build_patch_target_files(self, touched_paths: list[str]) -> list[dict[str, object]]:
        """生成目标文件快照列表。

包含 existed_before 与 sha256_before，用于后续 apply 时判定 state 是否被改写。
"""
        target_files: list[dict[str, object]] = []
        for touched_path in touched_paths:
            target_path = self._validate_diff_target_path(touched_path, must_exist=False)
            existed_before = target_path.is_file()
            target_files.append(
                {
                    "path": touched_path,
                    "existed_before": existed_before,
                    "sha256_before": self._sha256_file(target_path) if existed_before else None,
                    "operation": "modify" if existed_before else "create",
                }
            )
        return target_files

    def _sha256_file(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_patch_id(self, patch_id: str) -> None:
        if not isinstance(patch_id, str) or not patch_id.strip():
            raise ToolError("patch_id 必须是非空字符串")
        patch_id_parts = patch_id.split("_")
        if len(patch_id_parts) != 3:
            raise ToolError("patch_id 格式无效")
        date_part, time_part, digest_part = patch_id_parts
        if not (date_part.isdigit() and len(date_part) == 8):
            raise ToolError("patch_id 日期格式无效")
        if not (time_part.isdigit() and len(time_part) == 6):
            raise ToolError("patch_id 时间格式无效")
        if len(digest_part) != 12 or any(
            character not in "0123456789abcdef" for character in digest_part
        ):
            raise ToolError("patch_id sha256 摘要格式无效")

    def _validate_patch_contract_v03(
        self,
        instruction: str,
        diff: str,
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(instruction, str) or not instruction.strip():
            errors.append("instruction must be a non-empty string")

        if not isinstance(diff, str):
            errors.append("diff must be a string")
        elif not diff.strip():
            errors.append("diff must not be empty")

        return errors

    def _generate_diff_preview(self, diff_text: str, max_lines: int = 120) -> str:
        diff_lines = diff_text.splitlines()
        if len(diff_lines) <= max_lines:
            return diff_text
        preview_lines = "\n".join(diff_lines[:max_lines])
        return f"{preview_lines}\n...（预览已截断）"

    def _validate_readable_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise ToolError(f"path 不是文件: {self._format_repo_path(file_path)}")
        if self._has_hidden_path_part(file_path):
            raise ToolError(f"禁止读取隐藏路径: {self._format_repo_path(file_path)}")
        if self._is_secret_file(file_path):
            raise ToolError(f"禁止读取密钥或环境文件: {self._format_repo_path(file_path)}")
        if self._is_binary_file(file_path):
            raise ToolError(f"禁止读取二进制文件: {self._format_repo_path(file_path)}")

    def _parse_prefixed_diff_path(self, raw_path: str, expected_prefix: str, line_number: int) -> str:
        prefix = f"{expected_prefix}/"
        if not raw_path.startswith(prefix) or raw_path == prefix:
            raise ToolError(f"diff 路径必须使用 {prefix} 前缀: line {line_number}")
        return raw_path[len(prefix):]

    def _parse_file_diff_header(self, header_line: str, expected_marker: str, line_number: int) -> str | None:
        self._reject_unsupported_diff_line(header_line)
        header_match = DIFF_FILE_HEADER_PATTERN.match(header_line)
        if header_match is None or header_match.group(1) != expected_marker:
            raise ToolError(f"diff 文件头格式非法: line {line_number}")

        raw_path = header_match.group(2)
        if raw_path == "/dev/null":
            return None

        expected_prefix = "a" if expected_marker == "---" else "b"
        return self._parse_prefixed_diff_path(raw_path, expected_prefix, line_number)

    def _validate_diff_target_path(self, repo_path: str, must_exist: bool) -> Path:
        """校验 diff 目标路径边界：禁止绝对路径、路径穿越、隐藏/.git/secret 与 symlink。"""
        if Path(repo_path).is_absolute():
            raise ToolError("diff 路径必须是 repo 内相对路径，不能使用绝对路径")
        if any(path_part == ".." for path_part in Path(repo_path).parts):
            raise ToolError("diff 路径不能包含 ..")

        target_path = self._resolve_repo_path(repo_path)
        self._reject_symlink_path(target_path)
        if self._has_hidden_path_part(target_path):
            raise ToolError(f"禁止修改隐藏路径: {repo_path}")
        if ".git" in target_path.relative_to(self.repo_root).parts:
            raise ToolError(f"禁止修改 .git 路径: {repo_path}")
        if self._is_secret_file(target_path):
            raise ToolError(f"禁止修改密钥或环境文件: {repo_path}")

        if must_exist:
            self._validate_readable_file(target_path)
        elif target_path.exists() and target_path.is_dir():
            raise ToolError(f"diff 目标不能是目录: {repo_path}")

        return target_path

    def _reject_symlink_path(self, target_path: Path) -> None:
        """逐级拒绝路径中间组件为符号链接，避免通过链接改写受限路径。"""
        current_path = self.repo_root
        for path_part in target_path.relative_to(self.repo_root).parts:
            current_path = current_path / path_part
            if current_path.exists() and current_path.is_symlink():
                raise ToolError(f"禁止通过符号链接修改路径: {self._format_repo_path(current_path)}")

    def _reject_unsupported_diff_line(self, diff_line: str) -> None:
        if diff_line.startswith(BINARY_PATCH_MARKERS):
            raise ToolError("不支持二进制补丁")
        if diff_line.startswith(UNSUPPORTED_DIFF_HEADERS):
            raise ToolError("不支持 rename/copy 补丁")

    def _iter_searchable_files(self) -> list[Path]:
        """返回可被 search_text 遍历的文件列表（已应用目录/文件跳过规则）。"""
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

    def _iter_repo_index_files(self) -> list[Path]:
        """返回 repo 索引遍历文件列表，遵循 REPO_INDEX_SKIPPED_DIR_NAMES。"""
        indexed_files: list[Path] = []
        for current_dir, dir_names, file_names in os.walk(self.repo_root, topdown=True):
            dir_names[:] = [
                dir_name
                for dir_name in dir_names
                if dir_name not in REPO_INDEX_SKIPPED_DIR_NAMES
            ]

            current_dir_path = Path(current_dir)
            for file_name in file_names:
                file_path = current_dir_path / file_name
                try:
                    file_path.resolve().relative_to(self.repo_root)
                except (OSError, ValueError):
                    continue
                indexed_files.append(file_path)

        return sorted(indexed_files, key=lambda item: self._format_repo_path(item).lower())

    def _build_repo_index_file_record(self, file_path: Path) -> dict[str, object]:
        file_record: dict[str, object] = {
            "path": self._format_repo_path(file_path),
            "size_bytes": 0,
            "line_count": 0,
            "extension": file_path.suffix.lower(),
            "symbols": [],
        }

        try:
            file_record["size_bytes"] = file_path.stat().st_size
            file_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            file_record["error"] = f"decode error: {error}"
            return file_record
        except OSError as error:
            file_record["error"] = f"os error: {error}"
            return file_record

        file_record["line_count"] = len(file_text.splitlines())
        if file_path.suffix.lower() != ".py":
            return file_record

        try:
            parsed_module = ast.parse(file_text, filename=self._format_repo_path(file_path))
        except SyntaxError as error:
            file_record["error"] = f"syntax error: {error.msg}"
            return file_record

        file_record["symbols"] = self._extract_python_symbols(parsed_module)
        if self._has_python_main_guard(parsed_module):
            file_record["entrypoint_evidence"] = ["main_guard"]
        return file_record

    def _extract_python_symbols(self, parsed_module: ast.Module) -> list[dict[str, object]]:
        symbols: list[dict[str, object]] = []
        for statement in parsed_module.body:
            if isinstance(statement, ast.ClassDef):
                symbols.append({"name": statement.name, "kind": "class", "line": statement.lineno})
                symbols.extend(self._extract_python_method_symbols(statement))
            elif isinstance(statement, ast.AsyncFunctionDef):
                symbols.append({"name": statement.name, "kind": "async_function", "line": statement.lineno})
            elif isinstance(statement, ast.FunctionDef):
                symbols.append({"name": statement.name, "kind": "function", "line": statement.lineno})
        return symbols

    def _extract_python_method_symbols(self, class_node: ast.ClassDef) -> list[dict[str, object]]:
        method_symbols: list[dict[str, object]] = []
        for statement in class_node.body:
            if isinstance(statement, ast.AsyncFunctionDef):
                method_symbols.append({"name": statement.name, "kind": "async_method", "line": statement.lineno})
            elif isinstance(statement, ast.FunctionDef):
                method_symbols.append({"name": statement.name, "kind": "method", "line": statement.lineno})
        return method_symbols

    def _has_python_main_guard(self, parsed_module: ast.Module) -> bool:
        for statement in parsed_module.body:
            if isinstance(statement, ast.If) and self._is_python_main_guard_test(statement.test):
                return True
        return False

    def _is_python_main_guard_test(self, test_node: ast.expr) -> bool:
        if not isinstance(test_node, ast.Compare):
            return False
        if len(test_node.ops) != 1 or not isinstance(test_node.ops[0], ast.Eq):
            return False
        if len(test_node.comparators) != 1:
            return False
        left_text = self._main_guard_literal(test_node.left)
        right_text = self._main_guard_literal(test_node.comparators[0])
        return {left_text, right_text} == {"__name__", "__main__"}

    def _main_guard_literal(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name) and node.id == "__name__":
            return "__name__"
        if isinstance(node, ast.Constant) and node.value == "__main__":
            return "__main__"
        return ""

    def _get_repo_index_file_records(self, repo_index: dict[str, object]) -> list[dict[str, object]]:
        files = repo_index.get("files", [])
        if not isinstance(files, list):
            raise ToolError("项目索引 files 必须是列表")
        return [file_record for file_record in files if isinstance(file_record, dict)]

    def _select_main_python_modules(self, file_records: list[dict[str, object]]) -> list[dict[str, object]]:
        ranked_records = sorted(
            file_records,
            key=lambda file_record: (
                -self._symbol_count(file_record),
                -self._int_record_value(file_record, "size_bytes"),
                str(file_record.get("path", "")),
            ),
        )
        return [self._compact_python_record(file_record) for file_record in ranked_records[:10]]

    def _select_test_files(self, file_records: list[dict[str, object]]) -> list[dict[str, object]]:
        test_records = [
            file_record
            for file_record in file_records
            if self._is_test_file_record(file_record)
        ]
        ranked_records = sorted(test_records, key=lambda file_record: str(file_record.get("path", "")))
        return [self._compact_file_record(file_record) for file_record in ranked_records[:10]]

    def _select_entrypoint_candidates(self, file_records: list[dict[str, object]]) -> list[dict[str, object]]:
        entrypoint_records = [
            file_record
            for file_record in file_records
            if self._is_entrypoint_candidate(file_record)
        ]
        ranked_records = sorted(
            entrypoint_records,
            key=lambda file_record: (
                -len(self._entrypoint_reasons(file_record)),
                str(file_record.get("path", "")),
            ),
        )
        return [
            {
                **self._compact_python_record(file_record),
                "reasons": self._entrypoint_reasons(file_record),
            }
            for file_record in ranked_records[:10]
        ]

    def _select_large_files(self, file_records: list[dict[str, object]]) -> list[dict[str, object]]:
        ranked_records = sorted(
            file_records,
            key=lambda file_record: (
                -self._int_record_value(file_record, "size_bytes"),
                str(file_record.get("path", "")),
            ),
        )
        return [self._compact_large_file_record(file_record) for file_record in ranked_records[:10]]

    def _compact_python_record(self, file_record: dict[str, object]) -> dict[str, object]:
        compact_record = self._compact_file_record(file_record)
        compact_record["symbol_count"] = self._symbol_count(file_record)
        return compact_record

    def _compact_file_record(self, file_record: dict[str, object]) -> dict[str, object]:
        return {
            "path": str(file_record.get("path", "")),
            "size_bytes": self._int_record_value(file_record, "size_bytes"),
            "line_count": self._int_record_value(file_record, "line_count"),
        }

    def _compact_large_file_record(self, file_record: dict[str, object]) -> dict[str, object]:
        size_bytes = self._int_record_value(file_record, "size_bytes")
        line_count = self._int_record_value(file_record, "line_count")
        return {
            "path": str(file_record.get("path", "")),
            "size_bytes": size_bytes,
            "line_count": line_count,
            "exceeds_full_read_threshold": size_bytes > MAX_FULL_READ_BYTES or line_count > MAX_FULL_READ_LINES,
            "exceeds_bytes_threshold": size_bytes > MAX_FULL_READ_BYTES,
            "exceeds_lines_threshold": line_count > MAX_FULL_READ_LINES,
        }

    def _is_test_file_record(self, file_record: dict[str, object]) -> bool:
        path_text = str(file_record.get("path", ""))
        path_lower = path_text.lower()
        return "test" in path_lower or Path(path_text).name.startswith("test_")

    def _is_entrypoint_candidate(self, file_record: dict[str, object]) -> bool:
        return bool(self._entrypoint_reasons(file_record))

    def _entrypoint_reasons(self, file_record: dict[str, object]) -> list[str]:
        path_text = str(file_record.get("path", ""))
        file_name = Path(path_text).name
        reasons: list[str] = []
        if file_name in {"main.py", "app.py"}:
            reasons.append("filename")
        if file_name.startswith("run_") and file_name.endswith(".py"):
            reasons.append("run_filename")
        if self._has_symbol_named(file_record, "main"):
            reasons.append("main_symbol")
        evidence = file_record.get("entrypoint_evidence", [])
        if isinstance(evidence, list) and "main_guard" in evidence:
            reasons.append("main_guard")
        return reasons

    def _has_symbol_named(self, file_record: dict[str, object], symbol_name: str) -> bool:
        symbols = file_record.get("symbols", [])
        if not isinstance(symbols, list):
            return False
        return any(
            isinstance(symbol, dict) and symbol.get("name") == symbol_name
            for symbol in symbols
        )

    def _symbol_count(self, file_record: dict[str, object]) -> int:
        symbols = file_record.get("symbols", [])
        if not isinstance(symbols, list):
            return 0
        return len(symbols)

    def _int_record_value(self, file_record: dict[str, object], key: str) -> int:
        value = file_record.get(key, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _fit_inspect_repo_output(self, overview: dict[str, object]) -> dict[str, object]:
        if self._inspect_repo_output_fits(overview):
            return overview
        compact_overview = dict(overview)
        list_keys = (
            "main_python_modules",
            "test_files",
            "entrypoint_candidates",
            "large_files",
        )
        for list_key in list_keys:
            list_value = compact_overview.get(list_key, [])
            compact_overview[list_key] = list_value if isinstance(list_value, list) else []
        compact_overview["truncated"] = True
        max_list_items = 0
        for list_key in list_keys:
            list_value = compact_overview[list_key]
            if isinstance(list_value, list):
                max_list_items = max(max_list_items, len(list_value))
        for item_limit in range(max_list_items, -1, -1):
            for list_key in list_keys:
                list_value = compact_overview[list_key]
                compact_overview[list_key] = list_value[:item_limit] if isinstance(list_value, list) else []
            if self._inspect_repo_output_fits(compact_overview):
                return compact_overview
        compact_overview["index_path"] = ""
        if self._inspect_repo_output_fits(compact_overview):
            return compact_overview
        return compact_overview

    def _inspect_repo_output_fits(self, overview: dict[str, object]) -> bool:
        serialized_overview = json.dumps(overview, ensure_ascii=False, sort_keys=True)
        return len(serialized_overview) <= MAX_TOOL_OUTPUT_CHARS

    def _read_repo_index(self, index_path: Path) -> dict[str, object]:
        try:
            repo_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ToolError(f"读取项目索引失败: {self._format_repo_path(index_path)}") from error
        if not isinstance(repo_index, dict):
            raise ToolError("项目索引必须是 JSON 对象")
        return repo_index

    def _should_skip_dir(self, dir_name: str) -> bool:
        """判断目录是否进入 os.walk：隐藏目录与 SKIPPED_DIR_NAMES 一律跳过。"""
        return dir_name.startswith(".") or dir_name in SKIPPED_DIR_NAMES

    def _should_skip_file(self, file_path: Path) -> bool:
        """判断单个文件是否参与检索：

条件为：路径越界、隐藏路径、敏感文件、超大文件（非 .py）或二进制。
"""
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
        if file_size > MAX_FILE_BYTES and file_path.suffix.lower() != ".py":
            return True
        return self._is_binary_file(file_path)

    def _read_text_lines(self, file_path: Path) -> list[str] | None:
        try:
            return file_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return None

    def _normalize_search_path_glob(self, path_glob: str | None) -> str | None:
        if path_glob is None:
            return None
        if not isinstance(path_glob, str) or not path_glob.strip():
            raise ToolError("path_glob 必须是非空字符串或 null")
        if Path(path_glob).is_absolute() or ".." in Path(path_glob).parts:
            raise ToolError("path_glob 必须是 repo 内相对 glob")
        return path_glob.strip().replace("\\", "/")

    def _normalize_search_max_results(self, max_results: int) -> int:
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ToolError("max_results 必须是整数")
        if max_results < 1:
            raise ToolError("max_results 必须大于等于 1")
        if max_results > MAX_SEARCH_RESULTS:
            raise ToolError(f"max_results 不能超过 {MAX_SEARCH_RESULTS}")
        return max_results

    def _normalize_search_context_lines(self, context_lines: int) -> int:
        if not isinstance(context_lines, int) or isinstance(context_lines, bool):
            raise ToolError("context_lines 必须是整数")
        if context_lines < 0:
            raise ToolError("context_lines 必须大于等于 0")
        return context_lines

    def _matches_search_path_glob(self, file_path: Path, path_glob: str | None) -> bool:
        if path_glob is None:
            return True
        repo_path = self._format_repo_path(file_path)
        return Path(repo_path).match(path_glob)

    def _append_search_truncation_marker(self, matches: list[str]) -> None:
        if not matches or matches[-1] != "...（结果已截断）":
            matches.append("...（结果已截断）")

    def _format_numbered_lines(self, lines: list[str], start_line: int) -> list[str]:
        return [
            f"{line_number} | {line_text}"
            for line_number, line_text in enumerate(lines, start=start_line)
        ]

    def _has_hidden_path_part(self, file_path: Path) -> bool:
        """判断相对路径中是否含隐藏目录/文件（以 "." 开头）。"""
        relative_parts = file_path.relative_to(self.repo_root).parts
        return any(part.startswith(".") for part in relative_parts)

    def _is_secret_file(self, file_path: Path) -> bool:
        """判断文件名是否命中秘钥/环境文件名单。

命中后在 read/search/patch 阶段统一禁止读取或写入。
"""
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
