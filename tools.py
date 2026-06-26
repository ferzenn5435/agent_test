"""只读代码库工具集合。"""

from __future__ import annotations

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

from config import MAX_FILE_BYTES
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
    """面向单个目标代码库的只读工具。"""

    def __init__(self, repo_root: str | Path) -> None:
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
                description='读取 repo 内 UTF-8 文本文件并返回带行号内容，最大 20KB，参数: {"path": "相对路径"}',
                function=self.read_file,
            ),
            "read_file_range": ToolSpec(
                name="read_file_range",
                description=(
                    '读取 repo 内 UTF-8 文本文件的闭区间行范围并返回带行号内容，最大 20KB，'
                    '参数: {"path": "相对路径", "start_line": 1, "end_line": 10}'
                ),
                function=self.read_file_range,
            ),
            "search_text": ToolSpec(
                name="search_text",
                description='在 repo 内搜索文本关键字，返回文件名、行号和带行号上下文，参数: {"keyword": "关键字"}',
                function=self.search_text,
            ),
            "propose_patch": ToolSpec(
                name="propose_patch",
                description=(
                    '提交统一差异补丁草稿，参数: {"instruction": "修改说明", "diff": "unified diff"}，'
                    '返回: {"ok": bool, "patch_id": str, "patch_path": str, '
                    '"diff_preview": str, "warnings": list[str], "paths": list[str], '
                    '"errors": list[str]}'
                ),
                function=self.propose_patch,
            ),
            "apply_patch": ToolSpec(
                name="apply_patch",
                description='Apply a saved proposed patch, args: {"patch_id": "patch ID"}',
                function=self.apply_patch,
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
        if file_size > MAX_FILE_BYTES:
            raise ToolError(f"文件超过 20KB 限制: {path}")

        self._validate_readable_file(target_file)

        try:
            file_text = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(f"文件不是有效的 UTF-8 文本: {path}") from error

        return "\n".join(self._format_numbered_lines(file_text.splitlines(), start_line=1))

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

        target_file = self._resolve_repo_path(path)
        if not target_file.is_file():
            raise ToolError(f"path 不是文件: {self._format_repo_path(target_file)}")

        file_size = target_file.stat().st_size
        if file_size > MAX_FILE_BYTES:
            raise ToolError(f"文件超过 20KB 限制: {path}")

        self._validate_readable_file(target_file)

        try:
            file_lines = target_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ToolError(f"文件不是有效的 UTF-8 文本: {path}") from error

        if start_line > len(file_lines):
            return ""

        selected_lines = file_lines[start_line - 1:end_line]
        return "\n".join(
            self._format_numbered_lines(selected_lines, start_line=start_line)
        )

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

    def propose_patch(
        self,
        instruction: str,
        diff: str,
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
        patch_metadata = self._create_patch_metadata(
            patch_id=patch_id,
            instruction=instruction,
            paths=touched_paths,
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
                details={"paths": touched_paths},
            )
        except (OSError, ValueError) as error:
            if patch_path is not None and patch_path.exists():
                patch_path.unlink()
            metadata_path = self._get_patch_metadata_path(patch_id)
            if metadata_path.exists():
                metadata_path.unlink()
            response["errors"] = [f"保存补丁失败: {error}"]
            return response

        return {
            "ok": True,
            "patch_id": patch_id,
            "patch_path": self._format_repo_path(patch_path),
            "diff_preview": patch_preview,
            "warnings": warnings,
            "paths": touched_paths,
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

    def apply_patch(self, patch_id: str) -> dict[str, object]:
        """Apply a saved proposed patch with backup and rollback."""

        response: dict[str, object] = {
            "ok": False,
            "patch_id": patch_id if isinstance(patch_id, str) else "",
            "status": "rejected",
            "modified_files": [],
            "backup_dir": "",
            "new_files": [],
            "rollback": {"attempted": False, "ok": None, "errors": []},
            "errors": [],
        }

        try:
            self._validate_patch_id(patch_id)
            metadata = self._read_patch_metadata(patch_id)
            self._validate_apply_metadata(patch_id, metadata)
            patch_text = self._read_patch_file(patch_id)
            touched_paths = self.validate_unified_diff(patch_text)
            patched_files = self._build_patched_files(patch_text)
        except ToolError as error:
            response["errors"] = [str(error)]
            return response

        backup_dir = self.repopilot_backups_dir / patch_id
        existing_files: list[Path] = []
        new_files: list[str] = []
        for patched_file in patched_files:
            target_path = patched_file["target_path"]
            if not isinstance(target_path, Path):
                raise ToolError("invalid patched target")
            if patched_file["existed_before"]:
                existing_files.append(target_path)
            else:
                new_files.append(self._format_repo_path(target_path))
        response["backup_dir"] = self._format_repo_path(backup_dir)
        response["new_files"] = new_files
        response["modified_files"] = touched_paths

        self._append_run_event(patch_id, "apply_start", "started", {"paths": touched_paths})
        written_new_files: list[Path] = []
        written_existing_files: list[Path] = []
        try:
            self._backup_existing_files(patch_id, existing_files)
            for patched_file in patched_files:
                target_path = patched_file["target_path"]
                if not isinstance(target_path, Path):
                    raise ToolError("invalid patched target")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if patched_file["existed_before"]:
                    written_existing_files.append(target_path)
                else:
                    written_new_files.append(target_path)
                target_path.write_text(str(patched_file["content"]), encoding="utf-8")
        except (OSError, ToolError) as error:
            rollback_result = self._rollback_patch_apply(patch_id, written_existing_files, written_new_files)
            response["status"] = "failed"
            response["rollback"] = rollback_result
            response["errors"] = [str(error)]
            self._append_run_event(patch_id, "apply_failure", "failed", {"error": str(error), "rollback": rollback_result})
            return response

        applied_at = datetime.now().astimezone().isoformat()
        metadata["status"] = "applied"
        metadata["applied_at"] = applied_at
        metadata["updated_at"] = applied_at
        metadata["modified_files"] = touched_paths
        metadata["backup_dir"] = self._format_repo_path(backup_dir)
        metadata["new_files"] = new_files
        self._write_patch_metadata(metadata)
        self._append_run_event(
            patch_id,
            "apply_success",
            "applied",
            {"paths": touched_paths, "backup_dir": self._format_repo_path(backup_dir), "new_files": new_files},
        )
        response["ok"] = True
        response["status"] = "applied"
        response["rollback"] = {"attempted": False, "ok": None, "errors": []}
        return response

    def _validate_apply_metadata(self, patch_id: str, metadata: dict[str, object]) -> None:
        if metadata.get("patch_id") != patch_id:
            raise ToolError("metadata.patch_id does not match patch_id")
        status = metadata.get("status")
        if status == "applied":
            raise ToolError("patch is already applied")
        if status != "proposed":
            raise ToolError("only status=proposed patches can be applied")
        paths = metadata.get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ToolError("metadata.paths must be a list of strings")

    def _build_patched_files(self, patch_text: str) -> list[dict[str, object]]:
        diff_lines = patch_text.splitlines()
        line_index = 0
        patched_files: list[dict[str, object]] = []
        while line_index < len(diff_lines):
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
                        self._assert_hunk_source_line(original_lines, source_cursor, body_text, new_file_path)
                        patched_lines.append(body_text)
                        source_cursor += 1
                        removed_count += 1
                        added_count += 1
                    elif marker == "-":
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
        backup_dir = self.repopilot_backups_dir / patch_id
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=False)
        for source_path in existing_files:
            backup_path = backup_dir / source_path.relative_to(self.repo_root)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_path)
        return backup_dir

    def _rollback_patch_apply(self, patch_id: str, existing_files: list[Path], new_files: list[Path]) -> dict[str, object]:
        rollback_errors: list[str] = []
        backup_dir = self.repopilot_backups_dir / patch_id
        for target_path in existing_files:
            backup_path = backup_dir / target_path.relative_to(self.repo_root)
            try:
                if not backup_path.is_file():
                    raise ToolError(f"backup file is missing: {self._format_repo_path(target_path)}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, target_path)
            except (OSError, ToolError) as error:
                rollback_errors.append(str(error))
        for target_path in reversed(new_files):
            try:
                if target_path.exists():
                    target_path.unlink()
                self._remove_empty_parent_dirs(target_path.parent)
            except OSError as error:
                rollback_errors.append(str(error))
        rollback_ok = not rollback_errors
        self._append_run_event(
            patch_id,
            "rollback_success" if rollback_ok else "rollback_failure",
            "ok" if rollback_ok else "failed",
            {"errors": rollback_errors},
        )
        return {"attempted": True, "ok": rollback_ok, "errors": rollback_errors}

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
        """执行白名单内的测试命令。"""

        normalized_command_name = self._validate_run_command_name(command_name)
        command_map = {
            "unit": [sys.executable, "-m", "unittest", "discover"],
            "compile": [sys.executable, "-m", "compileall", "."],
        }

        command = command_map[normalized_command_name]
        run_id = self._generate_patch_id(f"run_tests:{normalized_command_name}")

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
        warnings: list[str] | None = None,
        status: str = "proposed",
        created_at: datetime | None = None,
    ) -> dict[str, object]:
        self._validate_patch_id(patch_id)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolError("instruction 必须是非空字符串")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ToolError("paths 必须是字符串列表")
        if warnings is not None and not all(isinstance(warning, str) for warning in warnings):
            raise ToolError("warnings 必须是字符串列表")
        if not isinstance(status, str) or not status.strip():
            raise ToolError("status 必须是非空字符串")

        timestamp = created_at or datetime.now().astimezone()
        return {
            "patch_id": patch_id,
            "created_at": timestamp.isoformat(),
            "instruction": instruction,
            "status": status,
            "paths": paths,
            "warnings": warnings or [],
        }

    def _write_patch_file(self, patch_id: str, patch_text: str) -> Path:
        self._validate_patch_id(patch_id)
        if not isinstance(patch_text, str):
            raise ToolError("patch_text 必须是字符串")

        patch_path = self.repopilot_patches_dir / f"{patch_id}.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        return patch_path

    def _read_patch_file(self, patch_id: str) -> str:
        patch_path = self._get_patch_file_path(patch_id)
        if not patch_path.is_file():
            raise ToolError(f"patch 文件不存在: {patch_id}")

        return patch_path.read_text(encoding="utf-8")

    def _write_patch_metadata(self, metadata: dict[str, object]) -> Path:
        patch_id = metadata.get("patch_id")
        if not isinstance(patch_id, str):
            raise ToolError("metadata.patch_id 必须是字符串")
        self._validate_patch_id(patch_id)

        metadata_path = self.repopilot_patches_dir / f"{patch_id}.json"
        serialized_metadata = json.dumps(metadata, ensure_ascii=False, indent=2)
        metadata_path.write_text(serialized_metadata, encoding="utf-8")
        return metadata_path

    def _read_patch_metadata(self, patch_id: str) -> dict[str, object]:
        metadata_path = self._get_patch_metadata_path(patch_id)
        if not metadata_path.is_file():
            raise ToolError(f"metadata 文件不存在: {patch_id}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ToolError(f"metadata 不是有效 JSON: {patch_id}") from error
        if not isinstance(metadata, dict):
            raise ToolError(f"metadata 必须是 JSON 对象: {patch_id}")

        return metadata

    def _update_patch_metadata_status(self, patch_id: str, status: str) -> dict[str, object]:
        if not isinstance(status, str) or not status.strip():
            raise ToolError("status 必须是非空字符串")

        metadata = self._read_patch_metadata(patch_id)
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

    def _get_patch_file_path(self, patch_id: str) -> Path:
        self._validate_patch_id(patch_id)
        return self.repopilot_patches_dir / f"{patch_id}.patch"

    def _get_patch_metadata_path(self, patch_id: str) -> Path:
        self._validate_patch_id(patch_id)
        return self.repopilot_patches_dir / f"{patch_id}.json"

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
