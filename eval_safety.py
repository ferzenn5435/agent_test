"""Edit eval 临时代码库安全标记与路径校验。"""

from __future__ import annotations

import json
import re
from pathlib import Path


EVAL_TEMP_MARKER = ".repopilot_eval_temp.json"


class EvalSafetyError(ValueError):
    """Edit eval 临时代码库安全校验失败。"""


def normalize_relative_path(path: str) -> str:
    """规范化 repo 内相对路径，拒绝逃逸或绝对路径。"""

    if not isinstance(path, str):
        raise EvalSafetyError("path 必须是字符串")

    normalized_path = path.replace("\\", "/").strip()
    if not normalized_path:
        raise EvalSafetyError("path 不能为空")
    if normalized_path.startswith("/") or Path(normalized_path).is_absolute():
        raise EvalSafetyError(f"path 不允许是绝对路径: {path}")
    if _is_windows_absolute_path(normalized_path):
        raise EvalSafetyError(f"path 不允许是 Windows 绝对路径: {path}")
    if normalized_path == "~" or normalized_path.startswith("~/"):
        raise EvalSafetyError(f"path 不允许使用 ~: {path}")
    if any(segment == ".." for segment in normalized_path.split("/")):
        raise EvalSafetyError(f"path 不允许包含 ..: {path}")

    return normalized_path


def write_eval_temp_marker(repo_path: Path, run_id: str, case_id: str, temp_root: Path) -> Path:
    """在 eval 临时代码库根目录写入安全 marker。"""

    if not isinstance(run_id, str) or not run_id.strip():
        raise EvalSafetyError("run_id 必须是非空字符串")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvalSafetyError("case_id 必须是非空字符串")

    resolved_repo_path = repo_path.resolve()
    resolved_temp_root = temp_root.resolve()
    _require_inside_temp_root(resolved_repo_path, resolved_temp_root)

    marker_path = resolved_repo_path / EVAL_TEMP_MARKER
    marker_payload = {
        "run_id": run_id.strip(),
        "case_id": case_id.strip(),
        "temp_root": str(resolved_temp_root),
        "repo_path": str(resolved_repo_path),
    }
    marker_path.write_text(
        json.dumps(marker_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return marker_path


def validate_eval_temp_repo(repo_path: Path, run_id: str) -> None:
    """验证 repo_path 是当前 run_id 绑定的 eval 临时代码库。"""

    if not isinstance(run_id, str) or not run_id.strip():
        raise EvalSafetyError("run_id 必须是非空字符串")

    resolved_repo_path = repo_path.resolve()
    marker_path = resolved_repo_path / EVAL_TEMP_MARKER
    if not marker_path.is_file():
        raise EvalSafetyError(f"eval temp marker 不存在: {marker_path}")

    try:
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvalSafetyError(f"eval temp marker 不是有效 JSON: {marker_path}") from error
    except OSError as error:
        raise EvalSafetyError(f"无法读取 eval temp marker: {marker_path}") from error

    if not isinstance(marker_payload, dict):
        raise EvalSafetyError("eval temp marker 必须是 JSON 对象")
    marker_run_id = marker_payload.get("run_id")
    if marker_run_id != run_id.strip():
        raise EvalSafetyError("eval temp marker run_id 不匹配")

    marker_temp_root = _require_marker_path(marker_payload, "temp_root").resolve()
    marker_repo_path = _require_marker_path(marker_payload, "repo_path").resolve()
    _require_inside_temp_root(resolved_repo_path, marker_temp_root)

    if marker_repo_path != resolved_repo_path:
        try:
            resolved_repo_path.relative_to(marker_temp_root)
        except ValueError as error:
            raise EvalSafetyError("当前 repo_path 与 marker 记录不匹配") from error


def _is_windows_absolute_path(value: str) -> bool:
    return re.match(r"^[A-Za-z]:/", value) is not None


def _require_marker_path(marker_payload: dict[str, object], field_name: str) -> Path:
    value = marker_payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvalSafetyError(f"eval temp marker 缺少有效 {field_name}")
    return Path(value)


def _require_inside_temp_root(repo_path: Path, temp_root: Path) -> None:
    try:
        repo_path.relative_to(temp_root)
    except ValueError as error:
        raise EvalSafetyError("eval repo_path 必须位于 temp_root 内部") from error
