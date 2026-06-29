"""Edit 评测用例加载器。"""

from __future__ import annotations

import json
import hashlib
import inspect
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from agent import CodeAnalysisAgent
from config import MAX_STEPS, load_llm_config_from_env
from eval_safety import EVAL_TEMP_MARKER, normalize_relative_path, write_eval_temp_marker
from llm_client import LlmClient
from logger import RunLogger
from main import CliApplyPatchApproval
from tools import RepositoryTools


class EditEvalConfigError(ValueError):
    """Edit 评测配置文件格式错误。"""


@dataclass(frozen=True)
class MustContainRule:
    """单条 must_contain 规则。"""

    path: str
    strings: tuple[str, ...]


@dataclass(frozen=True)
class EditEvalCase:
    """单条编辑评测配置。"""

    id: str
    fixture: str
    prompt: str
    max_steps: int
    allowed_changed_files: tuple[str, ...]
    must_contain: tuple[MustContainRule, ...]
    test_command: str | None = None
    expect_no_business_changes: bool = False
    must_not_read_full_files: tuple[str, ...] = ()
    max_total_tool_output_chars: int | None = None
    raw_case: dict[str, object] | None = None


@dataclass(frozen=True)
class EditEvalResult:
    """单条 edit eval 用例执行结果。"""

    case_id: str
    passed: bool
    reasons: tuple[str, ...]
    changed_files: tuple[str, ...]
    steps: int
    final_answer: str | None
    error: str | None
    test_results: dict[str, object] | None
    context_stats: dict[str, object] | None


@dataclass(frozen=True)
class RepoFileSnapshot:
    """repo 内单个文件的快照。"""

    path: str
    sha256: str


@dataclass(frozen=True)
class RepoSnapshotDiff:
    """两个文件快照之间的差异。"""

    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    changed_files: tuple[str, ...]


def copy_fixture_to_temp(fixture_path: Path, temp_root: Path, run_id: str, case_id: str) -> Path:
    """复制 fixture repo 到 eval 临时目录并写入安全 marker。"""

    if not fixture_path.is_dir():
        raise EditEvalConfigError(f"fixture 路径不是目录: {fixture_path}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise EditEvalConfigError("run_id 必须是非空字符串")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EditEvalConfigError("case_id 必须是非空字符串")

    temp_root.mkdir(parents=True, exist_ok=True)
    copied_repo_path = temp_root / _safe_temp_dir_name(run_id, case_id)
    shutil.copytree(fixture_path, copied_repo_path, symlinks=True)
    write_eval_temp_marker(copied_repo_path, run_id, case_id, temp_root)
    return copied_repo_path


def snapshot_repo_files(repo_path: Path) -> dict[str, RepoFileSnapshot]:
    """生成 repo 内普通文件快照，跳过 eval 内部文件和缓存目录。"""

    if not repo_path.is_dir():
        raise EditEvalConfigError(f"repo 路径不是目录: {repo_path}")

    resolved_repo_path = repo_path.resolve()
    snapshots: dict[str, RepoFileSnapshot] = {}
    for current_dir, dir_names, file_names in os.walk(resolved_repo_path, followlinks=False):
        dir_names[:] = [
            dir_name
            for dir_name in dir_names
            if dir_name not in {".repopilot", "__pycache__", ".pytest_cache"}
        ]

        current_dir_path = Path(current_dir)
        for file_name in file_names:
            if file_name == EVAL_TEMP_MARKER:
                continue
            file_path = current_dir_path / file_name
            if not file_path.is_file() or file_path.is_symlink():
                continue
            relative_path = file_path.relative_to(resolved_repo_path).as_posix()
            snapshots[relative_path] = RepoFileSnapshot(
                path=relative_path,
                sha256=_sha256_file(file_path),
            )

    return dict(sorted(snapshots.items(), key=lambda item: item[0].lower()))


def compare_snapshots(
    before: Mapping[str, RepoFileSnapshot],
    after: Mapping[str, RepoFileSnapshot],
) -> RepoSnapshotDiff:
    """比较两次 repo 文件快照。"""

    before_paths = set(before)
    after_paths = set(after)
    added = tuple(sorted(after_paths - before_paths))
    deleted = tuple(sorted(before_paths - after_paths))
    modified = tuple(
        sorted(
            path
            for path in before_paths & after_paths
            if before[path].sha256 != after[path].sha256
        )
    )
    changed_files = tuple(sorted((*added, *modified, *deleted)))
    return RepoSnapshotDiff(
        added=added,
        modified=modified,
        deleted=deleted,
        changed_files=changed_files,
    )


def check_allowed_changed_files(
    changed_files: Iterable[str],
    allowed_changed_files: Iterable[str],
) -> list[str]:
    """校验实际变更文件是否都在允许列表中。"""

    errors: list[str] = []
    allowed_paths: set[str] = set()
    for allowed_file in allowed_changed_files:
        try:
            allowed_paths.add(normalize_relative_path(allowed_file))
        except ValueError as error:
            errors.append(f"allowed_changed_files 包含非法路径 {allowed_file!r}: {error}")

    for changed_file in changed_files:
        try:
            normalized_changed_file = normalize_relative_path(changed_file)
        except ValueError as error:
            errors.append(f"changed_files 包含非法路径 {changed_file!r}: {error}")
            continue
        if normalized_changed_file not in allowed_paths:
            errors.append(f"未授权的变更文件: {normalized_changed_file}")

    return errors


def check_must_contain(repo_path: Path, rules: Iterable[MustContainRule | Mapping[str, object]]) -> list[str]:
    """校验指定文件必须包含的文本。"""

    errors: list[str] = []
    resolved_repo_path = repo_path.resolve()
    for rule in rules:
        raw_path, required_strings = _extract_must_contain_rule(rule)
        try:
            normalized_path = normalize_relative_path(raw_path)
        except ValueError as error:
            errors.append(f"must_contain 路径非法 {raw_path!r}: {error}")
            continue

        target_file = (resolved_repo_path / normalized_path).resolve()
        try:
            target_file.relative_to(resolved_repo_path)
        except ValueError:
            errors.append(f"must_contain 文件不在 repo 内: {normalized_path}")
            continue
        if not target_file.is_file():
            errors.append(f"must_contain 文件不存在: {normalized_path}")
            continue

        try:
            file_text = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"must_contain 文件不是有效 UTF-8: {normalized_path}")
            continue
        except OSError as error:
            errors.append(f"must_contain 文件读取失败 {normalized_path}: {error}")
            continue

        for required_string in required_strings:
            if required_string not in file_text:
                errors.append(f"must_contain 缺少文本: {normalized_path} -> {required_string}")

    return errors


def load_edit_cases(cases_path: Path) -> list[EditEvalCase]:
    """从 JSON 文件加载编辑评测用例。"""

    try:
        raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EditEvalConfigError(f"无法读取 eval 文件: {cases_path}") from error
    except json.JSONDecodeError as error:
        raise EditEvalConfigError(f"eval 文件不是有效 JSON: {cases_path}") from error

    if not isinstance(raw_cases, list):
        raise EditEvalConfigError("eval 文件根节点必须是非空数组")
    if not raw_cases:
        raise EditEvalConfigError("eval 文件至少需要一条用例")

    seen_ids: set[str] = set()
    parsed_cases: list[EditEvalCase] = []
    for case_index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise EditEvalConfigError(f"第 {case_index} 条用例必须是对象")

        case_id = _require_non_empty_string(raw_case, "id", case_index)
        if case_id in seen_ids:
            raise EditEvalConfigError(f"id 重复: {case_id}")
        seen_ids.add(case_id)

        fixture = _require_non_empty_string(raw_case, "fixture", case_index)
        prompt = _require_non_empty_string(raw_case, "prompt", case_index)
        max_steps = _normalize_max_steps(raw_case.get("max_steps"), case_index)
        allowed_changed_files = _normalize_allowed_changed_files(
            raw_case.get("allowed_changed_files"),
            case_index,
        )
        must_contain = _normalize_must_contain(raw_case.get("must_contain"), case_index)
        must_not_read_full_files = _normalize_optional_relative_paths(
            raw_case.get("must_not_read_full_files"),
            case_index,
            "must_not_read_full_files",
        )
        max_total_tool_output_chars = _normalize_optional_positive_int(
            raw_case.get("max_total_tool_output_chars"),
            case_index,
            "max_total_tool_output_chars",
        )
        test_command = raw_case.get("test_command")

        if test_command is not None:
            if not isinstance(test_command, str):
                raise EditEvalConfigError(
                    f"第 {case_index} 条用例 test_command 必须是 unit 或 compile"
                )
            if test_command not in {"unit", "compile"}:
                raise EditEvalConfigError(
                    f"第 {case_index} 条用例 test_command 必须是 unit 或 compile"
                )

        expect_no_business_changes = raw_case.get("expect_no_business_changes", False)
        if expect_no_business_changes is None:
            expect_no_business_changes = False
        if not isinstance(expect_no_business_changes, bool):
            raise EditEvalConfigError(
                f"第 {case_index} 条用例 expect_no_business_changes 必须是 bool"
            )

        parsed_cases.append(
            EditEvalCase(
                id=case_id,
                fixture=fixture,
                prompt=prompt,
                max_steps=max_steps,
                allowed_changed_files=allowed_changed_files,
                must_contain=must_contain,
                test_command=test_command,
                expect_no_business_changes=expect_no_business_changes,
                must_not_read_full_files=must_not_read_full_files,
                max_total_tool_output_chars=max_total_tool_output_chars,
                raw_case=dict(raw_case),
            )
        )

    return parsed_cases


def run_edit_case(
    case: EditEvalCase,
    project_root: Path,
    llm_client_factory: Callable[..., Any] | None = None,
) -> EditEvalResult:
    """在隔离临时目录中执行单条 edit eval 用例。"""

    run_id = f"edit-eval-{uuid.uuid4().hex}"
    changed_files: tuple[str, ...] = ()
    steps = 0
    final_answer: str | None = None
    error_message: str | None = None
    test_results: dict[str, object] | None = None
    context_stats: dict[str, object] | None = None
    reasons: list[str] = []

    try:
        resolved_project_root = Path(project_root).expanduser().resolve()
        fixture_path = Path(case.fixture).expanduser()
        if not fixture_path.is_absolute():
            fixture_path = resolved_project_root / fixture_path
        fixture_path = fixture_path.resolve()

        with tempfile.TemporaryDirectory(prefix="edit-eval-") as temp_directory:
            temp_root = Path(temp_directory).resolve()
            temp_repo = copy_fixture_to_temp(
                fixture_path=fixture_path,
                temp_root=temp_root,
                run_id=run_id,
                case_id=case.id,
            )
            before_snapshot = snapshot_repo_files(temp_repo)
            repository_tools = RepositoryTools(temp_repo)
            run_logger = RunLogger(
                repo_path=temp_repo,
                user_task=case.prompt,
                log_dir=repository_tools.repopilot_runs_dir,
            )
            llm_client = _build_llm_client(case, llm_client_factory)
            approval_gate = CliApplyPatchApproval(
                repository_tools,
                approval_mode="auto_for_eval",
                eval_run_id=run_id,
            )
            agent = CodeAnalysisAgent(
                llm_client=llm_client,
                repository_tools=repository_tools,
                run_logger=run_logger,
                max_steps=case.max_steps,
                tool_runner=approval_gate.run_tool,
            )

            case_exception: Exception | None = None
            try:
                agent.answer(_build_eval_agent_prompt(case))
            except Exception as error:  # noqa: BLE001 - eval 需保留异常后的文件变更证据
                case_exception = error

            try:
                after_snapshot = snapshot_repo_files(temp_repo)
                snapshot_diff = compare_snapshots(before_snapshot, after_snapshot)
                changed_files = snapshot_diff.changed_files
            except Exception as error:  # noqa: BLE001 - snapshot 失败也要返回清晰失败原因
                case_exception = error

            steps = _count_logger_steps(run_logger)
            final_answer = _optional_string(run_logger.payload.get("final_answer"))
            error_message = _optional_string(run_logger.payload.get("error"))
            context_stats = _optional_context_stats(run_logger.payload.get("context_stats"))
            reasons.extend(_check_context_constraints(case, context_stats))

            if case_exception is not None:
                error_message = str(case_exception)
                reasons.append(f"case 执行异常: {case_exception}")
                return EditEvalResult(
                    case_id=case.id,
                    passed=False,
                    reasons=tuple(reasons),
                    changed_files=changed_files,
                    steps=steps,
                    final_answer=final_answer,
                    error=error_message,
                    test_results=test_results,
                    context_stats=context_stats,
                )

            if not _logger_finished(run_logger):
                reasons.append("agent 未在 max_steps 内成功调用 finish")

            reasons.extend(
                check_allowed_changed_files(
                    changed_files=changed_files,
                    allowed_changed_files=case.allowed_changed_files,
                )
            )

            if case.expect_no_business_changes and changed_files:
                reasons.append(
                    "expect_no_business_changes 要求无业务文件变更，"
                    f"但实际变更: {', '.join(changed_files)}"
                )

            reasons.extend(check_must_contain(temp_repo, case.must_contain))

            if case.test_command is not None:
                test_results = repository_tools.run_tests(case.test_command)
                if test_results.get("timed_out") is True:
                    reasons.append(f"test_command {case.test_command} 执行超时")
                if test_results.get("exit_code") != 0:
                    reasons.append(
                        f"test_command {case.test_command} 退出码非 0: "
                        f"{test_results.get('exit_code')}"
                    )
    except Exception as error:  # noqa: BLE001 - eval runner 必须把单 case 异常转换为失败结果
        error_message = str(error)
        reasons.append(f"case 执行异常: {error}")

    return EditEvalResult(
        case_id=case.id,
        passed=not reasons,
        reasons=tuple(reasons),
        changed_files=changed_files,
        steps=steps,
        final_answer=final_answer,
        error=error_message,
        test_results=test_results,
        context_stats=context_stats,
    )


def run_edit_eval(
    cases_path: Path,
    project_root: Path,
    llm_client_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """加载并连续执行 edit eval 用例，单条失败不影响后续用例。"""

    cases = load_edit_cases(cases_path)
    results = [
        run_edit_case(
            case=case,
            project_root=project_root,
            llm_client_factory=llm_client_factory,
        )
        for case in cases
    ]
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)
    pass_rate = passed_count / total_count if total_count else 0.0

    return {
        "total": total_count,
        "passed": passed_count,
        "pass_rate": pass_rate,
        "results": [asdict(result) for result in results],
    }


def _build_eval_agent_prompt(case: EditEvalCase) -> str:
    """构建 edit eval 专用任务说明。"""

    allowed_files = ", ".join(case.allowed_changed_files) or "无"
    return (
        f"{case.prompt}\n\n"
        "Edit eval 执行约束：\n"
        f"- 只处理这些允许变更文件: {allowed_files}。\n"
        "- 这些目标文件已由评测用例明确给出；除非确实需要定位未知文件，"
        "不要额外调用 inspect_repo。\n"
        "- 读取完允许变更文件并确认修改点后，不要重复读取同一文件；"
        "下一步应提交 propose_patch。\n"
        "- 当前运行在评测临时仓库，apply_patch 已启用 auto_for_eval 自动批准；"
        "propose_patch 成功返回 patch_id 后，应直接调用 apply_patch。\n"
        "- 如果用例需要测试，apply_patch 成功后调用 run_tests；最后必须调用 finish。"
    )


def _build_llm_client(
    case: EditEvalCase,
    llm_client_factory: Callable[..., Any] | None,
) -> Any:
    if llm_client_factory is None:
        return LlmClient(load_llm_config_from_env())

    try:
        signature = inspect.signature(llm_client_factory)
    except (TypeError, ValueError):
        return llm_client_factory(case)

    positional_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        and parameter.default is inspect.Parameter.empty
    ]
    has_required_variadic = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if positional_parameters or has_required_variadic:
        return llm_client_factory(case)
    return llm_client_factory()


def _count_logger_steps(run_logger: RunLogger) -> int:
    steps = run_logger.payload.get("steps")
    if isinstance(steps, list):
        return len(steps)
    return 0


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _optional_context_stats(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


def _check_context_constraints(
    case: EditEvalCase,
    context_stats: dict[str, object] | None,
) -> list[str]:
    if context_stats is None:
        return []

    errors: list[str] = []
    full_file_reads = _normalize_context_path_list(context_stats.get("full_file_reads"))
    for forbidden_path in case.must_not_read_full_files:
        if forbidden_path in full_file_reads:
            errors.append(
                "must_not_read_full_files 违规: "
                f"forbidden_path={forbidden_path}, "
                f"actual_full_file_reads={full_file_reads}"
            )

    total_tool_output_chars = context_stats.get("total_tool_output_chars")
    if (
        case.max_total_tool_output_chars is not None
        and isinstance(total_tool_output_chars, int)
        and not isinstance(total_tool_output_chars, bool)
        and total_tool_output_chars > case.max_total_tool_output_chars
    ):
        errors.append(
            "max_total_tool_output_chars 违规: "
            f"limit={case.max_total_tool_output_chars}, "
            f"actual={total_tool_output_chars}"
        )

    return errors


def _normalize_context_path_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized_paths: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized_paths.append(item.replace("\\", "/"))
    return tuple(normalized_paths)


def _logger_finished(run_logger: RunLogger) -> bool:
    if _optional_string(run_logger.payload.get("error")) is not None:
        return False

    steps = run_logger.payload.get("steps")
    if not isinstance(steps, list):
        return False

    for raw_step in reversed(steps):
        if not isinstance(raw_step, dict):
            continue
        tool_call = raw_step.get("tool_call")
        tool_result = raw_step.get("tool_result")
        if not isinstance(tool_call, dict) or not isinstance(tool_result, dict):
            continue
        if tool_call.get("tool") == "finish" and tool_result.get("ok") is True:
            return _optional_string(run_logger.payload.get("final_answer")) is not None

    return False


def _safe_temp_dir_name(run_id: str, case_id: str) -> str:
    raw_name = f"{run_id.strip()}_{case_id.strip()}"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
    if not safe_name:
        raise EditEvalConfigError("无法生成 eval 临时目录名")
    return safe_name


def _sha256_file(file_path: Path) -> str:
    file_hash = hashlib.sha256()
    with file_path.open("rb") as file_stream:
        for chunk in iter(lambda: file_stream.read(1024 * 1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def _extract_must_contain_rule(
    rule: MustContainRule | Mapping[str, object],
) -> tuple[str, Sequence[str]]:
    if isinstance(rule, MustContainRule):
        return rule.path, rule.strings
    raw_path = rule.get("path")
    raw_strings = rule.get("strings")
    if not isinstance(raw_path, str):
        return "", ()
    if not isinstance(raw_strings, Sequence) or isinstance(raw_strings, (str, bytes)):
        return raw_path, ()
    strings = tuple(string for string in raw_strings if isinstance(string, str))
    return raw_path, strings


def _require_non_empty_string(raw_case: dict[str, object], name: str, case_index: int) -> str:
    """读取并校验必填的非空字符串字段。"""

    value = raw_case.get(name)
    if not isinstance(value, str) or not value.strip():
        raise EditEvalConfigError(f"第 {case_index} 条用例 {name} 缺少非空字符串")
    return value.strip()


def _normalize_max_steps(raw_value: object | None, case_index: int) -> int:
    """规范化 max_steps 字段。"""

    if raw_value is None:
        return MAX_STEPS

    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 max_steps 必须是正整数"
        )
    if raw_value <= 0:
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 max_steps 必须是正整数"
        )
    return raw_value


def _normalize_allowed_changed_files(raw_value: object | None, case_index: int) -> tuple[str, ...]:
    """规范化并校验 allowed_changed_files。"""

    return _normalize_required_relative_paths(raw_value, case_index, "allowed_changed_files")


def _normalize_required_relative_paths(
    raw_value: object | None,
    case_index: int,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(raw_value, list):
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 {field_name} 必须是字符串数组"
        )

    return _normalize_relative_path_list(raw_value, case_index, field_name)


def _normalize_optional_relative_paths(
    raw_value: object | None,
    case_index: int,
    field_name: str,
) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 {field_name} 必须是字符串数组"
        )

    return _normalize_relative_path_list(raw_value, case_index, field_name)


def _normalize_relative_path_list(
    raw_paths: list[object],
    case_index: int,
    field_name: str,
) -> tuple[str, ...]:

    normalized_files: list[str] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise EditEvalConfigError(
                f"第 {case_index} 条用例 {field_name} 包含非字符串"
            )
        try:
            normalized_file = normalize_relative_path(raw_path)
        except ValueError as error:
            raise EditEvalConfigError(
                f"第 {case_index} 条用例 {field_name} 包含非法路径 {raw_path!r}: {error}"
            ) from error
        normalized_files.append(normalized_file)

    return tuple(normalized_files)


def _normalize_optional_positive_int(
    raw_value: object | None,
    case_index: int,
    field_name: str,
) -> int | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 {field_name} 必须是正整数或 null"
        )
    if raw_value <= 0:
        raise EditEvalConfigError(
            f"第 {case_index} 条用例 {field_name} 必须是正整数或 null"
        )
    return raw_value


def _is_windows_absolute_path(value: str) -> bool:
    """检查 Windows 风格绝对路径。"""

    return (
        len(value) >= 3
        and value[1] == ":"
        and value[0].isalpha()
        and value[2] in {"/", "\\"}
    )


def _normalize_must_contain(raw_value: object | None, case_index: int) -> tuple[MustContainRule, ...]:
    """规范化并校验 must_contain。"""

    if not isinstance(raw_value, list):
        raise EditEvalConfigError(f"第 {case_index} 条用例 must_contain 必须是数组")

    normalized_rules: list[MustContainRule] = []
    for item_index, raw_rule in enumerate(raw_value, start=1):
        if not isinstance(raw_rule, dict):
            raise EditEvalConfigError(
                f"第 {case_index} 条用例 must_contain[{item_index}] 必须是对象"
            )

        path = _require_non_empty_string(raw_rule, "path", case_index)
        raw_strings = raw_rule.get("strings")
        if not isinstance(raw_strings, list) or not raw_strings:
            raise EditEvalConfigError(
                f"第 {case_index} 条用例 must_contain[{item_index}] 的 strings 必须是非空字符串数组"
            )

        strings: list[str] = []
        for string in raw_strings:
            if not isinstance(string, str) or not string.strip():
                raise EditEvalConfigError(
                    f"第 {case_index} 条用例 must_contain[{item_index}] 的 strings "
                    "每项必须是非空字符串"
                )
            strings.append(string)

        normalized_rules.append(MustContainRule(path=path.strip(), strings=tuple(strings)))

    return tuple(normalized_rules)
