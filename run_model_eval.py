"""多模型 profile edit eval 汇总 runner。

运行思路：
- 在给定 profile 列表和 trials 下构建 `profile × case × trial` 矩阵；
- 每个 trial 独立执行一条 edit case；
- 按超时控制单 trial；
- 汇总 `results.json`、`summary.csv`、`summary.md`。

该 runner 不变更 `run_edit_case` 的评测规则，只负责矩阵调度与汇总边界。
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_runner import (
    EditEvalConfigError,
    classify_failure_type,
    load_edit_cases,
    run_edit_case,
)
from llm_client import LlmClient


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval_cases" / "edit_cases.json"
DEFAULT_PROFILES = ("default",)
DEFAULT_TRIAL_TIMEOUT_SECONDS = 2.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析多模型 eval 命令行参数。"""

    parser = argparse.ArgumentParser(description="运行多 model profile edit eval 并汇总结果")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="edit eval 用例 JSON 文件路径，默认 eval_cases/edit_cases.json",
    )
    parser.add_argument(
        "--repo-root",
        default=str(PROJECT_ROOT),
        help="项目根目录，默认 run_model_eval.py 所在目录",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        help="要评测的 model profile 名称列表，例如 default fast strong",
    )
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=1,
        help="每个 profile/case 组合重复执行次数，必须为正整数",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
        help="单个 profile/case/trial 的最大执行秒数，默认 2 秒",
    )
    return parser.parse_args(argv)


def run_model_eval(
    *,
    cases_path: Path,
    repo_root: Path,
    profiles: Sequence[str],
    trials: int,
    trial_timeout_seconds: float = DEFAULT_TRIAL_TIMEOUT_SECONDS,
    timestamp: str | None = None,
) -> Path:
    """执行 profile/case/trial 矩阵并写入结果文件。

`trial` 用于重复采样同一 profile 的稳定性：
同一 profile 下同一 case 会重复运行 trials 次，结果全部落到单一 `profile` 分组。
每条 trial 都会保留 `profile` 和 `trial_index`，方便回溯哪一次超时/失败。
    """

    resolved_cases_path = Path(cases_path).expanduser().resolve()
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    normalized_profiles = tuple(_normalize_profiles(profiles))
    if trials <= 0:
        raise ValueError("trials 必须为正整数")
    if trial_timeout_seconds <= 0:
        raise ValueError("trial_timeout_seconds 必须为正数")

    started_at = datetime.now().astimezone()
    run_timestamp = timestamp or started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = resolved_repo_root / ".repopilot" / "model_evals" / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # profile×case×trial 的完整矩阵在此层面展开，单条 case 失败不阻断同 profile 其它 trial。
    cases = load_edit_cases(resolved_cases_path)
    trial_results: list[dict[str, object]] = []
    for profile in normalized_profiles:
        profile_error = _preflight_profile(profile)
        if profile_error is not None:
            # profile 初始化失败属于配置级问题，同一 profile 的所有 case/trial 都会失败；
            # 这里直接展开失败结果，避免反复构造客户端并淹没真正的 profile 错误。
            for trial_index in range(1, trials + 1):
                for eval_case in cases:
                    trial_results.append(
                        _build_profile_preflight_failure(
                            eval_case=eval_case,
                            profile=profile,
                            trial_index=trial_index,
                            error=profile_error,
                        )
                    )
            continue

        for trial_index in range(1, trials + 1):
            for eval_case in cases:
                trial_results.append(
                    _run_single_trial(
                        eval_case=eval_case,
                        repo_root=resolved_repo_root,
                        profile=profile,
                        trial_index=trial_index,
                        timeout_seconds=trial_timeout_seconds,
                    )
                )

    finished_at = datetime.now().astimezone()
    summary_rows = _build_summary_rows(trial_results, normalized_profiles)
    _write_results_json(
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
        cases_path=resolved_cases_path,
        repo_root=resolved_repo_root,
        profiles=normalized_profiles,
        trials=trials,
        trial_results=trial_results,
    )
    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_summary_md(output_dir / "summary.md", summary_rows)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """运行多模型 eval CLI。

CLI 负责：解析参数、执行 `run_model_eval`、在标准输出打印 report 目录。
退出码 0 代表主流程无配置级异常，1 代表参数/配置错误导致未完成。
"""

    args = parse_args(argv)
    try:
        output_dir = run_model_eval(
            cases_path=Path(args.cases),
            repo_root=Path(args.repo_root),
            profiles=tuple(args.profiles),
            trials=args.trials,
            trial_timeout_seconds=args.trial_timeout_seconds,
        )
    except EditEvalConfigError as error:
        print(f"model eval 配置错误: {error}", file=sys.stderr)
        return 1

    print(f"model eval report: {output_dir}")
    return 0


def _preflight_profile(profile: str) -> str | None:
    """对单个 profile 做一次轻量化预检。

预检目的：在进入 trial 循环前就验证 profile 配置是否可用，避免在大量 case/trial 中重复报同一初始化错误。
如果失败，返回错误文本由上层统一注入失败结果；否则返回 None。
    """

    try:
        LlmClient(model_profile=profile)
    except Exception as error:  # noqa: BLE001 - profile 构造失败应结构化记录并继续其他 profile
        return str(error)
    return None


def _build_profile_preflight_failure(
    *,
    eval_case: Any,
    profile: str,
    trial_index: int,
    error: str,
) -> dict[str, object]:
    case_id = str(getattr(eval_case, "id", "<unknown>"))
    reasons = (f"profile 配置预检失败: {error}",)
    return {
        "case_id": case_id,
        "passed": False,
        "reasons": list(reasons),
        "changed_files": [],
        "steps": 0,
        "final_answer": None,
        "error": error,
        "test_results": None,
        "context_stats": None,
        "model_profile": profile,
        "provider": None,
        "model": None,
        "llm_call_count": 0,
        "total_latency_ms": 0.0,
        "total_tokens": 0,
        "estimated_tokens": 0,
        "estimated_cost": None,
        "failure_type": classify_failure_type(
            passed=False,
            reasons=reasons,
            error=error,
            test_results=None,
            payload=None,
        ),
        "profile": profile,
        "trial_index": trial_index,
    }


def _run_single_trial(
    *,
    eval_case: Any,
    repo_root: Path,
    profile: str,
    trial_index: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """执行单个 (profile, case, trial) 试次。

    在当前实现里使用 `threading + queue` 做超时控制：
    若超过 `timeout_seconds` 仍未返回 worker 结果，则判定 trial timed out。
    """

    result_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)

    def run_trial_worker() -> None:
        # 子线程里只做一次无超时执行；超时判定统一在主线程 get(timeout=...) 完成，
        # 避免为每个用例引入复杂的 signal 或进程级中断逻辑。
        trial_payload = _run_single_trial_without_timeout(
            eval_case=eval_case,
            repo_root=repo_root,
            profile=profile,
            trial_index=trial_index,
        )
        try:
            result_queue.put_nowait(trial_payload)
        except queue.Full:
            # 主线程已经按 timeout 返回失败时，迟到的 worker 结果不再进入汇总，
            # 否则同一个 trial 会出现“超时失败”和“实际结果”两条记录。
            pass

    worker_thread = threading.Thread(
        target=run_trial_worker,
        name=f"model-eval-{profile}-{getattr(eval_case, 'id', 'unknown')}-{trial_index}",
        daemon=True,
    )
    worker_thread.start()
    try:
        return result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return _build_trial_timeout_failure(
            eval_case=eval_case,
            profile=profile,
            trial_index=trial_index,
            timeout_seconds=timeout_seconds,
        )


def _run_single_trial_without_timeout(
    *,
    eval_case: Any,
    repo_root: Path,
    profile: str,
    trial_index: int,
) -> dict[str, object]:
    """无超时包装的 trial 执行；异常会被转换为统一失败 payload。

    结果会补齐 `profile` 与 `trial_index` 两个维度元数据，统一由汇总函数处理。
    """

    case_id = str(getattr(eval_case, "id", "<unknown>"))
    try:
        eval_result = run_edit_case(
            case=eval_case,
            project_root=repo_root,
            llm_client_factory=lambda _case: LlmClient(model_profile=profile),
        )
        trial_payload = asdict(eval_result)
    except Exception as error:  # noqa: BLE001 - 单 trial 异常必须转换为失败结果并继续
        error_text = str(error)
        reasons = (f"case 执行异常: {error_text}",)
        trial_payload = {
            "case_id": case_id,
            "passed": False,
            "reasons": list(reasons),
            "changed_files": [],
            "steps": 0,
            "final_answer": None,
            "error": error_text,
            "test_results": None,
            "context_stats": None,
            "model_profile": profile,
            "provider": None,
            "model": None,
            "llm_call_count": 0,
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "estimated_cost": None,
            "failure_type": classify_failure_type(
                passed=False,
                reasons=reasons,
                error=error_text,
                test_results=None,
                payload=None,
            ),
        }

    trial_payload["profile"] = profile
    trial_payload["trial_index"] = trial_index
    if not isinstance(trial_payload.get("model_profile"), str):
        # 如果失败发生在 logger 写入模型字段之前，仍保留矩阵维度中的 profile，
        # 便于 summary 按 profile 归组而不丢失来源。
        trial_payload["model_profile"] = profile
    return trial_payload


def _build_trial_timeout_failure(
    *,
    eval_case: Any,
    profile: str,
    trial_index: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """构建超时失败的 trial 结果。

    将 `total_latency_ms` 统一按 timeout_ms 记录，便于与已执行成功 trial 的
    平均耗时进行对齐。
    """
    case_id = str(getattr(eval_case, "id", "<unknown>"))
    error_text = f"trial timed out after {timeout_seconds:.3f} seconds"
    reasons = (f"case 执行超时: {error_text}",)
    return {
        "case_id": case_id,
        "passed": False,
        "reasons": list(reasons),
        "changed_files": [],
        "steps": 0,
        "final_answer": None,
        "error": error_text,
        "test_results": None,
        "context_stats": None,
        "model_profile": profile,
        "provider": None,
        "model": None,
        "llm_call_count": 0,
        "total_latency_ms": timeout_seconds * 1000.0,
        "total_tokens": 0,
        "estimated_tokens": 0,
        "estimated_cost": None,
        "failure_type": classify_failure_type(
            passed=False,
            reasons=reasons,
            error=error_text,
            test_results=None,
            payload=None,
        ),
        "profile": profile,
        "trial_index": trial_index,
    }


def _build_summary_rows(
    trial_results: Sequence[dict[str, object]],
    profiles: Sequence[str],
) -> list[dict[str, str]]:
    """按 profile 聚合 trial 结果，生成汇总表行。

    `failure_type_counts` 仅统计未通过试次，`avg_*` 由数值字段直接计算。
    """

    rows: list[dict[str, str]] = []
    for profile in profiles:
        profile_results = [
            trial_payload
            for trial_payload in trial_results
            if trial_payload.get("profile") == profile
        ]
        total_count = len(profile_results)
        passed_count = sum(1 for trial_payload in profile_results if trial_payload.get("passed") is True)
        pass_rate = passed_count / total_count if total_count else 0.0
        failure_type_counts = Counter(
            str(trial_payload.get("failure_type") or "unknown")
            for trial_payload in profile_results
            if trial_payload.get("passed") is not True
        )
        # summary 只做 profile 维度聚合；case/trial 细节保留在 results.json，避免表格过宽。
        rows.append(
            {
                "profile": profile,
                "total": str(total_count),
                "passed": str(passed_count),
                "pass_rate": _format_float(pass_rate),
                "avg_steps": _format_float(_average_number(profile_results, "steps")),
                "avg_latency_ms": _format_float(
                    _average_number(profile_results, "total_latency_ms")
                ),
                "avg_tokens": _format_float(_average_number(profile_results, "total_tokens")),
                "avg_estimated_cost": _format_optional_float(
                    _average_optional_number(profile_results, "estimated_cost")
                ),
                "failure_type_counts": json.dumps(
                    dict(sorted(failure_type_counts.items())),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return rows


def _write_results_json(
    *,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
    cases_path: Path,
    repo_root: Path,
    profiles: Sequence[str],
    trials: int,
    trial_results: Sequence[dict[str, object]],
) -> None:
    """将批次完整结构持久化到 `results.json`。

输出字段与 `run_edit_eval` 保持一致（started/finished、profiles、summary 与 results），
便于后处理脚本统一消费。该文件是回溯和复现的主数据源。
    """

    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "cases_path": str(cases_path),
        "repo_root": str(repo_root),
        "profiles": list(profiles),
        "trials": trials,
        "total": len(trial_results),
        "passed": sum(1 for trial_payload in trial_results if trial_payload.get("passed") is True),
        "results": list(trial_results),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(summary_path: Path, summary_rows: Sequence[dict[str, str]]) -> None:
    """写入 `summary.csv`，字段顺序固定用于命令行快速 diff。

使用 `csv.DictWriter` 而非手工拼接，保证空值、逗号等特殊字符一致转义。
    """

    fieldnames = [
        "profile",
        "total",
        "passed",
        "pass_rate",
        "avg_steps",
        "avg_latency_ms",
        "avg_tokens",
        "avg_estimated_cost",
        "failure_type_counts",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_summary_md(summary_path: Path, summary_rows: Sequence[dict[str, str]]) -> None:
    """写入 Markdown 人类可读摘要。

该视图主要用于 CI 或本地快速人工审阅，不作为机器判定的唯一来源。
"""

    lines = [
        "# Model Eval Summary",
        "",
        "| profile | total | passed | pass_rate | avg_steps | avg_latency_ms | avg_tokens | avg_estimated_cost | failure_type_counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary_row in summary_rows:
        lines.append(
            "| {profile} | {total} | {passed} | {pass_rate} | {avg_steps} | "
            "{avg_latency_ms} | {avg_tokens} | {avg_estimated_cost} | {failure_type_counts} |".format(
                **summary_row
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    """标准化 profile 输入列表。

过滤空白条目，空列表视作配置错误：至少需要一个 profile。
"""

    normalized_profiles = tuple(profile.strip() for profile in profiles if profile.strip())
    if not normalized_profiles:
        raise ValueError("profiles 至少需要一个非空名称")
    return normalized_profiles


def _average_number(rows: Sequence[dict[str, object]], key: str) -> float:
    """计算某 key 的平均值；输入缺失则按 0 处理。"""

    values = _numeric_values(rows, key)
    return sum(values) / len(values) if values else 0.0


def _average_optional_number(rows: Sequence[dict[str, object]], key: str) -> float | None:
    """计算可缺失字段的平均值，全部缺失返回 None。"""

    values = _numeric_values(rows, key)
    if not values:
        return None
    return sum(values) / len(values)


def _numeric_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    """提取可参与数值计算的字段值，过滤掉 bool 与非数值。

`bool` 显式排除，避免被当作 0/1 进入平均值统计。
"""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _format_float(value: float) -> str:
    """按统一精度格式化浮点统计，便于 diff 与报告对齐。"""

    return f"{value:.6f}"


def _format_optional_float(value: float | None) -> str:
    """将可缺失数值转为表格字符串：缺失时返回空串。"""

    if value is None:
        return ""
    return _format_float(value)


def _positive_int(raw_value: str) -> int:
    """CLI 正整数参数解析器，失败时抛 argparse 错误。"""

    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed_value


def _positive_float(raw_value: str) -> float:
    """CLI 正浮点参数解析器，失败时抛 argparse 错误。"""

    try:
        parsed_value = float(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正数") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return parsed_value


if __name__ == "__main__":
    raise SystemExit(main())
