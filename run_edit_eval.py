"""Edit eval CLI runner。

职责：解析运行参数、触发批量 edit eval、输出摘要与写入 JSON report。
与 `eval_runner.run_edit_eval` 配合完成完整链路：
- 评测语义与校验在 `eval_runner` 内；
- 本文件聚焦 CLI 参数、报告路径、输出格式和运行入口。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from eval_runner import EditEvalConfigError, run_edit_eval


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval_cases" / "edit_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".repopilot" / "evals"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 edit eval 并持久化结果")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="edit eval 用例 JSON 文件路径，默认 eval_cases/edit_cases.json",
    )
    parser.add_argument(
        "--repo-root",
        default=str(PROJECT_ROOT),
        help="项目根目录，默认 run_edit_eval.py 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="结果输出目录，默认 .repopilot/evals",
    )
    return parser.parse_args()


def write_eval_report(
    eval_payload: dict[str, object],
    cases_path: Path,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """写入一次 edit eval 的 JSON 报告。

`result_data` 是主运行结果；report 文件只持久化汇总字段，
供 CI 或历史归档读取。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "cases_path": str(cases_path),
        "total": eval_payload.get("total", 0),
        "passed": eval_payload.get("passed", 0),
        "pass_rate": eval_payload.get("pass_rate", 0.0),
        "results": eval_payload.get("results", []),
    }
    report_path = output_dir / f"{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def print_summary(eval_payload: dict[str, object]) -> None:
    """输出每条用例结果和总通过率。

该输出为人类可读视图：
- `[PASS] case_id` 或 `[FAIL] case_id: reason`；
- 末尾输出 `pass rate: passed/total`。
failure_type 在 fail 时会作为额外标签输出，便于快速定位。
    """

    raw_results = eval_payload.get("results", [])
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            case_id = raw_result.get("case_id", "<unknown>")
            if raw_result.get("passed") is True:
                print(f"[PASS] {case_id}")
            else:
                reasons = raw_result.get("reasons", [])
                reason_text = _format_reasons(reasons)
                failure_type = raw_result.get("failure_type")
                if isinstance(failure_type, str) and failure_type.strip():
                    print(f"[FAIL] {case_id}: {reason_text} (failure_type={failure_type})")
                else:
                    print(f"[FAIL] {case_id}: {reason_text}")

    passed = eval_payload.get("passed", 0)
    total = eval_payload.get("total", 0)
    print(f"pass rate: {passed}/{total}")


def main() -> int:
    """运行 edit eval CLI。

    工作流程：
    1. 解析 `--cases` / `--repo-root` / `--output-dir`；
    2. 执行 `run_edit_eval`；
    3. 打印摘要并写入带时间戳的 report；
    4. 返回整体 pass/fail 状态码。
    """

    args = parse_args()
    cases_path = Path(args.cases).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    started_at = datetime.now().astimezone()

    try:
        eval_payload = run_edit_eval(cases_path=cases_path, project_root=repo_root)
    except EditEvalConfigError as error:
        print(f"edit eval 配置错误: {error}", file=sys.stderr)
        return 1

    finished_at = datetime.now().astimezone()
    print_summary(eval_payload)
    report_path = write_eval_report(
        eval_payload=eval_payload,
        cases_path=cases_path,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )
    print(f"report: {report_path}")
    return 0 if eval_payload.get("passed") == eval_payload.get("total") else 1


def _format_reasons(raw_reasons: object) -> str:
    if not isinstance(raw_reasons, list | tuple):
        return "未知失败原因"
    reasons = [str(reason) for reason in raw_reasons if str(reason)]
    if not reasons:
        return "未知失败原因"
    return "; ".join(reasons)


if __name__ == "__main__":
    raise SystemExit(main())
