"""简单 agent 评估脚本。

该脚本是当前轻量评测入口：
1. 加载 `eval_case.json`；
2. 按 case 调用 agent；
3. 校验 final_answer 是否包含 `must_contain`；
4. 输出结构化 `result_data` 与 overall pass/fail。

CLI 负责“逐题输出 + result_data 打点”，不会持久化复杂 report 文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from agent import CodeAnalysisAgent
from config import ConfigError
from eval_runner import classify_failure_type
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from tools import RepositoryTools, ToolError


DEFAULT_EVAL_FILE = Path(__file__).resolve().parent / "eval_case.json"


@dataclass(frozen=True)
class EvalCase:
    """单条评估用例。"""

    question: str
    must_contain: tuple[str, ...]


class EvalConfigError(ValueError):
    """eval 配置文件格式错误。"""


@dataclass(frozen=True)
class AgentEvalRun:
    """单次 agent 运行结果与日志 payload。"""

    final_answer: str
    payload: Mapping[str, object]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行一组简单 agent eval")
    parser.add_argument("--repo", default=".", help="要评估的本地 repo 路径，默认当前目录")
    parser.add_argument(
        "--eval-file",
        default=str(DEFAULT_EVAL_FILE),
        help="eval 用例 JSON 文件路径，默认 eval_case.json",
    )
    parser.add_argument(
        "--model-profile",
        default="default",
        help="LLM model profile 名称，默认 default",
    )
    return parser.parse_args()


def load_eval_cases(eval_file: Path) -> list[EvalCase]:
    """从 JSON 文件读取 eval 用例。"""

    try:
        raw_cases = json.loads(eval_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvalConfigError(f"无法读取 eval 文件: {eval_file}") from error
    except json.JSONDecodeError as error:
        raise EvalConfigError(f"eval 文件不是有效 JSON: {eval_file}") from error

    if not isinstance(raw_cases, list):
        raise EvalConfigError("eval 文件根节点必须是数组")

    eval_cases: list[EvalCase] = []
    for case_index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise EvalConfigError(f"第 {case_index} 条用例必须是对象")

        question = raw_case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise EvalConfigError(f"第 {case_index} 条用例缺少非空 question")

        must_contain = raw_case.get("must_contain")
        if not isinstance(must_contain, list) or not must_contain:
            raise EvalConfigError(f"第 {case_index} 条用例 must_contain 必须是非空数组")
        if not all(isinstance(keyword, str) and keyword for keyword in must_contain):
            raise EvalConfigError(f"第 {case_index} 条用例 must_contain 只能包含非空字符串")

        eval_cases.append(
            EvalCase(
                question=question.strip(),
                must_contain=tuple(must_contain),
            )
        )

    if not eval_cases:
        raise EvalConfigError("eval 文件至少需要一条用例")

    return eval_cases


def run_agent(repo_path: Path, question: str, model_profile: str = "default") -> AgentEvalRun:
    """调用 agent 并返回最终答案。"""

    repository_tools = RepositoryTools(repo_path)
    run_logger = RunLogger(repo_path=repo_path, user_task=question)
    llm_client = LlmClient(model_profile=model_profile)
    agent = CodeAnalysisAgent(
        llm_client=llm_client,
        repository_tools=repository_tools,
        run_logger=run_logger,
    )
    final_answer = agent.answer(question)
    return AgentEvalRun(final_answer=final_answer, payload=run_logger.payload)


def evaluate_answer(final_answer: str, must_contain: tuple[str, ...]) -> tuple[bool, list[str]]:
    """检查最终答案是否包含所有必需关键词。"""

    missing_keywords = [keyword for keyword in must_contain if keyword not in final_answer]
    return not missing_keywords, missing_keywords


def build_result_payload(
    *,
    case_index: int,
    eval_case: EvalCase,
    model_profile: str,
    passed: bool,
    final_answer: str | None,
    missing_keywords: list[str] | None = None,
    error: str | None = None,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构建 JSON-friendly 的 eval 单条结果。

    字段含义对应自动化采集逻辑：
    - passed / missing_keywords / error 用于判定；
- model_profile / usage 字段用于 trace；
- failure_type 统一走 classify_failure_type，便于与 edit eval 结果口径对齐。
    """

    reasons = _build_failure_reasons(missing_keywords, error)
    metrics = _extract_current_metrics(payload, model_profile)
    return {
        "case_index": case_index,
        "question": eval_case.question,
        "must_contain": list(eval_case.must_contain),
        "passed": passed,
        "missing_keywords": list(missing_keywords or []),
        "answer": final_answer,
        "error": error,
        "model_profile": metrics["model_profile"],
        "provider": metrics["provider"],
        "model": metrics["model"],
        "llm_call_count": metrics["llm_call_count"],
        "total_latency_ms": metrics["total_latency_ms"],
        "total_tokens": metrics["total_tokens"],
        "estimated_tokens": metrics["estimated_tokens"],
        "estimated_cost": metrics["estimated_cost"],
        "failure_type": classify_failure_type(
            passed=passed,
            reasons=reasons,
            error=error,
            test_results=None,
            payload=payload,
        ),
    }


def main() -> int:
    """运行 eval 并输出 pass/fail。

CLI 角色：
- 从 `--eval-file` 读取用例并逐条执行；
- 每条 case 打印 `question/must_contain/result/result_data`；
- 末尾打印 overall pass/fail；
- 以退出码表达整体状态（通过=0，失败=1）。
    """

    args = parse_args()
    repo_path = Path(args.repo).expanduser().resolve()
    eval_file = Path(args.eval_file).expanduser().resolve()
    model_profile = args.model_profile

    try:
        eval_cases = load_eval_cases(eval_file)
    except EvalConfigError as error:
        print(f"eval 配置错误: {error}", file=sys.stderr)
        return 1

    all_passed = True
    for case_index, eval_case in enumerate(eval_cases, start=1):
        print(f"[{case_index}] question: {eval_case.question}")
        print(f"must_contain: {', '.join(eval_case.must_contain)}")

        try:
            agent_run = run_agent(repo_path, eval_case.question, model_profile)
        except (ConfigError, LlmClientError, ToolError, OSError) as error:
            print(f"result: fail")
            print(f"error: {error}")
            print(
                "result_data: "
                + json.dumps(
                    build_result_payload(
                        case_index=case_index,
                        eval_case=eval_case,
                        model_profile=model_profile,
                        passed=False,
                        final_answer=None,
                        error=str(error),
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            all_passed = False
            continue
        except Exception as error:  # noqa: BLE001 - eval 单 case 失败需归类并继续
            print(f"result: fail")
            print(f"error: {error}")
            print(
                "result_data: "
                + json.dumps(
                    build_result_payload(
                        case_index=case_index,
                        eval_case=eval_case,
                        model_profile=model_profile,
                        passed=False,
                        final_answer=None,
                        error=str(error),
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            all_passed = False
            continue

        final_answer = agent_run.final_answer
        passed, missing_keywords = evaluate_answer(final_answer, eval_case.must_contain)
        if passed:
            print("result: pass")
        else:
            print("result: fail")
            print(f"missing: {', '.join(missing_keywords)}")
            all_passed = False

        print("answer:")
        print(final_answer)
        print(
            "result_data: "
            + json.dumps(
                build_result_payload(
                    case_index=case_index,
                    eval_case=eval_case,
                    model_profile=model_profile,
                    passed=passed,
                    final_answer=final_answer,
                    missing_keywords=missing_keywords,
                    payload=agent_run.payload,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    print("overall: pass" if all_passed else "overall: fail")
    return 0 if all_passed else 1


def _build_failure_reasons(missing_keywords: list[str] | None, error: str | None) -> tuple[str, ...]:
    if error is not None:
        return (f"case 执行异常: {error}",)
    if missing_keywords:
        return tuple(f"missing keyword: {keyword}" for keyword in missing_keywords)
    return ()


def _extract_current_metrics(
    payload: Mapping[str, object] | None,
    model_profile: str,
) -> dict[str, object]:
    if payload is None:
        return {
            "model_profile": model_profile,
            "provider": None,
            "model": None,
            "llm_call_count": 0,
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "estimated_cost": None,
        }

    usage_summary = payload.get("usage_summary")
    if not isinstance(usage_summary, Mapping):
        usage_summary = {}

    return {
        "model_profile": _optional_string(payload.get("model_profile")) or model_profile,
        "provider": _optional_string(payload.get("provider")),
        "model": _optional_string(payload.get("model")),
        "llm_call_count": _optional_int(usage_summary.get("llm_call_count")) or 0,
        "total_latency_ms": _optional_float(usage_summary.get("total_latency_ms")) or 0.0,
        "total_tokens": _optional_int(usage_summary.get("total_tokens")) or 0,
        "estimated_tokens": _optional_int(usage_summary.get("estimated_tokens")) or 0,
        "estimated_cost": _optional_float(usage_summary.get("estimated_cost")),
    }


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
