"""简单 agent 评估脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from agent import CodeAnalysisAgent
from config import ConfigError, load_llm_config_from_env
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行一组简单 agent eval")
    parser.add_argument("--repo", default=".", help="要评估的本地 repo 路径，默认当前目录")
    parser.add_argument(
        "--eval-file",
        default=str(DEFAULT_EVAL_FILE),
        help="eval 用例 JSON 文件路径，默认 eval_case.json",
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


def run_agent(repo_path: Path, question: str) -> str:
    """调用 agent 并返回最终答案。"""

    repository_tools = RepositoryTools(repo_path)
    run_logger = RunLogger(repo_path=repo_path, user_task=question)
    llm_config = load_llm_config_from_env()
    llm_client = LlmClient(llm_config)
    agent = CodeAnalysisAgent(
        llm_client=llm_client,
        repository_tools=repository_tools,
        run_logger=run_logger,
    )
    return agent.answer(question)


def evaluate_answer(final_answer: str, must_contain: tuple[str, ...]) -> tuple[bool, list[str]]:
    """检查最终答案是否包含所有必需关键词。"""

    missing_keywords = [keyword for keyword in must_contain if keyword not in final_answer]
    return not missing_keywords, missing_keywords


def main() -> int:
    """运行 eval 并输出 pass/fail。"""

    args = parse_args()
    repo_path = Path(args.repo).expanduser().resolve()
    eval_file = Path(args.eval_file).expanduser().resolve()

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
            final_answer = run_agent(repo_path, eval_case.question)
        except (ConfigError, LlmClientError, ToolError, OSError) as error:
            print(f"result: fail")
            print(f"error: {error}")
            all_passed = False
            continue

        passed, missing_keywords = evaluate_answer(final_answer, eval_case.must_contain)
        if passed:
            print("result: pass")
        else:
            print("result: fail")
            print(f"missing: {', '.join(missing_keywords)}")
            all_passed = False

        print("answer:")
        print(final_answer)

    print("overall: pass" if all_passed else "overall: fail")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
