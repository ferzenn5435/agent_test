"""命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import CodeAnalysisAgent
from config import ConfigError, load_llm_config_from_env
from llm_client import LlmClient, LlmClientError
from logger import RunLogger
from tools import RepositoryTools, ToolError


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="最小本地代码库分析 agent")
    parser.add_argument("repo_path", help="要分析的本地 repo 路径")
    parser.add_argument("question", help="用户问题")
    return parser.parse_args()


def main() -> int:
    """运行命令行程序。"""

    args = parse_args()
    repo_path = Path(args.repo_path).expanduser().resolve()
    question = str(args.question).strip()
    if not question:
        print("错误: question 不能为空", file=sys.stderr)
        return 1

    run_logger: RunLogger | None = None
    try:
        repository_tools = RepositoryTools(repo_path)
        run_logger = RunLogger(repo_path=repo_path, user_task=question)
        llm_config = load_llm_config_from_env()
        llm_client = LlmClient(llm_config)
        agent = CodeAnalysisAgent(
            llm_client=llm_client,
            repository_tools=repository_tools,
            run_logger=run_logger,
        )
        final_answer = agent.answer(question)
    except (ConfigError, LlmClientError, ToolError, OSError) as error:
        error_message = f"运行失败: {error}"
        if run_logger is not None:
            run_logger.set_error(error_message)
        print(error_message, file=sys.stderr)
        return 1

    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
