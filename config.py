"""应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_STEPS = 8
MAX_FILE_BYTES = 20 * 1024
MAX_FULL_READ_LINES = 300
MAX_FULL_READ_BYTES = 20_000
MAX_RANGE_READ_LINES = 120
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_SEARCH_RESULTS = 20
RUN_TEST_OUTPUT_MAX_BYTES = 20 * 1024
RUN_TEST_TIMEOUT_SECONDS = 60
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LLM_TIMEOUT_SECONDS = 60
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigError(ValueError):
    """配置缺失或无效。"""


@dataclass(frozen=True)
class LlmConfig:
    """LLM 连接配置。"""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS


def load_dotenv_file(env_file: Path = ENV_FILE) -> None:
    """加载本地 .env 文件，不覆盖已存在的系统环境变量。"""

    if not env_file.is_file():
        return

    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f".env 第 {line_number} 行格式错误，应为 KEY=value")

        name, raw_value = line.split("=", 1)
        name = name.strip()
        value = raw_value.strip()
        if not name:
            raise ConfigError(f".env 第 {line_number} 行变量名不能为空")
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(name, value)


def load_llm_config_from_env() -> LlmConfig:
    """从环境变量读取 LLM 配置。"""

    load_dotenv_file()

    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()

    missing_names: list[str] = []
    if not base_url:
        missing_names.append("LLM_BASE_URL")
    if not api_key:
        missing_names.append("LLM_API_KEY")
    if not model:
        missing_names.append("LLM_MODEL")

    if missing_names:
        missing_text = ", ".join(missing_names)
        raise ConfigError(f"缺少必要环境变量: {missing_text}")

    return LlmConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
