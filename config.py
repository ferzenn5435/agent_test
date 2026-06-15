"""应用配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_STEPS = 8
MAX_FILE_BYTES = 20 * 1024
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LLM_TIMEOUT_SECONDS = 60


class ConfigError(ValueError):
    """配置缺失或无效。"""


@dataclass(frozen=True)
class LlmConfig:
    """LLM 连接配置。"""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS


def load_llm_config_from_env() -> LlmConfig:
    """从环境变量读取 LLM 配置。"""

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
