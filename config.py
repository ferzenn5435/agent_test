"""应用配置与模型 profile 解析。

职责边界：
- 常量定义：CLI 限流阈值、日志目录、LLM 默认超时、Profile 配置文件路径。
- 环境变量加载：`load_dotenv_file()` 只做本地 `.env` 的“兜底注入”，不覆盖系统变量。
- 模型 Profile 解析：从 `model_profiles.json` 读取并校验 provider、env key 名称与数值类型。

安全边界：
- 系统环境变量优先，`.env` 仅作为默认来源。
- `load_model_profile()` 不接触密钥明文，只读取 env 名称并按名称取值。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
MODEL_PROFILES_FILE = PROJECT_ROOT / "model_profiles.json"
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"


class ConfigError(ValueError):
    """配置缺失或无效。

该异常仅用于配置读取阶段，便于上层对 CLI 与日志统一处理错误输出。
    """


@dataclass(frozen=True)
class LlmConfig:
    """LLM 连接配置。

该结构主要兼容旧调用链：在某些测试/兼容路径中直接通过环境变量组装
完整连接参数。
    """

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ModelPricing:
    """模型价格配置。

字段为“每百万 token 价格”，后续用量统计在提供价格且 token 可计算时
才会产出估算成本。
    """

    input_per_1m_tokens: float
    output_per_1m_tokens: float


@dataclass(frozen=True)
class ModelProfile:
    """解析后的模型 profile 配置。

profile name 一般来源于 CLI 的 `--model-profile`。
`default`/`fast`/`strong` 需分别在 `model_profiles.json` 中给出对应的
env 名称映射，值解析时按 profile 中定义的环境变量实际取值。
    """

    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    retry_count: int
    pricing: ModelPricing | None


def load_dotenv_file(env_file: Path = ENV_FILE) -> None:
    """加载本地 `.env` 文件，不覆盖已存在的系统环境变量。

实现是“只写入不存在的键”，确保系统环境变量始终具有更高优先级。
逐行校验 `KEY=value`，并支持被引号包裹的值。
    """

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
    """从环境变量读取旧版 LLM 连接配置。

用于兼容没有 profile 的旧路径：读取 `LLM_BASE_URL / LLM_API_KEY / LLM_MODEL`。
当任一变量缺失时抛出 `ConfigError`。
    """

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


def load_model_profile(
    profile_name: str = "default",
    profiles_path: Path = MODEL_PROFILES_FILE,
) -> ModelProfile:
    """读取并解析指定模型 profile。

返回已完整校验的 `ModelProfile`，供 `LlmClient` 和 provider 使用。
支持 `default`、`fast`、`strong` 等命名 profile；
如 `profile_name` 在文件中不存在会立即报错。

边界说明：
- `provider` 目前要求为 `openai_compatible`；其他值将触发 `ConfigError`。
- `base_url_env/api_key_env/model_env` 仅存 env 名称，不直接存敏感值。
- 解析失败或环境变量缺失时保留原始 field 信息在异常中，方便快速定位。
    """

    load_dotenv_file()
    profiles = _load_model_profile_documents(profiles_path)
    if profile_name not in profiles:
        raise ConfigError(f"Unknown model profile: {profile_name}")

    raw_profile = profiles[profile_name]
    if not isinstance(raw_profile, dict):
        raise ConfigError(f"model profile {profile_name} 必须是对象")

    provider = _read_required_string(raw_profile, "provider", f"{profile_name}.provider")
    if provider != OPENAI_COMPATIBLE_PROVIDER:
        raise ConfigError(
            f"{profile_name}.provider 仅支持 {OPENAI_COMPATIBLE_PROVIDER}: {provider}"
        )

    base_url_env = _read_required_string(
        raw_profile,
        "base_url_env",
        f"{profile_name}.base_url_env",
    )
    api_key_env = _read_required_string(
        raw_profile,
        "api_key_env",
        f"{profile_name}.api_key_env",
    )
    model_env = _read_required_string(
        raw_profile,
        "model_env",
        f"{profile_name}.model_env",
    )
    base_url = _read_env_value(base_url_env)
    api_key = _read_env_value(api_key_env)
    model = _read_env_value(model_env)

    missing_names = [
        env_name
        for env_name, value in [
            (base_url_env, base_url),
            (api_key_env, api_key),
            (model_env, model),
        ]
        if not value
    ]
    if missing_names:
        missing_text = ", ".join(missing_names)
        raise ConfigError(
            f"缺少 model profile {profile_name} 必要环境变量: {missing_text}"
        )

    return ModelProfile(
        name=profile_name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=_read_number(raw_profile, "temperature", f"{profile_name}.temperature"),
        max_output_tokens=_read_positive_int(
            raw_profile,
            "max_output_tokens",
            f"{profile_name}.max_output_tokens",
        ),
        timeout_seconds=_read_positive_int(
            raw_profile,
            "timeout_seconds",
            f"{profile_name}.timeout_seconds",
        ),
        retry_count=_read_non_negative_int(
            raw_profile,
            "retry_count",
            f"{profile_name}.retry_count",
        ),
        pricing=_read_pricing(raw_profile.get("pricing"), profile_name),
    )


def _load_model_profile_documents(profiles_path: Path) -> dict[str, Any]:
    """读取并解析 profile 文件。

仅做 JSON 结构校验，返回字典结构供 `load_model_profile` 做逐字段校验。
"""
    if not profiles_path.is_file():
        raise ConfigError(f"model profile 配置文件不存在: {profiles_path}")

    try:
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"model profile 配置文件 JSON 无效: {error.msg}") from error

    if not isinstance(profiles, dict):
        raise ConfigError("model profile 配置文件根节点必须是对象")
    return profiles


def _read_required_string(profile: dict[str, Any], key: str, field_name: str) -> str:
    """读取必填字符串字段，空字符串视为非法。"""
    value = profile.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _read_env_value(env_name: str) -> str:
    """读取环境变量并去空白，用于 profile 中的 env 名称映射。"""
    return os.environ.get(env_name, "").strip()


def _read_number(profile: dict[str, Any], key: str, field_name: str) -> float:
    """读取数值型配置，拒绝 bool 与非数值类型。"""
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} 必须是数字")
    return float(value)


def _read_positive_int(profile: dict[str, Any], key: str, field_name: str) -> int:
    """读取正整数型配置，供 timeout/retry/max_output_tokens 等字段。"""
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} 必须是正整数")
    return value


def _read_non_negative_int(profile: dict[str, Any], key: str, field_name: str) -> int:
    """读取非负整数型配置，供 retry_count 等可为 0 的字段。"""
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{field_name} 必须是非负整数")
    return value


def _read_pricing(raw_pricing: object, profile_name: str) -> ModelPricing | None:
    """读取可选 pricing；支持 `null` 表示不计费。

当存在 pricing 时要求包含输入/输出两类单价。
"""
    if raw_pricing is None:
        return None
    if not isinstance(raw_pricing, dict):
        raise ConfigError(f"{profile_name}.pricing 必须是对象或 null")

    return ModelPricing(
        input_per_1m_tokens=_read_number(
            raw_pricing,
            "input_per_1m_tokens",
            f"{profile_name}.pricing.input_per_1m_tokens",
        ),
        output_per_1m_tokens=_read_number(
            raw_pricing,
            "output_per_1m_tokens",
            f"{profile_name}.pricing.output_per_1m_tokens",
        ),
    )
