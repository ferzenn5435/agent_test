"""固定 LLM provider 包。

对外导出当前仓库支持的 provider 实现：
- `MockProvider`：测试与离线场景；
- `OpenAICompatibleProvider`：生产/默认 LLM 调用路径。

这样便于 `llm_client` 层统一 import 与 type 检查。
"""

from providers.mock_provider import MockProvider
from providers.openai_compatible import OpenAICompatibleProvider, ProviderError

__all__ = ["MockProvider", "OpenAICompatibleProvider", "ProviderError"]
