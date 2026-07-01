"""OpenAI-compatible Chat Completions provider。

负责把统一后的 `ModelProfile` 映射为 OpenAI 风格的 chat/completions 调用，
并将响应转换为仓库定义的 `LLMResponse`。

边界与行为：
- endpoint 归一化到 `/chat/completions`
- `HTTPError` 中仅在 429/5xx 按重试次数进行重试；其他状态码直接失败。
- `URLError/timeout` 会重试后上抛。
- 错误消息中会做敏感信息脱敏（api key、Authorization）。
"""

from __future__ import annotations

import json
import socket
import urllib.error
from collections.abc import Sequence
from time import perf_counter
from urllib.request import Request, urlopen

from config import ModelProfile
from model_provider import LLMResponse, TokenUsage
from usage_tracker import normalize_call_usage


class ProviderError(RuntimeError):
    """Provider 调用失败。"""


class OpenAICompatibleProvider:
    """使用已解析 ModelProfile 调用 Chat Completions API。

本类只做协议层转换，不关心上层业务：输入是消息序列，输出是统一响应。
"""

    def __init__(self, profile: ModelProfile) -> None:
        """绑定 profile 并预先规范化 endpoint。"""
        self.profile = profile
        self.endpoint = self._normalize_endpoint(profile.base_url)

    def call(
        self,
        messages: Sequence[dict[str, str]],
        profile_name: str,
    ) -> LLMResponse:
        """发送聊天消息并返回结构化响应。

`profile_name` 参数保留用于接口兼容，当前请求体以对象 profile 为准。
若调用成功：
- 解析响应 JSON
- 抽取第一条 choice 的 content
- 合并并规范化 usage（优先供应商返回）
"""

        del profile_name
        request_body = {
            "model": self.profile.model,
            "messages": list(messages),
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_output_tokens,
        }
        request_bytes = json.dumps(request_body).encode("utf-8")
        attempts = self.profile.retry_count + 1
        started_at = perf_counter()
        last_error: BaseException | None = None

        for attempt_index in range(attempts):
            request = Request(
                self.endpoint,
                data=request_bytes,
                headers={
                    "Authorization": f"Bearer {self.profile.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.profile.timeout_seconds) as response:
                    response_text = response.read().decode("utf-8")
                payload = self._parse_response_json(response_text)
                content = self._extract_message_content(payload)
                latency_ms = (perf_counter() - started_at) * 1000
                usage = normalize_call_usage(
                    prompt_text=self._messages_to_prompt_text(messages),
                    completion_text=content,
                    provider_usage=self._extract_provider_usage(payload),
                    pricing=self.profile.pricing,
                )
                return LLMResponse(
                    content=content,
                    provider="openai_compatible",
                    model=self.profile.model,
                    profile_name=self.profile.name,
                    latency_ms=latency_ms,
                    usage=usage,
                    raw=payload,
                )
            except urllib.error.HTTPError as error:
                last_error = error
                if not self._is_retryable_http_error(error) or attempt_index == attempts - 1:
                    raise ProviderError(f"LLM HTTP 错误 {error.code}") from error
            except (urllib.error.URLError, socket.timeout, TimeoutError) as error:
                last_error = error
                if attempt_index == attempts - 1:
                    raise ProviderError(f"LLM 网络或超时错误: {self._safe_error_reason(error)}") from error

        raise ProviderError(f"LLM 调用失败: {self._safe_error_reason(last_error)}")

    def _normalize_endpoint(self, base_url: str) -> str:
        """将 base_url 映射到 chat/completions 端点。"""
        normalized_url = base_url.rstrip("/")
        if normalized_url.endswith("/chat/completions"):
            return normalized_url
        return f"{normalized_url}/chat/completions"

    def _parse_response_json(self, response_text: str) -> dict[str, object]:
        """解析供应商响应并要求其为 JSON 对象。"""
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise ProviderError("LLM 响应不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise ProviderError("LLM 响应必须是 JSON 对象")
        return payload

    def _extract_message_content(self, payload: dict[str, object]) -> str:
        """从 `choices[0].message.content` 提取 content。

若结构不符合预期，抛 `ProviderError` 让上游统一处理。"""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("LLM 响应缺少 choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderError("LLM choice 格式错误")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("LLM choice 缺少 message")

        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("LLM message.content 不是字符串")
        return content

    def _extract_provider_usage(self, payload: dict[str, object]) -> TokenUsage | None:
        """提取供应商 usage，缺省时返回 None。"""
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return TokenUsage(
            prompt_tokens=self._read_optional_int(usage.get("prompt_tokens")),
            completion_tokens=self._read_optional_int(usage.get("completion_tokens")),
            total_tokens=self._read_optional_int(usage.get("total_tokens")),
        )

    def _read_optional_int(self, value: object) -> int | None:
        """安全读取可选整数：bool 不作为合法整数。"""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _messages_to_prompt_text(self, messages: Sequence[dict[str, str]]) -> str:
        """将 message list 拼为纯文本，供缺失 usage 时估算 token。"""
        return "\n".join(message.get("content", "") for message in messages)

    def _is_retryable_http_error(self, error: urllib.error.HTTPError) -> bool:
        """重试判定：限流 429 或服务器 5xx。"""
        return error.code == 429 or 500 <= error.code <= 599

    def _safe_error_reason(self, error: BaseException | None) -> str:
        """统一脱敏错误细节。

避免日志或终端输出带出 api key 与 Authorization 字段文本。
"""
        if error is None:
            return "unknown"
        reason = getattr(error, "reason", error)
        return str(reason).replace(self.profile.api_key, "[redacted]").replace(
            "Authorization",
            "[redacted-header]",
        )
