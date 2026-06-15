"""LLM 客户端封装。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

from config import LlmConfig


class LlmClientError(RuntimeError):
    """LLM 调用失败。"""


class LlmClient:
    """OpenAI-compatible Chat Completions 客户端。"""

    def __init__(self, config: LlmConfig) -> None:
        self.endpoint = self._normalize_endpoint(config.base_url)
        self.api_key = config.api_key
        self.model = config.model
        self.timeout_seconds = config.timeout_seconds

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """发送聊天消息并返回模型文本。"""

        request_body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
        }
        request_bytes = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=request_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_text = error.read().decode("utf-8", errors="replace")
            raise LlmClientError(f"LLM HTTP 错误 {error.code}: {error_text}") from error
        except urllib.error.URLError as error:
            raise LlmClientError(f"LLM 网络错误: {error.reason}") from error

        try:
            response_payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise LlmClientError("LLM 响应不是有效 JSON") from error

        return self._extract_message_content(response_payload)

    def _normalize_endpoint(self, base_url: str) -> str:
        normalized_url = base_url.rstrip("/")
        if normalized_url.endswith("/chat/completions"):
            return normalized_url
        return f"{normalized_url}/chat/completions"

    def _extract_message_content(self, response_payload: object) -> str:
        if not isinstance(response_payload, dict):
            raise LlmClientError("LLM 响应必须是 JSON 对象")

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmClientError("LLM 响应缺少 choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LlmClientError("LLM choice 格式错误")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LlmClientError("LLM choice 缺少 message")

        message_content = message.get("content")
        if not isinstance(message_content, str):
            raise LlmClientError("LLM message.content 不是字符串")

        return message_content
