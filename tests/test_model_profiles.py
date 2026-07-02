"""模型 profile 配置加载测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ConfigError, load_model_profile


class TestModelProfiles(unittest.TestCase):
    """验证模型 profile JSON 与环境变量解析。"""

    def setUp(self) -> None:
        self.dotenv_patcher = patch("config.load_dotenv_file")
        self.dotenv_patcher.start()
        self.addCleanup(self.dotenv_patcher.stop)

    def test_default_profile_uses_default_llm_env_names(self) -> None:
        env_values = {
            "DEFAULT_LLM_BASE_URL": "https://example.test/v1",
            "DEFAULT_LLM_API_KEY": "secret-default-key",
            "DEFAULT_LLM_MODEL": "default-model",
        }

        with patch.dict(os.environ, env_values, clear=True):
            profile = load_model_profile("default")

        self.assertEqual("default", profile.name)
        self.assertEqual("openai_compatible", profile.provider)
        self.assertEqual("https://example.test/v1", profile.base_url)
        self.assertEqual("secret-default-key", profile.api_key)
        self.assertEqual("default-model", profile.model)
        self.assertIsInstance(profile.temperature, float)
        self.assertIsInstance(profile.max_output_tokens, int)
        self.assertIsInstance(profile.timeout_seconds, int)
        self.assertIsInstance(profile.retry_count, int)
        self.assertIsNone(profile.pricing)

    def test_fast_and_strong_profiles_use_distinct_env_names(self) -> None:
        env_values = {
            "FAST_LLM_BASE_URL": "https://fast.example.test/v1",
            "FAST_LLM_API_KEY": "secret-fast-key",
            "FAST_LLM_MODEL": "fast-model",
            "STRONG_LLM_BASE_URL": "https://strong.example.test/v1",
            "STRONG_LLM_API_KEY": "secret-strong-key",
            "STRONG_LLM_MODEL": "strong-model",
        }

        with patch.dict(os.environ, env_values, clear=True):
            fast_profile = load_model_profile("fast")
            strong_profile = load_model_profile("strong")

        self.assertEqual("https://fast.example.test/v1", fast_profile.base_url)
        self.assertEqual("secret-fast-key", fast_profile.api_key)
        self.assertEqual("fast-model", fast_profile.model)
        self.assertEqual("https://strong.example.test/v1", strong_profile.base_url)
        self.assertEqual("secret-strong-key", strong_profile.api_key)
        self.assertEqual("strong-model", strong_profile.model)

    def test_unknown_profile_reports_missing_profile(self) -> None:
        env_values = {
            "DEFAULT_LLM_BASE_URL": "https://example.test/v1",
            "DEFAULT_LLM_API_KEY": "secret-default-key",
            "DEFAULT_LLM_MODEL": "default-model",
        }

        with patch.dict(os.environ, env_values, clear=True):
            with self.assertRaisesRegex(ConfigError, "^Unknown model profile: missing$"):
                load_model_profile("missing")

    def test_missing_required_env_reports_names_without_secret_values(self) -> None:
        env_values = {
            "DEFAULT_LLM_BASE_URL": "https://example.test/v1",
            "DEFAULT_LLM_API_KEY": "secret-default-key",
        }

        with patch.dict(os.environ, env_values, clear=True):
            with self.assertRaises(ConfigError) as error_context:
                load_model_profile("default")

        error_text = str(error_context.exception)
        self.assertIn("DEFAULT_LLM_MODEL", error_text)
        self.assertNotIn("secret-default-key", error_text)
        self.assertNotIn("https://example.test/v1", error_text)

    def test_pricing_fields_are_loaded_when_present(self) -> None:
        profile_text = """
{
  "priced": {
    "provider": "openai_compatible",
    "base_url_env": "PRICED_LLM_BASE_URL",
    "api_key_env": "PRICED_LLM_API_KEY",
    "model_env": "PRICED_LLM_MODEL",
    "temperature": 0.2,
    "max_output_tokens": 1234,
    "timeout_seconds": 45,
    "retry_count": 2,
    "pricing": {
      "input_per_1m_tokens": 0.15,
      "output_per_1m_tokens": 0.6
    }
  }
}
""".strip()
        env_values = {
            "PRICED_LLM_BASE_URL": "https://priced.example.test/v1",
            "PRICED_LLM_API_KEY": "secret-priced-key",
            "PRICED_LLM_MODEL": "priced-model",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_path = Path(temp_dir) / "model_profiles.json"
            profiles_path.write_text(profile_text, encoding="utf-8")
            with patch.dict(os.environ, env_values, clear=True):
                profile = load_model_profile("priced", profiles_path=profiles_path)

        pricing = profile.pricing
        self.assertIsNotNone(pricing)
        if pricing is None:
            self.fail("pricing 应该已解析")
        self.assertEqual(0.15, pricing.input_per_1m_tokens)
        self.assertEqual(0.6, pricing.output_per_1m_tokens)

    def test_invalid_provider_and_numeric_fields_are_rejected(self) -> None:
        profile_text = """
{
  "bad": {
    "provider": "other",
    "base_url_env": "BAD_LLM_BASE_URL",
    "api_key_env": "BAD_LLM_API_KEY",
    "model_env": "BAD_LLM_MODEL",
    "temperature": "warm",
    "max_output_tokens": 1234,
    "timeout_seconds": 45,
    "retry_count": 2,
    "pricing": null
  }
}
""".strip()

        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_path = Path(temp_dir) / "model_profiles.json"
            profiles_path.write_text(profile_text, encoding="utf-8")
            with self.assertRaises(ConfigError) as error_context:
                load_model_profile("bad", profiles_path=profiles_path)

        self.assertIn("bad.provider", str(error_context.exception))


if __name__ == "__main__":
    unittest.main()
