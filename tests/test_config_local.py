from __future__ import annotations

from compass.config import LLMConfig


def test_llm_api_base_default_local() -> None:
    assert LLMConfig().llm_api_base == "http://localhost:8000/v1"


def test_default_config_has_no_proprietary_api_models() -> None:
    config = LLMConfig()
    defaults = [
        config.hyde_model,
        config.vision_model,
        *config.judge_models,
    ]
    forbidden = ["gpt-4o", "claude", "mistral-large-latest", "gpt-4o-mini"]
    assert not any(token in value for value in defaults for token in forbidden)


def test_llm_api_base_env_override(monkeypatch) -> None:
    monkeypatch.setenv("COMPASS_LLM_API_BASE", "http://vllm.local:9000/v1")
    assert LLMConfig().llm_api_base == "http://vllm.local:9000/v1"
