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
    assert not any(token in str(value) for value in defaults for token in forbidden)


def test_llm_api_base_env_override(monkeypatch) -> None:
    monkeypatch.setenv("COMPASS_LLM_API_BASE", "http://vllm.local:9000/v1")
    assert LLMConfig().llm_api_base == "http://vllm.local:9000/v1"


def test_judge_models_env_accepts_comma_separated_list(monkeypatch) -> None:
    monkeypatch.setenv("COMPASS_JUDGE_MODELS", "Qwen/Qwen2.5-7B-Instruct")
    assert LLMConfig().judge_models == ["Qwen/Qwen2.5-7B-Instruct"]


def test_hf_device_cpu_sets_cpu_pipeline_device(monkeypatch) -> None:
    monkeypatch.setenv("COMPASS_HF_DEVICE", "cpu")
    config = LLMConfig()
    assert config.hf_pipeline_kwargs() == {"device": -1}
    assert config.hf_model_device() == "cpu"


def test_default_config_is_onyxia_16gb_profile() -> None:
    config = LLMConfig()
    assert config.judge_models == ["Qwen/Qwen2.5-3B-Instruct"]
    assert config.hyde_model == "Qwen/Qwen2.5-3B-Instruct"
    assert config.vision_model is None