"""LLM client abstraction for COMPASS.

Local mode talks directly to an OpenAI-compatible endpoint such as vLLM. API
mode keeps LiteLLM as the multi-provider adapter.
"""

from __future__ import annotations

from typing import Any

from compass.config import LLMConfig, settings


def complete_chat(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, str] | None = None,
    config: LLMConfig | None = None,
) -> str:
    """Return the assistant message content for a chat completion."""
    cfg = config or settings
    temperature = cfg.llm_temperature if temperature is None else temperature
    max_tokens = cfg.llm_max_tokens if max_tokens is None else max_tokens

    if cfg.llm_backend == "local":
        return _complete_openai_compatible(
            cfg,
            model_name,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    return _complete_litellm(
        model_name,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        config=cfg,
    )


def _complete_openai_compatible(
    cfg: LLMConfig,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, str] | None,
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Local LLM mode requires the optional dependency 'openai'. "
            "Install full requirements with: pip install -r requirements-full.txt"
        ) from exc

    client = OpenAI(base_url=cfg.llm_api_base, api_key=cfg.llm_api_key)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        if response_format is None:
            raise
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _complete_litellm(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, str] | None,
    config: LLMConfig,
) -> str:
    try:
        import litellm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "API LLM mode requires the optional dependency 'litellm'. "
            "Install full requirements with: pip install -r requirements-full.txt"
        ) from exc

    kwargs: dict[str, Any] = dict(
        config.litellm_kwargs(model_name),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        if response_format is not None:
            resp = litellm.completion(response_format=response_format, **kwargs)
        else:
            resp = litellm.completion(**kwargs)
    except litellm.BadRequestError:
        if response_format is None:
            raise
        resp = litellm.completion(**kwargs)
    return (resp.choices[0].message.content or "").strip()