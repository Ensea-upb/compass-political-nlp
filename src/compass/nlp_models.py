"""Cached Hugging Face pipelines used across COMPASS components."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transformers import pipeline as hf_pipeline

from compass.config import settings


@lru_cache(maxsize=8)
def _cached_pipeline(task: str, model: str, device_key: str) -> Any:
    kwargs = settings.hf_pipeline_kwargs()
    return hf_pipeline(task, model=model, **kwargs)


def _device_key() -> str:
    return str(settings.hf_pipeline_kwargs())


def nli_pipeline() -> Any:
    """Shared multilingual NLI/text-classification pipeline."""
    return _cached_pipeline("text-classification", settings.nli_model, _device_key())


def zero_shot_pipeline(model: str | None = None) -> Any:
    """Shared zero-shot classification pipeline."""
    return _cached_pipeline("zero-shot-classification", model or settings.nli_model, _device_key())


def political_classifier_pipeline() -> Any:
    """Shared Political DEBATE zero-shot pipeline."""
    return zero_shot_pipeline(settings.political_classifier)