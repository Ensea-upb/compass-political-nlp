from __future__ import annotations

from types import SimpleNamespace

from compass.config import LLMConfig
from compass.llm_client import complete_chat


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(text="fallback text")])


class _FakeChatCompletions:
    def create(self, **kwargs):
        raise RuntimeError("vLLM chat route failed")


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())
        self.completions = _FakeCompletions()


def test_local_client_falls_back_to_completions(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    config = LLMConfig(llm_backend="local")

    result = complete_chat(
        "Qwen/Qwen2.5-3B-Instruct",
        [{"role": "user", "content": "Bonjour"}],
        config=config,
    )

    assert result == "fallback text"