from datetime import date

import compass.chat.engine as chat_engine
from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import build_citations, format_citations


class FakeMemory:
    def query_documents(self, question, as_of, k=12, party_id=None, include_unverified=False):
        return [
            {
                "segment_id": "doc1:p000c000",
                "text": "The party supports democratic accountability and transparent elections.",
                "meta": {
                    "doc_id": "doc1",
                    "country_iso3": "DEU",
                    "party_id": party_id or "41320",
                    "doc_date": "2009-09-01",
                    "doc_type": "manifesto_api_text",
                    "reliability": "official",
                },
            }
        ]


def test_chat_engine_uses_llm_and_returns_citations(monkeypatch):
    def fake_complete(model_name, messages, **kwargs):
        assert "democracy" in messages[-1]["content"]
        return "The party supports democratic accountability [S1]."

    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert response.llm_used is True
    assert response.model_used == "local-test-model"
    assert response.citations[0].ref_id == "S1"
    assert "[S1]" in response.answer


def test_chat_engine_falls_back_when_llm_fails(monkeypatch):
    def broken_complete(*args, **kwargs):
        raise RuntimeError("vLLM offline")

    monkeypatch.setattr(chat_engine, "complete_chat", broken_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about elections?", as_of=date(2009, 9, 27))
    )

    assert response.llm_used is False
    assert "Réponse extractive COMPASS" in response.answer
    assert response.citations


def test_build_citations_and_format_sources():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))

    assert citations[0].country_iso3 == "DEU"
    assert "doc1:p000c000" in format_citations(citations)

def test_gradio_history_normalizer_accepts_tuple_history():
    from apps.chat_gradio import _normalize_history

    messages = _normalize_history([("hello", "hi"), ("next", None)])

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]

def test_gradio_greeting_detector():
    from apps.chat_gradio import _is_greeting

    assert _is_greeting("salut") is True
    assert _is_greeting("What does the party say?") is False

def test_gradio_answer_message_handles_greeting():
    from apps.chat_gradio import _answer_message

    answer = _answer_message("salut", [], engine=None, cutoff=date(2009, 9, 27), party_id=None, k=8)

    assert "Bonjour" in answer