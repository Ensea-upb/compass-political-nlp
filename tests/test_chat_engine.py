from datetime import date

import compass.chat.engine as chat_engine
from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import (
    build_citations,
    build_general_context_items,
    build_messages,
    build_retrieval_query,
    compact_history,
    extract_segment_ids,
    format_citations,
    format_general_context_for_prompt,
    infer_answer_language,
    strip_appended_sources,
)


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


class HybridMemory(FakeMemory):
    def __init__(self):
        self.hybrid_called = False
        self.hybrid_general_called = False

    def query_documents_hybrid(
        self,
        question,
        as_of,
        k=12,
        party_id=None,
        include_unverified=False,
        include_parent_segments=False,
    ):
        if include_parent_segments:
            self.hybrid_general_called = True
            return [
                {
                    "segment_id": "doc1:p000",
                    "text": "General context: this manifesto section discusses democratic institutions.",
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
        self.hybrid_called = True
        return super().query_documents(question, as_of, k, party_id, include_unverified)


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
    assert response.prompt_messages


def test_chat_engine_prefers_hybrid_retrieval(monkeypatch):
    def fake_complete(model_name, messages, **kwargs):
        return "The party supports democratic accountability [S1]."

    memory = HybridMemory()
    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    ChatEngine(memory, model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert memory.hybrid_called is True
    assert memory.hybrid_general_called is True


def test_chat_engine_falls_back_when_llm_fails(monkeypatch):
    def broken_complete(*args, **kwargs):
        raise RuntimeError("vLLM offline")

    monkeypatch.setattr(chat_engine, "complete_chat", broken_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about elections?", as_of=date(2009, 9, 27))
    )

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer
    assert response.citations


def test_build_citations_and_format_sources():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    formatted = format_citations(citations)

    assert citations[0].country_iso3 == "DEU"
    assert "doc1:p000c000" in formatted
    assert "party=41320" in formatted
    assert "excerpt:" in formatted


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


def test_chat_prompt_respects_requested_french():
    assert infer_answer_language("reponds en francais") == "French"


def test_chat_prompt_requires_evidence_linked_claims():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    messages = build_messages(
        "What about democracy?",
        citations,
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320"),
    )

    system = messages[0]["content"]
    assert "Every substantive political claim" in system
    assert "Do not overinterpret" in system
    assert "Do not use outside knowledge" in system
    assert "Never cite [C1]" in system
    assert "provided evidence is insufficient" in system
    assert "GENERAL_CONTEXT - background only" in messages[-1]["content"]
    assert "CITED_EVIDENCE - the only claim-supporting evidence" in messages[-1]["content"]
    assert "Answer contract" in messages[-1]["content"]


def test_strip_appended_sources_removes_model_bibliography():
    answer = "Analysis [S1].\n\nSources\n- [S1] duplicated"

    assert strip_appended_sources(answer) == "Analysis [S1]."


def test_compact_history_trims_sources_and_long_answers():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "A" * 900 + "\n\nSources\n- many"},
    ]

    compacted = compact_history(history, max_chars=80)

    assert len(compacted[-1]["content"]) <= 83
    assert "Sources" not in compacted[-1]["content"]


def test_extract_segment_ids():
    ids = extract_segment_ids("je veux `doc1:p303c001` et doc1:p303c001")

    assert ids == ["doc1:p303c001"]


def test_chat_engine_fetches_exact_segment_id_with_metadata():
    class MemoryWithFetch(FakeMemory):
        def fetch_records_by_ids(self, segment_ids):
            return [
                {
                    "segment_id": segment_ids[0],
                    "text": "Exact corpus passage.",
                    "meta": {
                        "doc_id": "doc1",
                        "country_iso3": "DEU",
                        "party_id": "41320",
                        "doc_date": "2009-09-01",
                        "doc_type": "manifesto_api_text",
                        "reliability": "official",
                    },
                }
            ]

    response = ChatEngine(MemoryWithFetch()).ask(
        ChatRequest(question="je veux le passage doc1:p303c001", as_of=date(2009, 9, 27))
    )

    assert response.llm_used is False
    assert "Exact corpus passage" in response.answer
    assert response.citations[0].segment_id == "doc1:p303c001"
    assert response.citations[0].country_iso3 == "DEU"
    assert "party=41320" in response.answer


def test_economic_retrieval_query_adds_priority_terms():
    query = build_retrieval_query("What are the economic priorities?")

    assert "employment" in query
    assert "wages" in query
    assert "innovation" in query


def test_general_context_is_added_without_becoming_cited_evidence(monkeypatch):
    class MemoryWithGeneralContext:
        def __init__(self):
            self.queries = []

        def query_documents(self, question, as_of, k=12, party_id=None, include_unverified=False, include_parent_segments=False):
            self.queries.append((question, include_parent_segments))
            if include_parent_segments:
                return [
                    {
                        "segment_id": "doc1:p000",
                        "text": "Overall manifesto context: the party frames its program around democratic renewal.",
                        "meta": {
                            "doc_id": "doc1",
                            "country_iso3": "DEU",
                            "party_id": "41320",
                            "doc_date": "2009-09-01",
                            "doc_type": "manifesto_api_text",
                            "reliability": "official",
                        },
                    }
                ]
            return [
                {
                    "segment_id": "doc1:p000c000",
                    "text": "The party supports democratic accountability.",
                    "meta": {
                        "doc_id": "doc1",
                        "country_iso3": "DEU",
                        "party_id": "41320",
                        "doc_date": "2009-09-01",
                        "doc_type": "manifesto_api_text",
                        "reliability": "official",
                        "parent_segment_id": "doc1:p000",
                    },
                }
            ]

        def fetch_by_ids(self, segment_ids):
            return {"doc1:p000": "Parent context: democracy is one theme in the manifesto section."}

    seen = {}

    def fake_complete(model_name, messages, **kwargs):
        seen["prompt"] = messages[-1]["content"]
        return "The party supports democratic accountability [S1]."

    memory = MemoryWithGeneralContext()
    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    response = ChatEngine(memory, model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert response.llm_used is True
    assert response.general_context[0].segment_id == "doc1:p000"
    assert any(include_parent for _, include_parent in memory.queries)
    assert "GENERAL_CONTEXT - background only" in seen["prompt"]
    assert "local_parent_context=Parent context" in seen["prompt"]
    assert response.citations[0].segment_id == "doc1:p000c000"


def test_general_context_format_is_background_only():
    context = build_general_context_items(
        [
            {
                "segment_id": "doc1:p000",
                "text": "Overall program context.",
                "meta": {"country_iso3": "DEU", "party_id": "41320", "doc_date": "2009-09-01", "doc_type": "manifesto_api_text"},
            }
        ]
    )

    formatted = format_general_context_for_prompt(context)

    assert "[C1]" in formatted
    assert "segment=doc1:p000" in formatted


def test_chat_prompt_is_bounded_for_small_vllm_contexts():
    citations = build_citations(
        [
            {
                "segment_id": f"doc1:p{i:03d}c000",
                "text": "Evidence text " + ("x" * 1200),
                "parent_text": "Parent context " + ("y" * 1200),
                "meta": {
                    "doc_id": "doc1",
                    "country_iso3": "DEU",
                    "party_id": "41320",
                    "doc_date": "2009-09-01",
                    "doc_type": "manifesto_api_text",
                    "reliability": "official",
                },
            }
            for i in range(12)
        ]
    )
    general = build_general_context_items(
        [
            {
                "segment_id": f"doc1:p{i:03d}",
                "text": "General context " + ("z" * 1500),
                "meta": {
                    "country_iso3": "DEU",
                    "party_id": "41320",
                    "doc_date": "2009-09-01",
                    "doc_type": "manifesto_api_text",
                },
            }
            for i in range(4)
        ]
    )
    messages = build_messages(
        "What does the party say about democracy?",
        citations,
        ChatRequest(
            question="What does the party say about democracy?",
            as_of=date(2009, 9, 27),
            party_id="41320",
            history=[
                {"role": "user", "content": "previous " + ("h" * 2000)},
                {"role": "assistant", "content": "answer " + ("a" * 2000)},
            ],
        ),
        general,
    )

    prompt = "\n".join(message["content"] for message in messages)
    assert len(prompt) < 7000
    assert "[S6]" in prompt
    assert "[S7]" not in prompt
    assert "[C2]" in prompt
    assert "[C3]" not in prompt
