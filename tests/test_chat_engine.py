from datetime import date

import compass.chat.engine as chat_engine
from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import (
    build_analytical_context,
    build_citations,
    build_general_context_items,
    build_messages,
    build_retrieval_query,
    compact_history,
    extract_segment_ids,
    format_citations,
    format_general_context_for_prompt,
    infer_answer_language,
    route_chat_question,
    strip_appended_sources,
    validate_llm_answer,
    validation_policy_for_route,
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


def test_chat_engine_falls_back_when_llm_cites_general_context(monkeypatch):
    def bad_complete(*args, **kwargs):
        return "The party supports democracy based on the general context [C1]."

    monkeypatch.setattr(chat_engine, "complete_chat", bad_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer


def test_chat_engine_falls_back_when_llm_invents_source_id(monkeypatch):
    def bad_complete(*args, **kwargs):
        return "The party supports democracy [S9]."

    monkeypatch.setattr(chat_engine, "complete_chat", bad_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer


def test_chat_engine_falls_back_when_llm_answers_without_citation(monkeypatch):
    def bad_complete(*args, **kwargs):
        return "The party strongly supports democracy and transparent elections."

    monkeypatch.setattr(chat_engine, "complete_chat", bad_complete)
    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="41320")
    )

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer


def test_chat_engine_routes_corpus_scope_without_retrieval_or_llm(monkeypatch):
    memory = FakeMemory()
    memory.country = "DEU"

    def fail_retrieval(*args, **kwargs):
        raise AssertionError("retrieval must not run for corpus scope")

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM must not run for corpus scope")

    memory.query_documents = fail_retrieval
    monkeypatch.setattr(chat_engine, "complete_chat", fail_llm)
    response = ChatEngine(memory, model_name="local-test-model").ask(
        ChatRequest(
            question="tu es connecte a quel corpus ?",
            as_of=date(2009, 9, 27),
            party_id="41320",
        )
    )

    assert response.llm_used is False
    assert response.retrieval_count == 0
    assert "DEU" in response.answer
    assert "41320" in response.answer
    assert "2009-09-27" in response.answer
    assert "ChromaDB" in response.answer


def test_chat_question_router_distinguishes_scope_lookup_and_evidence():
    assert route_chat_question("tu es connecte a quel corpus ?") == "corpus_scope"
    assert route_chat_question("je veux doc1:p303c001") == "direct_lookup"
    assert route_chat_question("Que dit le parti sur la democratie ?") == "evidence_query"


def test_llm_router_uses_only_allowed_route_labels(monkeypatch):
    monkeypatch.setattr(chat_engine, "complete_chat", lambda *args, **kwargs: "corpus_scope")

    route = route_chat_question(
        "Peux-tu me dire quelles donnees sont actives ?",
        mode="llm",
        model_name="local-test-model",
    )

    assert route == "corpus_scope"


def test_llm_router_falls_back_to_deterministic_on_invalid_output(monkeypatch):
    monkeypatch.setattr(
        chat_engine,
        "complete_chat",
        lambda *args, **kwargs: "I think this is probably about metadata.",
    )

    route = route_chat_question(
        "tu es connecte a quel corpus ?",
        mode="llm",
        model_name="local-test-model",
    )

    assert route == "corpus_scope"


def test_chat_request_defaults_to_deterministic_routing():
    request = ChatRequest(question="q", as_of=date(2009, 9, 27))

    assert request.routing_mode == "deterministic"


def test_answer_validation_policy_depends_on_route():
    assert validation_policy_for_route("direct_lookup") == "none"
    assert validation_policy_for_route("corpus_scope") == "none"
    assert validation_policy_for_route("evidence_query") == "strict_evidence"
    validate_llm_answer("Session corpus description without citation.", [], route="corpus_scope")


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
    assert "Never cite [A]" in system
    assert "provided evidence is insufficient" in system
    assert "ANALYTICAL_CONTEXT - conceptual reading frame" in messages[-1]["content"]
    assert "GENERAL_CONTEXT - background only" in messages[-1]["content"]
    assert "CITED_EVIDENCE - the only claim-supporting evidence" in messages[-1]["content"]
    assert "Answer contract" in messages[-1]["content"]


def test_strip_appended_sources_removes_model_bibliography():
    answer = "Analysis [S1].\n\nSources\n- [S1] duplicated"

    assert strip_appended_sources(answer) == "Analysis [S1]."


def test_validate_llm_answer_allows_insufficiency_without_citation():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))

    validate_llm_answer("The provided evidence is insufficient to answer this question.", citations)


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


def test_analytical_context_is_conceptual_not_evidence():
    context = build_analytical_context("Give me evidence on European integration.")

    assert "[A] Analytical frame" in context
    assert "not evidence" in context
    assert "European integration" in context
    assert "must" not in context.lower() or "evidence" in context.lower()


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
    assert len(prompt) < 5000
    assert "[S4]" in prompt
    assert "[S5]" not in prompt
    assert "[C1]" in prompt
    assert "segment=doc1:p001" not in prompt
