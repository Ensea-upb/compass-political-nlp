from datetime import date
from types import SimpleNamespace

import compass.chat.engine as chat_engine
import pytest
from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import (
    build_analytical_context,
    build_citations,
    build_general_context_items,
    build_messages,
    compact_history,
    extract_segment_ids,
    format_citations,
    format_general_context_for_prompt,
    infer_answer_language,
    route_chat_question,
    strip_appended_sources,
    validate_llm_answer,
    validate_semantic_grounding,
    validation_policy_for_route,
)


@pytest.fixture(autouse=True)
def _disable_nli_by_default(monkeypatch):
    monkeypatch.setattr(chat_engine.settings, "chat_semantic_validation_enabled", False)


class FakeMemory:
    country = "TST"

    def describe_corpus(self, as_of=None, party_id=None):
        return {
            "country_iso3": "TST",
            "n_documents": 1,
            "parties": [{"party_id": party_id or "P100", "name": "Test Party"}],
            "document_dates": ["2009-09-01"],
            "document_types": ["manifesto_api_text"],
        }

    def query_documents(self, question, as_of, k=12, party_id=None, include_unverified=False):
        return [
            {
                "segment_id": "doc1:p000c000",
                "text": "The party supports democratic accountability and transparent elections.",
                "meta": {
                    "doc_id": "doc1",
                    "country_iso3": "TST",
                    "party_id": party_id or "P100",
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


def test_chat_engine_runs_scientific_pipeline_route():
    class Service:
        def analyze(self, variable_id, **kwargs):
            assert variable_id == "v2pavote"
            assert kwargs["party_id"] == "P100"
            assert kwargs["election_id"] == "E1"
            return SimpleNamespace(
                variable_id=variable_id,
                abstained=False,
                score=3.0,
                confidence=0.75,
                attribution_checked=True,
                residual_uncertainty="aucun manque",
                main_evidence=[],
                counter_evidence=[],
            )

    response = ChatEngine(FakeMemory(), scientific_service=Service()).ask(ChatRequest(
        question="/analyse v2pavote",
        as_of=date(2009, 9, 27),
        party_id="P100",
        election_id="E1",
    ))

    assert response.route == "SCIENTIFIC_ANALYSIS"
    assert "Score : 3.0" in response.answer
    assert "Attribution NLI vérifiée : oui" in response.answer


def test_chat_engine_exposes_scientific_validation_and_variable_routes():
    class Service:
        def available_variables(self):
            return ["v1", "v2"]

        def validate_cached(self, variable_id):
            assert variable_id == "v1"
            return SimpleNamespace(
                stratum="chat_session", n_cases=1, n_abstentions=0,
                mae=0.1, spearman=1.0, interval_coverage=1.0,
                ece=0.05, attribution_rate=1.0,
            )

        def contamination_check(self, variable_id, **kwargs):
            assert variable_id == "v1"
            return [{"model": "judge", "claims_knowledge": False, "raw": "inconnu"}]

    engine = ChatEngine(FakeMemory(), scientific_service=Service())
    variables = engine.ask(ChatRequest(question="/variables", as_of=date(2009, 9, 27)))
    validation = engine.ask(ChatRequest(question="/valider v1", as_of=date(2009, 9, 27)))
    contamination = engine.ask(ChatRequest(
        question="/contamination v1", as_of=date(2009, 9, 27), party_id="P100",
    ))

    assert variables.route == "SCIENTIFIC_VARIABLES"
    assert "v1" in variables.answer
    assert validation.route == "SCIENTIFIC_VALIDATION"
    assert "MAE : 0.1000" in validation.answer
    assert contamination.route == "SCIENTIFIC_CONTAMINATION"
    assert "claims_knowledge=False" in contamination.answer


def test_chat_engine_distinguishes_retrieval_and_prompt_evidence_budgets(monkeypatch):
    class ManyResultsMemory(FakeMemory):
        def query_documents(self, question, as_of, k=12, party_id=None, include_unverified=False):
            base = super().query_documents(question, as_of, k, party_id, include_unverified)[0]
            return [
                {
                    **base,
                    "segment_id": f"doc1:p{index:03d}c000",
                    "text": f"Substantive political evidence passage number {index} supports democracy.",
                }
                for index in range(8)
            ]

    monkeypatch.setattr(chat_engine, "complete_chat", lambda *args, **kwargs: "Supported answer [S4].")
    response = ChatEngine(ManyResultsMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27))
    )

    assert response.retrieval_count == 8
    assert response.prompt_citation_count == 4
    assert len(response.citations) == 4


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


def test_chat_engine_uses_structured_llm_subqueries(monkeypatch):
    class QueryMemory(FakeMemory):
        def __init__(self):
            self.queries = []

        def query_documents(self, question, as_of, k=12, party_id=None, include_unverified=False):
            self.queries.append(question)
            return super().query_documents(question, as_of, k, party_id, include_unverified)

    def fake_complete(model_name, messages, **kwargs):
        if kwargs.get("response_format"):
            return (
                '{"actors":["Test Party"],"themes":["constitutional reform"],'
                '"period":null,"answer_type":"position","language":"en",'
                '"subqueries":["constitutional reform institutions",'
                '"constitutional change commitments evidence"]}'
            )
        assert "QUESTION_ANALYSIS" not in messages[-1]["content"]
        assert "RETRIEVAL_TRACE" not in messages[-1]["content"]
        return "The party supports democratic accountability [S1]."

    memory = QueryMemory()
    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    response = ChatEngine(memory, model_name="local-test-model").ask(
        ChatRequest(
            question="What does this party say about constitutional reform?",
            as_of=date(2009, 9, 27),
            party_id="P100",
        )
    )

    evidence_queries = [query for query in memory.queries if "manifesto overall" not in query]
    assert evidence_queries[:3] == [
        "What does this party say about constitutional reform?",
        "constitutional reform institutions",
        "constitutional change commitments evidence",
    ]
    assert any("conditions limits exceptions" in query for query in evidence_queries)
    assert any("opposition rejection criticism" in query for query in evidence_queries)
    assert response.query_analysis["method"] == "llm"
    assert response.query_analysis["themes"] == ["constitutional reform"]


def test_chat_injects_inferred_graph_context_for_relational_questions(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.calls = []

        def query_party(self, party_id, as_of, k_hops, top_k):
            self.calls.append((party_id, as_of, k_hops, top_k))
            return [{"summary": "[INFERRED] Test Party - alliance - Partner"}]

    seen = {}
    graph = FakeGraph()

    def fake_complete(model_name, messages, **kwargs):
        seen["prompt"] = messages[-1]["content"]
        return "The manifesto mentions cooperation with partners [S1]."

    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    response = ChatEngine(
        FakeMemory(), model_name="local-test-model", graph=graph,
    ).ask(ChatRequest(
        question="What alliances and partners does the party mention?",
        as_of=date(2009, 9, 27),
        party_id="P100",
    ))

    assert graph.calls
    assert response.graph_context
    assert "RELATIONAL_CONTEXT" in seen["prompt"]
    assert "[R1] [INFERRED, NOT EVIDENCE]" in seen["prompt"]
    assert response.citations[0].ref_id == "S1"


def test_chat_rejects_graph_context_as_a_citation(monkeypatch):
    class FakeGraph:
        def query_party(self, **kwargs):
            return [{"summary": "[INFERRED] Party - alliance - Partner"}]

    monkeypatch.setattr(
        chat_engine,
        "complete_chat",
        lambda *args, **kwargs: "The party has an alliance [R1].",
    )
    response = ChatEngine(
        FakeMemory(), model_name="local-test-model", graph=FakeGraph(),
    ).ask(ChatRequest(
        question="What alliance does the party have?",
        as_of=date(2009, 9, 27),
        party_id="P100",
    ))

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer


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
            party_id="P100",
        )
    )

    assert response.llm_used is False
    assert response.retrieval_count == 0
    assert "TST" in response.answer
    assert "P100" in response.answer
    assert "Test Party" in response.answer
    assert "Documents distincts : 1" in response.answer
    assert "n'est pas nécessairement" in response.answer
    assert "2009-09-27" in response.answer
    assert "ChromaDB" in response.answer


def test_chat_question_router_distinguishes_scope_lookup_and_evidence():
    assert route_chat_question("tu es connecte a quel corpus ?") == "corpus_scope"
    assert route_chat_question("je veux doc1:p303c001") == "direct_lookup"
    assert route_chat_question("Que dit le parti sur la democratie ?") == "evidence_query"


def test_scope_routes_are_intentional_and_comparison_is_party_aware():
    entities = ["P100", "P200", "Alpha Party", "Beta Party"]

    assert route_chat_question("quels étaient tous les partis en 2009 ?") == "OUT_OF_CORPUS"
    assert route_chat_question("compare P100 et P200", party_entities=entities) == "COMPARISON_NEEDS_MORE_CORPUS"
    assert route_chat_question("compare la position du parti sur X et Y", party_entities=entities) == "evidence_query"
    assert route_chat_question("qui gouvernait ?") == "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA"
    assert route_chat_question("quelles sont les sources exactes ?") == "FOLLOW_UP_SOURCES"
    assert route_chat_question("/analyse v2pavote") == "SCIENTIFIC_ANALYSIS"
    assert route_chat_question("/valider v2pavote") == "SCIENTIFIC_VALIDATION"
    assert route_chat_question("/variables") == "SCIENTIFIC_VARIABLES"
    assert route_chat_question("/contamination v2pavote") == "SCIENTIFIC_CONTAMINATION"


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
    assert validation_policy_for_route("OUT_OF_CORPUS") == "none"
    assert validation_policy_for_route("COMPARISON_NEEDS_MORE_CORPUS") == "none"
    assert validation_policy_for_route("ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA") == "none"
    assert validation_policy_for_route("FOLLOW_UP_SOURCES") == "none"
    assert validation_policy_for_route("SCIENTIFIC_ANALYSIS") == "none"
    assert validation_policy_for_route("SCIENTIFIC_VALIDATION") == "none"
    assert validation_policy_for_route("SCIENTIFIC_VARIABLES") == "none"
    assert validation_policy_for_route("SCIENTIFIC_CONTAMINATION") == "none"
    validate_llm_answer("Session corpus description without citation.", [], route="corpus_scope")


def test_build_citations_and_format_sources():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    formatted = format_citations(citations)

    assert citations[0].country_iso3 == "TST"
    assert "doc1:p000c000" in formatted
    assert "party=P100" in formatted
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
    assert "RELATIONAL_CONTEXT" in system
    assert "[R1]" in system
    assert "provided evidence is insufficient" in system
    assert "absence of evidence as evidence of absence" in system
    assert "list is exhaustive" in system
    assert "ANALYTICAL_CONTEXT - conceptual reading frame" in messages[-1]["content"]
    assert "GENERAL_CONTEXT - background only" in messages[-1]["content"]
    assert "RELATIONAL_CONTEXT - inferred graph" in messages[-1]["content"]
    assert "CITED_EVIDENCE - the only claim-supporting evidence" in messages[-1]["content"]
    assert "Answer contract" in messages[-1]["content"]


def test_generation_prompt_excludes_retrieval_telemetry():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    messages = build_messages(
        "What about democracy?",
        citations,
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27)),
        retrieval=chat_engine.RetrievalBundle(trace=[{
            "stage": "query",
            "lane": "primary",
            "query": "democracy",
        }]),
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "QUESTION_ANALYSIS" not in prompt
    assert "RETRIEVAL_TRACE" not in prompt


def test_build_citations_deduplicates_text_and_parent_records():
    parent_text = "This parent passage provides substantial democratic policy context."
    child_text = "The party explicitly supports democratic participation and accountability."
    citations = build_citations([
        {
            "segment_id": "doc:p001c001",
            "text": child_text,
            "parent_text": parent_text,
            "meta": {"parent_segment_id": "doc:p001"},
            "evidence_role": "primary",
        },
        {
            "segment_id": "duplicate:p001c001",
            "text": "  THE PARTY explicitly supports democratic participation and accountability. ",
            "parent_text": parent_text,
            "meta": {"parent_segment_id": "doc:p001"},
            "evidence_role": "nuance",
        },
        {
            "segment_id": "doc:p001",
            "text": parent_text,
            "meta": {},
            "evidence_role": "counter",
        },
    ])

    assert len(citations) == 1
    assert citations[0].segment_id == "doc:p001c001"


def test_short_fragment_without_parent_is_not_citable():
    citations = build_citations([
        {"segment_id": "doc:p001c001", "text": "Protecting citizens' rights.", "meta": {}},
        {
            "segment_id": "doc:p002c001",
            "text": "Direct democracy.",
            "parent_text": "The party supports referendums and stronger democratic citizen participation.",
            "meta": {"parent_segment_id": "doc:p002"},
        },
    ])

    assert [citation.segment_id for citation in citations] == ["doc:p002c001"]


def test_general_context_does_not_repeat_cited_parent_or_text():
    parent_text = "The manifesto presents a broad and substantive democratic orientation."
    citations = build_citations([{
        "segment_id": "doc:p001c001",
        "text": "The party supports accountable democratic institutions and participation.",
        "parent_text": parent_text,
        "meta": {"parent_segment_id": "doc:p001"},
    }])
    general = build_general_context_items([
        {"segment_id": "doc:p001", "text": parent_text, "meta": {}},
        {"segment_id": "duplicate:p001", "text": parent_text.upper(), "meta": {}},
    ])
    messages = build_messages(
        "What about democracy?",
        citations,
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27)),
        general,
    )
    prompt = messages[-1]["content"]

    assert prompt.casefold().count(parent_text.casefold()) == 1
    assert "segment=doc:p001" not in prompt
    assert "No separate general context retrieved." in prompt


def test_current_question_is_removed_from_history_before_prompting():
    question = "What does the party say about democracy?"
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    messages = build_messages(
        question,
        citations,
        ChatRequest(
            question=question,
            as_of=date(2009, 9, 27),
            history=[
                {"role": "assistant", "content": "Previous answer [S1]."},
                {"role": "user", "content": question},
            ],
        ),
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert prompt.count(question) == 1


def test_strip_appended_sources_removes_model_bibliography():
    answer = "Analysis [S1].\n\nSources\n- [S1] duplicated"

    assert strip_appended_sources(answer) == "Analysis [S1]."


def test_validate_llm_answer_allows_insufficiency_without_citation():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))

    validate_llm_answer("The provided evidence is insufficient to answer this question.", citations)


def test_optional_semantic_grounding_accepts_entailment(monkeypatch):
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))
    monkeypatch.setattr(
        "compass.nlp_models.nli_pipeline",
        lambda: lambda pair: {"label": "entailment", "score": 0.99},
    )

    validate_semantic_grounding("The party supports democratic accountability [S1].", citations)


def test_semantic_grounding_accepts_jointly_supported_synthesis(monkeypatch):
    records = FakeMemory().query_documents("q", date(2009, 9, 27))
    records.append({
        **records[0],
        "segment_id": "doc1:p001c000",
        "text": "The party explicitly supports transparent and democratic elections.",
    })
    citations = build_citations(records)

    def classifier(pair):
        if "\n" in pair["text"]:
            return {"label": "entailment", "score": 0.95}
        return {"label": "neutral", "score": 0.8}

    monkeypatch.setattr("compass.nlp_models.nli_pipeline", lambda: classifier)

    validate_semantic_grounding(
        "Taken together, the passages suggest support for democratic accountability and elections [S1] [S2].",
        citations,
    )


def test_chat_repairs_rejected_nli_claim_with_same_evidence(monkeypatch):
    class CountingMemory(FakeMemory):
        def __init__(self):
            self.calls = 0

        def query_documents(self, *args, **kwargs):
            self.calls += 1
            return super().query_documents(*args, **kwargs)

    answers = iter([
        "The party abolishes democratic accountability [S1].",
        "The party supports democratic accountability [S1].",
    ])

    def fake_complete(model, messages, **kwargs):
        return next(answers)

    def classifier(pair):
        if "abolishes" in pair["text_pair"]:
            return {"label": "neutral", "score": 0.91}
        return {"label": "entailment", "score": 0.98}

    memory = CountingMemory()
    monkeypatch.setattr(chat_engine.settings, "chat_query_analysis_enabled", False)
    monkeypatch.setattr(chat_engine.settings, "chat_semantic_validation_enabled", True)
    monkeypatch.setattr(chat_engine.settings, "chat_repair_max_attempts", 1)
    monkeypatch.setattr(chat_engine, "complete_chat", fake_complete)
    monkeypatch.setattr("compass.nlp_models.nli_pipeline", lambda: classifier)

    response = ChatEngine(memory, model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="P100")
    )

    assert response.llm_used is True
    assert "supports democratic accountability" in response.answer
    assert any(step["status"] == "rejected" for step in response.validation_trace)
    assert response.validation_trace[-1]["status"] == "accepted"
    assert response.prompt_messages[-1]["role"] == "user"
    assert "same evidence" in response.prompt_messages[-1]["content"]


def test_chat_falls_back_after_repair_is_exhausted(monkeypatch):
    monkeypatch.setattr(chat_engine.settings, "chat_query_analysis_enabled", False)
    monkeypatch.setattr(chat_engine.settings, "chat_semantic_validation_enabled", True)
    monkeypatch.setattr(chat_engine.settings, "chat_repair_max_attempts", 1)
    monkeypatch.setattr(
        chat_engine,
        "complete_chat",
        lambda *args, **kwargs: "The party abolishes accountability [S1].",
    )
    monkeypatch.setattr(
        "compass.nlp_models.nli_pipeline",
        lambda: lambda pair: {"label": "neutral", "score": 0.99},
    )

    response = ChatEngine(FakeMemory(), model_name="local-test-model").ask(
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27), party_id="P100")
    )

    assert response.llm_used is False
    assert "Reponse extractive COMPASS" in response.answer
    rejected = [step for step in response.validation_trace if step["status"] == "rejected"]
    assert len(rejected) >= 2


def test_syntax_validator_requires_citation_for_each_claim():
    citations = build_citations(FakeMemory().query_documents("q", date(2009, 9, 27)))

    with pytest.raises(chat_engine.AnswerContractError, match="uncited substantive claim"):
        validate_llm_answer(
            "The party supports accountability [S1]. It also supports an uncited reform.",
            citations,
        )


def test_prompt_separates_primary_nuance_and_counter_evidence():
    records = []
    for role, index in (("primary", 1), ("nuance", 2), ("counter", 3)):
        record = FakeMemory().query_documents("q", date(2009, 9, 27))[0]
        records.append({
            **record,
            "segment_id": f"doc1:p00{index}c000",
            "text": f"The {role} passage provides distinct substantive evidence about democracy.",
            "evidence_role": role,
        })
    citations = build_citations(records)
    messages = build_messages(
        "What about democracy?",
        citations,
        ChatRequest(question="What about democracy?", as_of=date(2009, 9, 27)),
    )
    prompt = messages[-1]["content"]

    assert "PRIMARY_EVIDENCE" in prompt
    assert "NUANCE_EVIDENCE" in prompt
    assert "COUNTER_EVIDENCE_CANDIDATES" in prompt


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


def test_direct_lookup_fallback_warns_when_metadata_are_unavailable():
    class LegacyMemory:
        country = "TST"

        def describe_corpus(self, **kwargs):
            return {"country_iso3": "TST", "n_documents": 1}

        def fetch_by_ids(self, segment_ids):
            return {segment_ids[0]: "Exact legacy passage."}

    response = ChatEngine(LegacyMemory()).ask(
        ChatRequest(question="Show me exact passage doc1:p303c001", as_of=date(2009, 9, 27))
    )

    assert "Exact legacy passage" in response.answer
    assert "métadonnées sont absentes" in response.answer
    assert "UNK" in response.answer


def test_scope_limit_is_consultative_and_runtime_derived():
    response = ChatEngine(FakeMemory()).ask(
        ChatRequest(question="quels étaient tous les partis en 2009 ?", as_of=date(2009, 9, 27))
    )

    assert response.route == "OUT_OF_CORPUS"
    assert "TST" in response.answer
    assert "P100" in response.answer
    assert "façon exhaustive" in response.answer
    assert "DEU" not in response.answer


def test_follow_up_sources_uses_structured_previous_citations():
    previous = [{
        "ref_id": "S1",
        "segment_id": "doc1:p000c000",
        "text": "Stored proof.",
        "doc_id": "doc1",
        "country_iso3": "TST",
        "party_id": "P100",
        "doc_date": "2009-09-01",
        "doc_type": "manifesto",
        "reliability": "official",
    }]

    response = ChatEngine(FakeMemory()).ask(ChatRequest(
        question="quelles sont les sources exactes ?",
        as_of=date(2009, 9, 27),
        previous_citations=previous,
    ))

    assert response.route == "FOLLOW_UP_SOURCES"
    assert "Stored proof" in response.answer
    assert response.citations[0].segment_id == "doc1:p000c000"


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
                "text": f"Evidence passage {i} provides substantive political support " + ("x" * 1200),
                "parent_text": f"Parent context {i} provides broader manifesto meaning " + ("y" * 1200),
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
                "text": f"General context {i} presents a distinct manifesto orientation " + ("z" * 1500),
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
        retrieval=chat_engine.RetrievalBundle(trace=[
            {
                "stage": "query",
                "lane": "primary",
                "query": "long query " + ("q" * 600),
                "retrieved": 24,
            }
            for _ in range(12)
        ]),
    )

    prompt = "\n".join(message["content"] for message in messages)
    target_chars = int(
        (chat_engine.settings.chat_llm_context_window - chat_engine.settings.chat_prompt_reserved_output_tokens)
        * chat_engine.settings.chat_prompt_chars_per_token
    )
    assert len(prompt) <= target_chars
    assert "[S4]" in prompt
    assert "[S5]" not in prompt
    assert "[C1]" in prompt
    assert "[C3]" in prompt
    assert "segment=doc1:p003" not in prompt
