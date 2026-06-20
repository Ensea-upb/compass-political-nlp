from datetime import date
from threading import Event
from time import sleep

from apps.chat_web import (
    ChatJobQueue,
    answer_question,
    answer_question_payload,
    format_scope_banner,
    is_greeting,
    render_prompt_page,
)
from compass.chat.engine import Citation


class DummyEngine:
    def ask(self, request):
        class Response:
            answer = "Evidence-based answer [S1]."
            citations = [Citation(
                ref_id="S1", segment_id="doc:p001c001", text="Proof text.",
                doc_id="doc", country_iso3="TST", party_id="P100",
                doc_date="2020-01-01", doc_type="manifesto", reliability="official",
            )]
            prompt_messages = [{"role": "user", "content": "prompt"}]
            route = "evidence_query"
            retrieval_count = 8
            prompt_citation_count = 1
        return Response()


def test_chat_web_greeting():
    assert is_greeting("salut") is True
    assert is_greeting("salut, c'est comment") is True
    assert is_greeting("What about democracy?") is False


def test_chat_web_answer_question_greeting():
    answer = answer_question(
        engine=DummyEngine(),
        question="bonjour",
        history=[],
        cutoff=date(2009, 9, 27),
        party_id="41320",
        k=8,
    )

    assert "Bonjour" in answer


def test_chat_web_answer_question_calls_engine():
    answer = answer_question(
        engine=DummyEngine(),
        question="What about democracy?",
        history=[],
        cutoff=date(2009, 9, 27),
        party_id="41320",
        k=8,
    )

    assert "Evidence-based answer" in answer
    assert "\n\nSources\n" not in answer


def test_chat_web_payload_adds_prompt_link():
    store = {}

    payload = answer_question_payload(
        engine=DummyEngine(),
        question="What about democracy?",
        history=[],
        cutoff=date(2009, 9, 27),
        party_id="41320",
        k=8,
        prompt_store=store,
    )

    assert payload["prompt_url"].startswith("./prompt/")
    assert store
    assert "\n\nSources\n" not in payload["answer"]
    assert payload["route"] == "evidence_query"
    assert payload["retrieval_count"] == 8
    assert payload["prompt_citation_count"] == 1
    assert payload["sources"][0]["segment_id"] == "doc:p001c001"
    assert "Proof text" in payload["sources_markdown"]


def test_chat_web_forwards_selected_routing_mode():
    class RoutingEngine:
        def ask(self, request):
            assert request.routing_mode == "llm"
            assert request.election_id == "E1"
            return type("Response", (), {
                "answer": "Routed answer.",
                "citations": [],
                "prompt_messages": [],
                "route": "evidence_query",
                "retrieval_count": 0,
                "prompt_citation_count": 0,
            })()

    payload = answer_question_payload(
        engine=RoutingEngine(),
        question="What does the party say?",
        history=[],
        cutoff=date(2009, 9, 27),
        party_id="41320",
        election_id="E1",
        k=8,
        routing_mode="llm",
    )

    assert payload["answer"] == "Routed answer."


def test_chat_web_forwards_structured_previous_sources():
    class FollowUpEngine:
        def ask(self, request):
            assert request.previous_citations[0]["segment_id"] == "doc:p001c001"
            return type("Response", (), {
                "answer": "Stored proof.", "citations": [], "prompt_messages": [],
                "route": "FOLLOW_UP_SOURCES", "retrieval_count": 0,
                "prompt_citation_count": 0,
            })()

    payload = answer_question_payload(
        engine=FollowUpEngine(), question="What are your exact sources?", history=[],
        cutoff=date(2009, 9, 27), party_id=None, k=8,
        previous_citations=[{"segment_id": "doc:p001c001"}],
    )

    assert payload["route"] == "FOLLOW_UP_SOURCES"


def test_chat_web_prompt_page_is_human_readable():
    page = render_prompt_page([
        {"role": "system", "content": "Do not use outside knowledge."},
        {"role": "user", "content": "ANALYTICAL_CONTEXT\nframe\n\nGENERAL_CONTEXT\ncontext\n\nRELATIONAL_CONTEXT\n[R1] inferred\n\nCITED_EVIDENCE\n[S1] proof\n\nAnswer contract"},
    ])

    assert "Prompt envoye au LLM" in page
    assert "<mark>ANALYTICAL_CONTEXT</mark>" in page
    assert "<mark>GENERAL_CONTEXT</mark>" in page
    assert "<mark>RELATIONAL_CONTEXT</mark>" in page
    assert "<mark>CITED_EVIDENCE</mark>" in page
    assert "Voir le JSON exact envoye" in page

def test_chat_web_uses_relative_ask_endpoint():
    from apps.chat_web import HTML

    assert "fetch('./ask'" in HTML
    assert "fetch('/ask'" not in HTML
    assert "Non-JSON response from server" in HTML
    assert "fetch('./result/'" in HTML
    assert "prompt_url" in HTML
    assert "Voir le prompt LLM" in HTML
    assert "compass_prompt_viewer" in HTML
    assert "window.open(promptUrl, 'compass_prompt_viewer')" in HTML
    assert 'name="routing_mode"' in HTML
    assert 'value="deterministic" checked' in HTML
    assert 'value="llm"' in HTML
    assert "__ROUTING_CLASS__" in HTML
    assert "last_sources" in HTML
    assert "sources_markdown" in HTML
    assert "data-question" in HTML


def test_chat_job_queue_returns_long_running_result_without_blocking_submit():
    queue = ChatJobQueue()
    release = Event()
    try:
        job_id = queue.submit(lambda: (release.wait(2), {"answer": "done"})[1])
        assert queue.get(job_id)["status"] in {"pending", "running"}
        release.set()
        for _ in range(100):
            job = queue.get(job_id)
            if job["status"] == "completed":
                break
            sleep(0.01)
        assert job["result"] == {"answer": "done"}
    finally:
        queue.close()


def test_scope_banner_is_runtime_derived():
    banner = format_scope_banner({
        "country_iso3": "TST",
        "n_documents": 3,
        "parties": [{"party_id": "P100", "name": "Test Party"}],
        "document_types": ["manifesto"],
    }, date(2020, 1, 2))

    assert "TST" in banner
    assert "P100" in banner
    assert "documents=3" in banner
    assert "as_of=2020-01-02" in banner
