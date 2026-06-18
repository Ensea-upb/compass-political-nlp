from datetime import date

from apps.chat_web import answer_question, is_greeting, is_source_followup, latest_sources_from_history


class DummyEngine:
    def ask(self, request):
        class Response:
            answer = "Evidence-based answer [S1]."
            citations = []
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
    assert "Sources" in answer


def test_chat_web_reuses_last_sources_for_followup():
    history = [
        {"role": "user", "content": "What about democracy?"},
        {"role": "assistant", "content": "Answer [S1].\n\nSources\n- [S1] source detail"},
    ]

    answer = answer_question(
        engine=DummyEngine(),
        question="What are the exact sources for your answer?",
        history=history,
        cutoff=date(2009, 9, 27),
        party_id="41320",
        k=8,
    )

    assert "source detail" in answer
    assert "Evidence-based answer" not in answer


def test_chat_web_source_followup_helpers():
    assert is_source_followup("What are the exact sources?") is True
    assert latest_sources_from_history([
        {"role": "assistant", "content": "A\n\nSources\n- [S1] detail"}
    ]) == "- [S1] detail"

def test_chat_web_uses_relative_ask_endpoint():
    from apps.chat_web import HTML

    assert "fetch('./ask'" in HTML
    assert "fetch('/ask'" not in HTML
    assert "Non-JSON response from server" in HTML
