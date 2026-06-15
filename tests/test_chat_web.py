from datetime import date

from apps.chat_web import answer_question, is_greeting


class DummyEngine:
    def ask(self, request):
        class Response:
            answer = "Evidence-based answer [S1]."
            citations = []
        return Response()


def test_chat_web_greeting():
    assert is_greeting("salut") is True
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