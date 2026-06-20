from compass.chat.query_analysis import (
    analyze_question,
    deterministic_question_analysis,
    format_query_analysis,
)


SCOPE = {
    "parties": [{"party_id": "41320", "name": "SPD"}],
    "document_dates": ["2009-09-01"],
}


def test_llm_analysis_returns_strict_structured_plan(monkeypatch):
    from compass.chat import query_analysis as qa

    monkeypatch.setattr(qa.settings, "chat_query_analysis_enabled", True)

    def complete(model, messages, **kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "Do not answer" in messages[0]["content"]
        return (
            '{"actors":["SPD"],"themes":["democracy","participation"],'
            '"period":"2009","answer_type":"evidence","language":"en",'
            '"subqueries":["SPD democracy participation",'
            '"SPD democratic commitments evidence"]}'
        )

    analysis = analyze_question(
        "What evidence describes the SPD position on democracy in 2009?",
        scope=SCOPE,
        model_name="local-model",
        complete=complete,
    )

    assert analysis.method == "llm"
    assert analysis.actors == ["SPD"]
    assert analysis.period == "2009"
    assert analysis.answer_type == "evidence"
    assert analysis.subqueries[0].startswith("What evidence")
    assert len(analysis.subqueries) == 3


def test_invalid_or_decorated_json_uses_deterministic_fallback(monkeypatch):
    from compass.chat import query_analysis as qa

    monkeypatch.setattr(qa.settings, "chat_query_analysis_enabled", True)
    analysis = analyze_question(
        "Que dit ce parti sur la démocratie en 2009 ?",
        scope=SCOPE,
        model_name="local-model",
        complete=lambda *args, **kwargs: "```json\n{}\n```",
    )

    assert analysis.method == "deterministic"
    assert analysis.language == "fr"
    assert analysis.period == "2009"
    assert "democratie" in analysis.themes


def test_llm_cannot_inject_an_actor_outside_question_or_scope(monkeypatch):
    from compass.chat import query_analysis as qa

    monkeypatch.setattr(qa.settings, "chat_query_analysis_enabled", True)
    raw = (
        '{"actors":["Imaginary Party"],"themes":["democracy"],"period":null,'
        '"answer_type":"position","language":"en",'
        '"subqueries":["democracy position","democracy evidence"]}'
    )
    analysis = analyze_question(
        "What does this party say about democracy?",
        scope=SCOPE,
        model_name="local-model",
        complete=lambda *args, **kwargs: raw,
    )

    assert analysis.actors == ["SPD"]
    assert "Imaginary Party" not in analysis.actors


def test_llm_cannot_inject_unrelated_query_or_period(monkeypatch):
    from compass.chat import query_analysis as qa

    monkeypatch.setattr(qa.settings, "chat_query_analysis_enabled", True)
    raw = (
        '{"actors":[],"themes":["democracy"],"period":"2017",'
        '"answer_type":"position","language":"en",'
        '"subqueries":["unrelated military conflict",'
        '"democracy institutional commitments"]}'
    )
    analysis = analyze_question(
        "What does this party say about democracy?",
        scope=SCOPE,
        model_name="local-model",
        complete=lambda *args, **kwargs: raw,
    )

    assert analysis.period is None
    assert "unrelated military conflict" not in analysis.subqueries
    assert "democracy institutional commitments" in analysis.subqueries


def test_deterministic_analysis_is_domain_agnostic():
    analysis = deterministic_question_analysis(
        "Quelles priorités concernent la réforme constitutionnelle ?",
        scope=SCOPE,
    )
    combined = " ".join(analysis.subqueries).lower()

    assert analysis.method == "deterministic"
    assert analysis.answer_type == "list"
    assert "reforme" in analysis.themes
    assert "constitutionnelle" in analysis.themes
    assert "employment" not in combined
    assert "taxation" not in combined


def test_analysis_can_be_disabled_without_calling_llm(monkeypatch):
    from compass.chat import query_analysis as qa

    monkeypatch.setattr(qa.settings, "chat_query_analysis_enabled", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("LLM should not be called")

    analysis = analyze_question(
        "What does this party say about democracy?",
        scope=SCOPE,
        model_name="local-model",
        complete=forbidden,
    )

    assert analysis.method == "deterministic"


def test_query_analysis_has_human_readable_prompt_format():
    analysis = deterministic_question_analysis(
        "Que dit ce parti sur la démocratie en 2009 ?",
        scope=SCOPE,
    )
    rendered = format_query_analysis(analysis)

    assert "actors: SPD" in rendered
    assert "themes:" in rendered
    assert "period: 2009" in rendered
    assert "retrieval_subqueries:" in rendered
