from datetime import date

from compass.chat.expert_retrieval import retrieve_expert
from compass.chat.query_analysis import QueryAnalysis


def _record(
    segment_id: str,
    text: str,
    *,
    parent: str,
    country: str = "DEU",
    party: str = "41320",
    election: str = "DEU_2009",
    doc_date: str = "2009-09-01",
):
    return {
        "segment_id": segment_id,
        "text": text,
        "parent_text": f"Parent context for {text}",
        "rerank_score": 0.9,
        "meta": {
            "doc_id": "doc1",
            "country_iso3": country,
            "party_id": party,
            "election_id": election,
            "doc_date": doc_date,
            "temporal_ok": 1,
            "parent_segment_id": parent,
        },
    }


def _analysis():
    return QueryAnalysis(
        actors=["SPD"],
        themes=["democracy"],
        period="2009",
        answer_type="position",
        language="en",
        subqueries=["SPD democracy", "democratic commitments"],
        method="llm",
    )


def test_expert_retrieval_runs_three_lanes_and_keeps_roles(monkeypatch):
    from compass.chat import expert_retrieval as er

    monkeypatch.setattr(er.settings, "chat_strict_election_scope", True)

    def query(memory, question, **kwargs):
        if "conditions limits" in question:
            return [_record("nuance", "Support applies under constitutional limits.", parent="p2")]
        if "opposition rejection" in question:
            return [_record("counter", "The proposal rejects direct plebiscites.", parent="p3")]
        return [
            _record("main", "The party supports parliamentary democracy.", parent="p1"),
            _record("duplicate", "The party supports parliamentary democracy.", parent="p1"),
        ]

    bundle = retrieve_expert(
        object(),
        _analysis(),
        query_function=query,
        as_of=date(2009, 9, 27),
        country_iso3="DEU",
        party_id="41320",
        election_id="DEU_2009",
        include_unverified=False,
        k=6,
    )
    prompt = bundle.prompt_records(4)

    assert bundle.primary[0]["evidence_role"] == "primary"
    assert bundle.nuances[0]["evidence_role"] == "nuance"
    assert bundle.counter[0]["evidence_role"] == "counter"
    assert {item["evidence_role"] for item in prompt} == {"primary", "nuance", "counter"}
    assert any(step["stage"] == "lane_fusion" for step in bundle.trace)
    assert bundle.sufficiency >= 0.35


def test_expert_retrieval_strictly_filters_active_scope(monkeypatch):
    from compass.chat import expert_retrieval as er

    monkeypatch.setattr(er.settings, "chat_strict_election_scope", True)

    def query(memory, question, **kwargs):
        return [
            _record("valid", "Valid evidence.", parent="p1"),
            _record("country", "Wrong country.", parent="p2", country="FRA"),
            _record("party", "Wrong party.", parent="p3", party="999"),
            _record("election", "Wrong election.", parent="p4", election="DEU_2013"),
            _record("future", "Future text.", parent="p5", doc_date="2013-01-01"),
        ]

    bundle = retrieve_expert(
        object(),
        _analysis(),
        query_function=query,
        as_of=date(2009, 9, 27),
        country_iso3="DEU",
        party_id="41320",
        election_id="DEU_2009",
        include_unverified=False,
        k=6,
    )

    assert [item["segment_id"] for item in bundle.primary] == ["valid"]
    assert any(step.get("scope_rejections", 0) > 0 for step in bundle.trace)


def test_expert_retrieval_diversifies_parent_sections(monkeypatch):
    from compass.chat import expert_retrieval as er

    monkeypatch.setattr(er.settings, "chat_strict_election_scope", False)

    def query(memory, question, **kwargs):
        if "conditions limits" in question or "opposition rejection" in question:
            return []
        return [
            _record("a", "Democracy and citizen participation.", parent="p1"),
            _record("b", "Democracy with public participation.", parent="p1"),
            _record("c", "Parliament controls the executive.", parent="p2"),
        ]

    bundle = retrieve_expert(
        object(), _analysis(), query_function=query,
        as_of=date(2009, 9, 27), country_iso3="DEU", party_id="41320",
        election_id=None, include_unverified=False, k=3,
    )

    assert [item["segment_id"] for item in bundle.primary[:2]] == ["a", "c"]

