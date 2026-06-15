from compass.internal_retrieval import InternalRetriever
from compass.schemas import EvidenceRegime, VariableMethod, VariableSheet


def test_hyde_prompt_is_scale_neutral(monkeypatch):
    captured = {}

    def fake_complete(model_name, messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return "hypothetical neutral passage"

    monkeypatch.setattr("compass.internal_retrieval.complete_chat", fake_complete)
    retriever = object.__new__(InternalRetriever)
    sheet = VariableSheet(
        variable_id="v2paplur",
        question="Commitment to pluralism?",
        definition="Pluralism definition.",
        scale={0: "none", 4: "full"},
        method=VariableMethod.LLM_GUIDED,
        evidence_regimes=[EvidenceRegime.DECLARED],
        required_sources=["manifesto"],
        adherence_passed=True,
    )

    assert retriever._generate_hyde_doc(sheet) == "hypothetical neutral passage"
    prompt = captured["prompt"]
    assert "faible, moyen ou eleve" in prompt
    assert "score " + "eleve" not in prompt
