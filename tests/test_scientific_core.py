from datetime import date

import pytest

from compass.aggregation import aggregate, combined_confidence
from compass.diagnostic_engine import DiagnosisEngine
from compass.general_memory import GeneralMemory
from compass.guardrails import TemporalViolation, assert_temporal_integrity
from compass.schemas import (
    CaseKey,
    DocumentMeta,
    EvidenceItem,
    EvidenceRegime,
    JudgeAnswer,
    OutputType,
    Segment,
    SourceReliability,
    SufficiencyVerdict,
    VariableMethod,
    VariableSheet,
)
from compass.sufficiency_gate import SufficiencyGate, extract_features


def _sheet() -> VariableSheet:
    return VariableSheet(
        variable_id="v_test",
        question="Le parti soutient-il la démocratie ?",
        definition="Position déclarée et observée sur la démocratie.",
        scale={0: "non", 4: "oui"},
        method=VariableMethod.LLM_GUIDED,
        output_type=OutputType.ORDINAL,
        evidence_regimes=[EvidenceRegime.DECLARED, EvidenceRegime.OBSERVED],
        required_sources=["manifeste", "presse"],
    )


def _segment(country: str = "TST", doc_date: date = date(2020, 1, 1)) -> Segment:
    meta = DocumentMeta(
        doc_id="doc",
        country_iso3=country,
        party_id=None,
        doc_date=doc_date,
        doc_type="manifeste",
        language="fr",
        reliability=SourceReliability.OFFICIAL,
    )
    return Segment(segment_id="doc:p000c000", doc_id="doc", text="Preuve.", meta=meta)


def _evidence(doc_date: date = date(2020, 1, 1), supports: bool = True) -> EvidenceItem:
    return EvidenceItem(
        segment=_segment(doc_date=doc_date),
        regime=EvidenceRegime.DECLARED,
        supports=supports,
        relevance=0.9,
        qualification_method="test",
    )


def test_aggregation_combines_judges_and_proof_confidence():
    answers = [
        JudgeAnswer(judge_id="j1", model_name="m1", score=2, rationale="r", confidence=0.8),
        JudgeAnswer(judge_id="j2", model_name="m2", score=4, rationale="r", confidence=0.8),
    ]

    result = aggregate(answers, method="mean")

    assert result.score == 3.0
    assert result.disagreement == 1.0
    assert combined_confidence(result.disagreement, 0.8, 4) == 0.6


def test_sufficiency_gate_uses_evidence_regime_coverage_before_training():
    passages = [{
        "text": "preuve",
        "relevance": 0.9,
        "meta": {"doc_type": "manifeste", "reliability": "official"},
    }]
    features = extract_features(passages, _sheet())
    verdict, probability = SufficiencyGate(threshold=0.8).decide(
        passages, _sheet(), searches_done=0,
    )

    assert features.shape == (1, 6)
    assert probability == 0.5
    assert verdict is SufficiencyVerdict.SEARCH_MORE


def test_diagnosis_reports_missing_regime_without_loading_a_model():
    engine = DiagnosisEngine.__new__(DiagnosisEngine)
    engine._nli = lambda pair: {"label": "neutral", "score": 0.9}
    engine._max_pairs = 60
    case = CaseKey(
        country_iso3="TST", party_id="P100", election_id="E1",
        election_date=date(2020, 2, 1),
    )

    diagnosis = engine.diagnose(case, _sheet(), [_evidence()])

    assert len(diagnosis.convergent) == 1
    assert any("observed" in item for item in diagnosis.missing)


def test_temporal_guardrail_rejects_future_evidence():
    with pytest.raises(TemporalViolation):
        assert_temporal_integrity([_evidence(date(2021, 1, 1))], date(2020, 1, 1))


def test_general_memory_accepts_only_general_scope():
    class Collection:
        def __init__(self):
            self.calls = 0

        def upsert(self, **kwargs):
            self.calls += 1

    memory = GeneralMemory.__new__(GeneralMemory)
    memory._col = Collection()

    with pytest.raises(ValueError):
        memory.add([_segment(country="TST")])

    memory.add([_segment(country="GEN")])
    assert memory._col.calls == 1
