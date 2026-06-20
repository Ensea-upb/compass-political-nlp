from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from compass.chat.scientific_service import (
    ScientificChatService,
    ScientificConfigurationError,
)


class FakeRegistry:
    def list_ids(self):
        return ["v_test"]

    def get(self, variable_id, production=True):
        if variable_id != "v_test":
            raise KeyError(variable_id)
        return SimpleNamespace(variable_id=variable_id, adherence_passed=True)


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run_case(self, case, variable_ids):
        self.calls.append((case, variable_ids))
        return [SimpleNamespace(variable_id=variable_ids[0])]


class FakeMemory:
    def describe_corpus(self):
        return {"parties": [{"party_id": "P100"}]}

    def query_structured(self, sql, params):
        return pd.DataFrame([{"election_id": "E2020"}])


def test_scientific_service_builds_case_from_active_memory():
    runner = FakeRunner()
    service = ScientificChatService(
        FakeMemory(), runner=runner, registry=FakeRegistry(),
    )

    result = service.analyze(
        "v_test",
        country_iso3="tst",
        party_id=None,
        election_id=None,
        as_of=date(2020, 1, 2),
    )

    case, variable_ids = runner.calls[0]
    assert result.variable_id == "v_test"
    assert case.country_iso3 == "TST"
    assert case.party_id == "P100"
    assert case.election_id == "E2020"
    assert variable_ids == ["v_test"]


def test_scientific_service_rejects_unknown_variable_before_running():
    runner = FakeRunner()
    service = ScientificChatService(
        FakeMemory(), runner=runner, registry=FakeRegistry(),
    )

    with pytest.raises(ScientificConfigurationError, match="Variable inconnue"):
        service.analyze(
            "unknown",
            country_iso3="TST",
            party_id="P100",
            election_id="E2020",
            as_of=date(2020, 1, 2),
        )

    assert runner.calls == []


def test_scientific_validation_is_separate_and_uses_cached_outputs(monkeypatch):
    runner = FakeRunner()
    service = ScientificChatService(
        FakeMemory(), runner=runner, registry=FakeRegistry(),
    )
    service.analyze(
        "v_test",
        country_iso3="TST",
        party_id="P100",
        election_id="E2020",
        as_of=date(2020, 1, 2),
    )
    expected = SimpleNamespace(stratum="chat_session")

    class Vault:
        def truth(self):
            return pd.DataFrame([{"party_id": "P100"}])

    class Validator:
        def __init__(self, truth):
            assert not truth.empty

        def evaluate(self, answers, stratum):
            assert len(answers) == 1
            assert stratum == "chat_session"
            return expected

    monkeypatch.setattr("compass.validation.EvaluationVault", Vault)
    monkeypatch.setattr("compass.validation.Validator", Validator)

    assert service.validate_cached("v_test") is expected


def test_contamination_probe_is_explicit_and_separate(monkeypatch):
    service = ScientificChatService(
        FakeMemory(), runner=FakeRunner(), registry=FakeRegistry(),
    )
    monkeypatch.setattr("compass.config.settings.judge_models", ["judge-a"])
    monkeypatch.setattr(
        "compass.guardrails.contamination_probe",
        lambda model, party, year, variable: {
            "model": model,
            "claims_knowledge": False,
            "raw": "inconnu",
            "party": party,
            "year": year,
            "variable": variable,
        },
    )

    results = service.contamination_check(
        "v_test", party_id="P100", election_year=2020,
    )

    assert results[0]["model"] == "judge-a"
    assert results[0]["claims_knowledge"] is False
