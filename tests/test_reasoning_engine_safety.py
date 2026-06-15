import math
from datetime import date

import pytest

from compass.reasoning_engine import parse_scale_score, safe_eval_formula
from compass.schemas import CaseKey, Diagnosis, VariableMethod, VariableSheet


def test_safe_eval_formula_valid_v2xpa_popul_shape():
    env = {"v2papeople": 2, "v2paanteli": 4}

    assert safe_eval_formula("(v2papeople + v2paanteli) / 2", env) == 3.0


def test_safe_eval_formula_missing_variable():
    with pytest.raises(NameError):
        safe_eval_formula("missing + 1", {})


def test_safe_eval_formula_zero_division():
    with pytest.raises(ZeroDivisionError):
        safe_eval_formula("1 / denominator", {"denominator": 0})


def test_safe_eval_formula_rejects_function_call():
    with pytest.raises(ValueError):
        safe_eval_formula("abs(x)", {"x": -1})


def test_safe_eval_formula_rejects_attribute_access():
    with pytest.raises(ValueError):
        safe_eval_formula("x.real", {"x": 1})


def test_parse_scale_score_numeric_prefix():
    assert parse_scale_score("3: committed") == 3.0


def test_parse_scale_score_non_numeric_label():
    with pytest.raises(ValueError):
        parse_scale_score("interval: 0 to 1")


class FakeCountry:
    def __init__(self, rows):
        self.rows = rows

    def query_structured(self, sql, params=()):
        import pandas as pd

        return pd.DataFrame(self.rows)


def _sheet(rule):
    return VariableSheet(
        variable_id="v2xpa_popul",
        question="Derived populism index",
        definition="Test formula",
        scale={0: "low", 1: "high"},
        method=VariableMethod.DETERMINISTIC_RULE,
        evidence_regimes=[],
        required_sources=[],
        decision_rules=[rule],
        adherence_passed=True,
    )


def _diagnosis():
    return Diagnosis(
        case=CaseKey(country_iso3="DEU", party_id="41320", election_id="DEU_2009", election_date=date(2009, 9, 27)),
        variable_id="v2xpa_popul",
    )


def test_deterministic_formula_division_by_zero_returns_controlled_answer():
    from compass.reasoning_engine import ReasoningEngine

    engine = object.__new__(ReasoningEngine)
    engine._country = FakeCountry([{"variable_id": "denominator", "score": 0}])

    answer = engine._deterministic(_sheet("1 / denominator"), _diagnosis(), "model", "standard")

    assert math.isnan(answer.score)
    assert answer.confidence == 0.0