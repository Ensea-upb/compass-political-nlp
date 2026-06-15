from __future__ import annotations

from compass.adherence import run_v2paplur_adherence, v2paplur_report_markdown
from compass.vparty_registry import VPartyRegistry


def test_v2paplur_r1_generated_tests_cover_required_families() -> None:
    registry = VPartyRegistry()
    tests = registry.build_adherence_tests("v2paplur")
    kinds = {test["kind"] for test in tests}
    assert "scale_permutation" in kinds
    assert "definition_paraphrase" in kinds
    assert "inclusion_probe" in kinds
    assert "exclusion_probe" in kinds


def test_v2paplur_curated_r1_probes_all_pass() -> None:
    _, _, results = run_v2paplur_adherence()
    assert results
    assert all(result.passed for result in results)
    families = {result.probe.family for result in results}
    assert "scale_permutation" in families
    assert "definition_paraphrase" in families
    assert "inclusion_declared_full" in families
    assert "inclusion_observed_contradiction" in families
    assert "exclusion_post_election" in families
    assert "ambiguous_case" in families


def test_v2paplur_report_records_pass_decision() -> None:
    sheet, generated_tests, results = run_v2paplur_adherence()
    report = v2paplur_report_markdown(sheet, generated_tests, results)
    assert "Status: `PASS`" in report
    assert "Other registry sheets remain blocked" in report

def test_v2paplur_is_unblocked_but_other_sheets_remain_blocked() -> None:
    from compass.vparty_registry import AdherenceError

    registry = VPartyRegistry()
    assert registry.get("v2paplur", production=True).adherence_passed is True

    blocked = [var_id for var_id in registry.list_ids() if var_id != "v2paplur"]
    assert blocked
    for var_id in blocked:
        try:
            registry.get(var_id, production=True)
        except AdherenceError:
            continue
        raise AssertionError(f"{var_id} should remain blocked until its own R-1 report passes")
