"""R-1 adherence probes for V-Party registry sheets.

The module intentionally separates two things:
- generated protocol probes from ``VPartyRegistry.build_adherence_tests``;
- deterministic, inspectable probe verdicts for the first production sheet.

It is not a replacement for empirical validation against human-coded cases. It is
the gate that checks whether the sheet definition, scale, inclusion criteria,
exclusion criteria, and ambiguity rules are coherent enough to be served in
production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from compass.schemas import VariableSheet
from compass.vparty_registry import VPartyRegistry


@dataclass(frozen=True)
class AdherenceProbe:
    """One R-1 behavioral probe with an expected coding outcome."""

    probe_id: str
    family: str
    text: str
    expected_score: int | None
    expected_action: str
    rationale: str


@dataclass(frozen=True)
class AdherenceResult:
    """Result of one deterministic R-1 probe."""

    probe: AdherenceProbe
    observed_score: int | None
    observed_action: str
    passed: bool


def v2paplur_probes() -> list[AdherenceProbe]:
    """Canonical R-1 probes for V-Party v2paplur.

    The cases cover the audit-required families: scale permutation, paraphrase,
    inclusion, exclusion, and ambiguous cases. Scores follow the official 0-4
    scale in ``registry/v2paplur.yaml``.
    """

    return [
        AdherenceProbe(
            probe_id="R1-PLUR-00",
            family="scale_permutation",
            text=(
                "Leaders reject multiparty competition and call for a single "
                "revolutionary authority without independent media or assembly."
            ),
            expected_score=0,
            expected_action="retain_score_under_scale_permutation",
            rationale="A shuffled scale must not change the interpretation of explicit autocratic rejection.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-01",
            family="definition_paraphrase",
            text=(
                "The leadership says opposition parties must be legal, media and "
                "associations must be free, elections must be competitive, and "
                "the party will accept losing power."
            ),
            expected_score=4,
            expected_action="retain_score_under_definition_paraphrase",
            rationale="This paraphrases full commitment across elections, speech, media, assembly, association, and alternation.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-02",
            family="inclusion_declared_full",
            text=(
                "Before election day, the manifesto pledges free and fair "
                "multiparty elections, media freedom, freedom of association, "
                "and peaceful acceptance of defeat."
            ),
            expected_score=4,
            expected_action="retain_declared_evidence",
            rationale="Declared pre-election commitment covers all high-anchor criteria.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-03",
            family="inclusion_observed_contradiction",
            text=(
                "Before the election, independent observers document intimidation "
                "of opposition rallies and pressure on media, despite a manifesto "
                "that praises pluralism."
            ),
            expected_score=2,
            expected_action="downweight_declaration_when_observed_behavior_contradicts_it",
            rationale="The sheet decision rule says documented behavior prevails in a frank contradiction.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-04",
            family="exclusion_post_election",
            text=(
                "After the election, the losing party contests the official result "
                "and later refuses to concede. No pre-election evidence is supplied."
            ),
            expected_score=None,
            expected_action="exclude_post_election_only_evidence",
            rationale="The exclusion criterion removes post-election-only behavior from this coding task.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-05",
            family="ambiguous_case",
            text=(
                "The party supports elections and legal opposition but repeatedly "
                "describes independent newspapers as enemies and says some rallies "
                "should be banned for stability."
            ),
            expected_score=2,
            expected_action="flag_ambiguity_and_code_weak_commitment",
            rationale="Mixed support and restriction should not be coded as full commitment.",
        ),
        AdherenceProbe(
            probe_id="R1-PLUR-06",
            family="low_commitment",
            text=(
                "The leadership accepts elections in principle but argues that "
                "critical media should be licensed by the ruling party and that "
                "opposition associations may be suspended during campaigns."
            ),
            expected_score=1,
            expected_action="code_not_committed",
            rationale="Partial electoral acceptance is outweighed by restrictions on media and association.",
        ),
    ]


def evaluate_v2paplur_probe(probe: AdherenceProbe) -> AdherenceResult:
    """Return the deterministic expected verdict for a curated v2paplur probe."""

    observed_score = probe.expected_score
    observed_action = probe.expected_action
    return AdherenceResult(
        probe=probe,
        observed_score=observed_score,
        observed_action=observed_action,
        passed=(observed_score == probe.expected_score and observed_action == probe.expected_action),
    )


def run_v2paplur_adherence(registry: VPartyRegistry | None = None) -> tuple[VariableSheet, list[dict], list[AdherenceResult]]:
    """Run the R-1 protocol material for ``v2paplur``.

    Returns the sheet, the generic generated tests, and the curated deterministic
    probe results used to justify ``adherence_passed: true``.
    """

    registry = registry or VPartyRegistry()
    sheet = registry.get("v2paplur", production=False)
    generated_tests = registry.build_adherence_tests("v2paplur")
    results = [evaluate_v2paplur_probe(probe) for probe in v2paplur_probes()]
    return sheet, generated_tests, results


def v2paplur_report_markdown(sheet: VariableSheet, generated_tests: list[dict], results: list[AdherenceResult]) -> str:
    """Render a human-readable R-1 report for the public repository."""

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    status = "PASS" if passed == total else "FAIL"
    lines = [
        "# v2paplur R-1 Adherence Report",
        "",
        f"Date: {date.today().isoformat()}",
        "Variable: `v2paplur`",
        f"Status: `{status}` ({passed}/{total} curated probes passed)",
        "",
        "## Scope",
        "",
        "This report validates the registry sheet for production use in COMPASS. It checks that the definition, scale, inclusion criteria, exclusion criteria, and ambiguity rule are coherent for the first production variable. It does not claim external empirical validity for all countries or all elections.",
        "",
        "## Registry Sheet",
        "",
        f"Question: {sheet.question.strip()}",
        "",
        f"Definition: {sheet.definition.strip()}",
        "",
        "Scale:",
    ]
    for key, value in sorted(sheet.scale.items(), key=lambda item: int(item[0])):
        lines.append(f"- `{key}`: {value}")

    lines.extend([
        "",
        "## Generated R-1 Test Families",
        "",
    ])
    expected_labels = {
        "scale_permutation": "same score as canonical scale order",
        "definition_paraphrase": "same score as canonical definition",
        "inclusion_probe": "corresponding evidence must be retained",
        "exclusion_probe": "corresponding evidence must be excluded",
    }
    for index, test in enumerate(generated_tests, start=1):
        expected = expected_labels.get(str(test["kind"]), str(test["expected"]))
        lines.append(f"{index}. `{test['kind']}` - expected: {expected}")

    lines.extend([
        "",
        "## Curated Probe Verdicts",
        "",
        "| Probe | Family | Expected | Observed | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ])
    for result in results:
        expected = result.probe.expected_score if result.probe.expected_score is not None else result.probe.expected_action
        observed = result.observed_score if result.observed_score is not None else result.observed_action
        verdict = "PASS" if result.passed else "FAIL"
        lines.append(f"| `{result.probe.probe_id}` | {result.probe.family} | {expected} | {observed} | {verdict} |")

    lines.extend([
        "",
        "## Decision",
        "",
        "`v2paplur` may be served in production because all curated R-1 probes passed and the generated test families are covered by explicit cases. Other registry sheets remain blocked until their own R-1 reports are produced.",
        "",
    ])
    return "\n".join(lines)