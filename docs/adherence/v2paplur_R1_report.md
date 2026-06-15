# v2paplur R-1 Adherence Report

Date: 2026-06-15
Variable: `v2paplur`
Status: `PASS` (7/7 curated probes passed)

## Scope

This report validates the registry sheet for production use in COMPASS. It checks that the definition, scale, inclusion criteria, exclusion criteria, and ambiguity rule are coherent for the first production variable. It does not claim external empirical validity for all countries or all elections.

## Registry Sheet

Question: Prior to this election, to what extent was the leadership of this political party clearly committed to free and fair elections with multiple parties, freedom of speech, media, assembly and association?

Definition: No commitment: leaders openly support an autocratic form of government without elections or freedom of speech, assembly and association (e.g. theocracy, single-party rule, revolutionary regime). Full commitment: leaders unambiguously support freedom of speech, media, assembly and association and pledge to accept defeat in free and fair elections.

Scale:
- `0`: Not at all committed.
- `1`: Not committed.
- `2`: Weakly committed.
- `3`: Committed.
- `4`: Fully committed.

## Generated R-1 Test Families

1. `scale_permutation` - expected: same score as canonical scale order
2. `definition_paraphrase` - expected: same score as canonical definition
3. `inclusion_probe` - expected: corresponding evidence must be retained
4. `inclusion_probe` - expected: corresponding evidence must be retained
5. `exclusion_probe` - expected: corresponding evidence must be excluded

## Curated Probe Verdicts

| Probe | Family | Expected | Observed | Verdict |
| --- | --- | --- | --- | --- |
| `R1-PLUR-00` | scale_permutation | 0 | 0 | PASS |
| `R1-PLUR-01` | definition_paraphrase | 4 | 4 | PASS |
| `R1-PLUR-02` | inclusion_declared_full | 4 | 4 | PASS |
| `R1-PLUR-03` | inclusion_observed_contradiction | 2 | 2 | PASS |
| `R1-PLUR-04` | exclusion_post_election | exclude_post_election_only_evidence | exclude_post_election_only_evidence | PASS |
| `R1-PLUR-05` | ambiguous_case | 2 | 2 | PASS |
| `R1-PLUR-06` | low_commitment | 1 | 1 | PASS |

## Decision

`v2paplur` may be served in production because all curated R-1 probes passed and the generated test families are covered by explicit cases. Other registry sheets remain blocked until their own R-1 reports are produced.
