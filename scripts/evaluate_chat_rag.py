"""Run the COMPASS Chat reference questions and compare JSON snapshots."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_SOURCE_RE = re.compile(r"\[S\d+\]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True)
    parser.add_argument("--party", required=True)
    parser.add_argument("--election-id", required=True)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "evaluation" / "rag_reference_questions.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    from compass.chat import ChatEngine, ChatRequest
    from compass.country_memory import CountryMemory

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    memory = CountryMemory(args.country)
    engine = ChatEngine(memory)
    results = []
    for item in questions:
        started = time.perf_counter()
        response = engine.ask(ChatRequest(
            question=item["question"],
            as_of=args.as_of,
            party_id=args.party,
            election_id=args.election_id,
            k=args.k,
        ))
        evidence_text = " ".join(citation.text for citation in response.citations).casefold()
        terms = [str(term).casefold() for term in item.get("expected_evidence_terms", [])]
        matched_terms = [term for term in terms if term in evidence_text]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": response.answer,
            "route": response.route,
            "llm_used": response.llm_used,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "retrieval_count": response.retrieval_count,
            "prompt_citation_count": response.prompt_citation_count,
            "inline_citations": len(_SOURCE_RE.findall(response.answer)),
            "evidence_roles": sorted({citation.evidence_role for citation in response.citations}),
            "term_recall": len(matched_terms) / len(terms) if terms else None,
            "matched_terms": matched_terms,
            "query_analysis": response.query_analysis,
            "retrieval_trace": response.retrieval_trace,
            "validation_trace": response.validation_trace,
            "sources": [citation.segment_id for citation in response.citations],
        })

    report = {
        "scope": {
            "country": args.country,
            "party": args.party,
            "election_id": args.election_id,
            "as_of": args.as_of.isoformat(),
        },
        "summary": _summary(results),
        "results": results,
    }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["comparison"] = _compare(baseline.get("summary", {}), report["summary"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report.get("comparison"):
        print("Comparison:")
        print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"Report written: {args.output.resolve()}")


def _summary(results: list[dict]) -> dict:
    recalls = [item["term_recall"] for item in results if item["term_recall"] is not None]
    return {
        "questions": len(results),
        "mean_term_recall": round(mean(recalls), 4) if recalls else None,
        "mean_latency_seconds": round(mean(item["latency_seconds"] for item in results), 4),
        "llm_success_rate": round(mean(float(item["llm_used"]) for item in results), 4),
        "mean_inline_citations": round(mean(item["inline_citations"] for item in results), 4),
        "responses_with_validation_rejection": sum(
            any(step.get("status") == "rejected" for step in item["validation_trace"])
            for item in results
        ),
    }


def _compare(before: dict, after: dict) -> dict:
    comparison = {}
    for key, value in after.items():
        previous = before.get(key)
        if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
            comparison[key] = {
                "before": previous,
                "after": value,
                "delta": round(value - previous, 4),
            }
    return comparison


if __name__ == "__main__":
    main()
