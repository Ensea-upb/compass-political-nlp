"""Run a public example aligned with the real COMPASS architecture.

Two modes are provided:

``smoke`` validates the architecture contracts without downloading heavy models.
``full`` seeds a tiny synthetic case and calls the real CompassRunner. It needs
the full requirements, ChromaDB, sentence-transformers, transformers/torch and
internet access the first time models are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_smoke() -> None:
    from compass.aggregation import aggregate
    from compass.guardrails import assert_temporal_integrity
    from compass.schemas import (
        AggregatedJudgment,
        CaseKey,
        Diagnosis,
        DocumentMeta,
        EvidenceItem,
        EvidenceRegime,
        JudgeAnswer,
        Segment,
        SourceReliability,
        VariableMethod,
        VariableSheet,
    )
    from compass.vparty_registry import VPartyRegistry

    print("COMPASS real-architecture smoke test")
    print("Step 1/5: shared contracts loaded from compass.schemas")

    case = CaseKey(
        country_iso3="CIV",
        party_id="DEMO",
        election_id="CIV_2020_LEG",
        election_date=date(2020, 10, 31),
    )
    meta = DocumentMeta(
        doc_id="demo_manifesto",
        country_iso3="CIV",
        party_id="DEMO",
        doc_date=date(2020, 9, 1),
        publication_date=date(2020, 9, 1),
        doc_type="manifeste",
        language="en",
        election_id="CIV_2020_LEG",
        reliability=SourceReliability.OFFICIAL,
    )
    segment = Segment(
        segment_id="demo_manifesto:0001",
        doc_id=meta.doc_id,
        text="The party defends transparent elections, democratic rights and institutional accountability.",
        meta=meta,
    )
    evidence = EvidenceItem(
        segment=segment,
        regime=EvidenceRegime.DECLARED,
        supports=True,
        relevance=0.92,
        qualification_method="public_smoke_fixture",
    )
    assert_temporal_integrity([evidence], case.election_date)
    print("Step 2/5: C15 temporal guardrail passed")

    registry = VPartyRegistry(ROOT / "registry")
    print(f"Step 3/5: C05 registry loaded ({len(registry.list_ids())} variable sheets)")

    sheet = VariableSheet(
        variable_id="demo_democracy",
        question="Does the party support democratic institutions?",
        definition="Synthetic smoke-test variable for the public architecture example.",
        scale={0: "no support", 1: "support"},
        method=VariableMethod.STRUCTURED_QUERY,
        evidence_regimes=[EvidenceRegime.DECLARED],
        required_sources=["synthetic manifesto"],
        adherence_passed=True,
    )
    diagnosis = Diagnosis(
        case=case,
        variable_id=sheet.variable_id,
        convergent=[evidence],
        dominant_language="en",
    )
    print("Step 4/5: C09 diagnosis object assembled with typed evidence")

    judges = [
        JudgeAnswer(
            judge_id="structured::demo_democracy",
            model_name="sqlite",
            score=1.0,
            rationale="Synthetic structured answer for smoke test.",
            confidence=1.0,
        )
    ]
    judgment: AggregatedJudgment = aggregate(judges, method="median")
    print(
        "Step 5/5: C12 aggregation completed "
        f"(score={judgment.score}, disagreement={judgment.disagreement})"
    )
    print("Smoke status: passed")

    optional = [
        "chromadb",
        "sentence_transformers",
        "transformers",
        "torch",
        "fitz",
        "trafilatura",
        "htmldate",
        "lingua",
        "litellm",
        "networkx",
        "spacy",
    ]
    missing = [name for name in optional if importlib.util.find_spec(name) is None]
    if missing:
        print("Full-mode dependencies not yet available: " + ", ".join(missing))
    else:
        print("Full-mode dependency import check: passed")


def run_full(reset: bool, variables: list[str] | None = None) -> None:
    _configure_full_environment(reset=reset)

    from compass.config import settings
    from compass.country_memory import CountryMemory
    from compass.document_pipeline import DocumentPipeline, make_meta
    from compass.general_memory import GeneralMemory
    from compass.active_search import ActiveSearchEngine
    from compass.orchestrator import CompassRunner
    from compass.political_graph import PoliticalGraph
    from compass.schemas import CaseKey, SourceReliability
    from compass.vparty_registry import VPartyRegistry

    settings.ensure_dirs()
    demo_dir = settings.data_dir / "onyxia_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        demo_dir / "elections.csv",
        ["election_id", "country_iso3", "election_date", "election_type"],
        [["CIV_2020_LEG", "CIV", "2020-10-31", "legislative"]],
    )
    _write_csv(
        demo_dir / "parties.csv",
        ["party_id", "country_iso3", "name", "pf_id", "founded", "dissolved"],
        [["DEMO", "CIV", "Demo Reform Party", "", "2010-01-01", ""]],
    )
    _write_csv(
        demo_dir / "results.csv",
        ["election_id", "party_id", "vote_share", "seats", "seats_total"],
        [["CIV_2020_LEG", "DEMO", "42.5", "84", "255"]],
    )
    _write_csv(
        demo_dir / "events.csv",
        ["event_id", "country_iso3", "event_date", "party_id", "event_type", "description"],
        [["EVT_DEMO_001", "CIV", "2020-08-15", "DEMO", "campaign", "Synthetic campaign launch"]],
    )

    country = CountryMemory("CIV")
    country.import_csv("elections", demo_dir / "elections.csv", {})
    country.import_csv("parties", demo_dir / "parties.csv", {})
    country.import_csv("results", demo_dir / "results.csv", {})
    country.import_csv("events", demo_dir / "events.csv", {})

    pipeline = DocumentPipeline()
    manifesto = (
        "The Demo Reform Party presents a program before the legislative election. "
        "It supports transparent elections, democratic rights, institutional accountability "
        "and public investment in jobs. The party also defends national sovereignty and "
        "cooperation with regional partners."
    )
    segments = pipeline.ingest_text(
        manifesto,
        make_meta(
            country_iso3="CIV",
            party_id="DEMO",
            election_id="CIV_2020_LEG",
            doc_date=date(2020, 9, 1),
            doc_type="manifeste",
            source_path="examples/synthetic_manifesto",
            reliability=SourceReliability.OFFICIAL,
        ),
    )
    country.add_documents(segments)
    graph = PoliticalGraph("CIV")
    graph.load()
    graph.ingest(segments)
    graph.save()

    general = GeneralMemory()
    general.add(
        pipeline.ingest_text(
            "V-Party variables should be interpreted through explicit scales, evidence regimes and validation gates.",
            make_meta(
                country_iso3="GEN",
                doc_date=date(2020, 1, 1),
                doc_type="codebook_note",
                source_path="examples/synthetic_codebook_note",
                reliability=SourceReliability.ACADEMIC,
            ),
        )
    )

    demo_registry = _build_demo_registry(ROOT / "registry", demo_dir / "registry")
    runner = CompassRunner(
        country=country,
        general=general,
        registry=VPartyRegistry(demo_registry),
        search=ActiveSearchEngine(pipeline),
        graph=graph,
    )
    case = CaseKey(
        country_iso3="CIV",
        party_id="DEMO",
        election_id="CIV_2020_LEG",
        election_date=date(2020, 10, 31),
    )
    selected_variables = variables or ["v2pavote"]
    answers = runner.run_case(case, selected_variables)
    print("Full status: completed")
    for answer in answers:
        print(
            f"{answer.variable_id}: score={answer.score}, "
            f"confidence={answer.confidence}, abstained={answer.abstained}"
        )


def _configure_full_environment(reset: bool) -> None:
    data_dir = ROOT / "data" / "onyxia_real_architecture"
    if reset and data_dir.exists():
        shutil.rmtree(data_dir)
    os.environ.setdefault("COMPASS_DATA_DIR", str(data_dir))
    os.environ.setdefault("COMPASS_CHROMA_DIR", str(data_dir / "chroma"))
    os.environ.setdefault("COMPASS_SQLITE_PATH", str(data_dir / "compass_structured.db"))
    os.environ.setdefault("COMPASS_TRACE_DIR", str(data_dir / "traces"))
    os.environ.setdefault("COMPASS_GRAPH_PATH", str(data_dir / "political_graph.graphml"))
    os.environ.setdefault("COMPASS_SEARCH_MAX_ITERATIONS", "0")
    os.environ.setdefault("COMPASS_SUFFICIENCY_THRESHOLD", "0.0")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _build_demo_registry(source: Path, target: Path) -> Path:
    import yaml

    target.mkdir(parents=True, exist_ok=True)
    for source_file in source.glob("*.yaml"):
        payload = yaml.safe_load(source_file.read_text(encoding="utf-8"))
        if payload.get("variable_id") == "v2pavote":
            payload["adherence_passed"] = True
        (target / source_file.name).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "full"], help="Example execution mode")
    parser.add_argument("--reset", action="store_true", help="Delete generated demo data before full run")
    parser.add_argument(
        "--variables",
        default="v2pavote",
        help="Comma-separated variable IDs to run in full mode, for example v2paplur or v2pavote,v2paplur",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke()
    else:
        variables = [item.strip() for item in args.variables.split(",") if item.strip()]
        run_full(reset=args.reset, variables=variables)


if __name__ == "__main__":
    main()
