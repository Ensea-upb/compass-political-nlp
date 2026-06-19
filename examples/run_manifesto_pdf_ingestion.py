"""Download Manifesto Project PDFs and ingest them into the real COMPASS pipeline.

Examples:
    MANIFESTO_API_KEY=... python examples/run_manifesto_pdf_ingestion.py \
        --keys 41320_200909 --country-iso3 DEU --party-id 41320 \
        --election-id DEU_2009 --doc-date 2009-09-01

    MANIFESTO_API_KEY=... python examples/run_manifesto_pdf_ingestion.py \
        --manifest examples/manifesto_pdf_manifest_sample.csv --reset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.manifesto_api import ManifestoAPI, ManifestoAPIError, find_pdf_url
from compass.schemas import SourceReliability


@dataclass
class ManifestoIngestionRow:
    key: str
    country_iso3: str
    party_id: str | None
    election_id: str | None
    doc_date: date
    metadata_version: str | None = None
    doc_type: str = "manifesto"
    language: str = "und"
    reliability: SourceReliability = SourceReliability.OFFICIAL
    pdf_url: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Manifesto PDFs, then fall back to Manifesto API text if PDFs are blocked.")
    parser.add_argument("--manifest", type=Path, help="CSV manifest with key, country_iso3 and doc_date columns")
    parser.add_argument("--keys", nargs="*", help="Manifesto keys such as party_date; useful for one-off downloads")
    parser.add_argument("--metadata-version", help="Manifesto metadata version, for example 2024-1")
    parser.add_argument("--country-iso3", help="Country ISO3 for all --keys rows")
    parser.add_argument("--party-id", help="Party id for all --keys rows")
    parser.add_argument("--election-id", help="Election id for all --keys rows")
    parser.add_argument("--doc-date", help="Document date for all --keys rows, YYYY-MM-DD")
    parser.add_argument("--language", default="und", help="Language hint for all --keys rows")
    parser.add_argument("--pdf-field", help="Optional dotted metadata field containing the PDF URL")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data" / "manifesto_pdfs")
    parser.add_argument("--text-dir", type=Path, default=ROOT / "data" / "manifesto_texts")
    parser.add_argument("--translation", help="Optional texts_and_annotations translation, for example en")
    parser.add_argument("--no-text-fallback", action="store_true", help="Fail instead of ingesting texts_and_annotations when PDF download is blocked")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "manifesto_pdf_ingestion_report.json")
    parser.add_argument("--dry-run", action="store_true", help="Resolve metadata and URLs without downloading or indexing")
    parser.add_argument("--print-metadata", action="store_true", help="Print raw metadata for debugging URL fields")
    parser.add_argument("--limit", type=int, help="Limit the number of rows processed")
    parser.add_argument("--reset", action="store_true", help="Delete generated Manifesto PDF and index data before running")
    parser.add_argument("--no-graph", action="store_true", help="Disable political graph extraction")
    args = parser.parse_args()

    rows = _load_rows(args)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.limit == 0:
        print("Manifesto ingestion CLI ready; --limit 0 selected, no rows processed.")
        return
    if not rows:
        raise SystemExit("No Manifesto rows to process. Use --manifest or --keys.")

    if args.reset and not args.dry_run:
        _reset_generated_data(args.download_dir, args.text_dir)

    os.environ.setdefault("COMPASS_DATA_DIR", str(ROOT / "data" / "manifesto_ingestion"))
    os.environ.setdefault("COMPASS_CHROMA_DIR", str(ROOT / "data" / "manifesto_ingestion" / "chroma"))
    os.environ.setdefault("COMPASS_SQLITE_PATH", str(ROOT / "data" / "manifesto_ingestion" / "compass_structured.db"))
    os.environ.setdefault("COMPASS_TRACE_DIR", str(ROOT / "data" / "manifesto_ingestion" / "traces"))
    os.environ.setdefault("COMPASS_GRAPH_PATH", str(ROOT / "data" / "manifesto_ingestion" / "political_graph.graphml"))
    from compass.config import settings
    from compass.country_memory import CountryMemory
    from compass.document_pipeline import DocumentPipeline, make_meta
    from compass.political_graph import PoliticalGraph

    settings.ensure_dirs()

    api = ManifestoAPI()
    pipeline = DocumentPipeline()
    memories: dict[str, CountryMemory] = {}
    graphs: dict[str, PoliticalGraph] = {}
    report: list[dict[str, Any]] = []

    for row in rows:
        entry: dict[str, Any] = {
            "key": row.key,
            "country_iso3": row.country_iso3,
            "party_id": row.party_id,
            "election_id": row.election_id,
            "doc_date": row.doc_date.isoformat(),
            "status": "pending",
        }
        try:
            metadata = _resolve_metadata(api, row, args.pdf_field)
            pdf_url = row.pdf_url or metadata.get("pdf_url")
            text_key = metadata.get("text_key") or row.key
            entry["pdf_url"] = pdf_url
            entry["text_key"] = text_key
            if args.print_metadata:
                entry["metadata"] = metadata.get("raw", {})
            if args.dry_run:
                entry["status"] = "resolved"
                report.append(entry)
                if pdf_url:
                    print(f"{row.key}: would download {pdf_url}")
                elif not args.no_text_fallback:
                    print(f"{row.key}: no PDF URL found; would try texts_and_annotations for {text_key}")
                else:
                    print(f"{row.key}: no PDF URL found")
                continue

            memory = memories.setdefault(row.country_iso3.upper(), CountryMemory(row.country_iso3))
            graph = None
            if not args.no_graph:
                country_key = row.country_iso3.upper()
                if country_key not in graphs:
                    graphs[country_key] = PoliticalGraph(country_key)
                    graphs[country_key].load()
                graph = graphs[country_key]
            if pdf_url:
                pdf_path = args.download_dir / row.country_iso3.upper() / f"{_safe_name(row.key)}.pdf"
                try:
                    api.download_pdf(pdf_url, pdf_path)
                    meta = make_meta(
                        country_iso3=row.country_iso3,
                        party_id=row.party_id,
                        election_id=row.election_id,
                        doc_date=row.doc_date,
                        doc_type=row.doc_type,
                        language=row.language,
                        source_url=pdf_url,
                        source_path=str(pdf_path),
                        reliability=row.reliability,
                    )
                    segments = pipeline.ingest_pdf(pdf_path, meta)
                    memory.add_documents(segments)
                    graph_edges = _update_graph(graph, segments)
                    entry.update({
                        "status": "ingested_pdf",
                        "pdf_path": str(pdf_path),
                        "segments": len(segments),
                        "graph_new_edges": graph_edges,
                        "graph_total_edges": graph.edge_count if graph else 0,
                    })
                    print(f"{row.key}: downloaded PDF and indexed ({len(segments)} segments)")
                    report.append(entry)
                    continue
                except ManifestoAPIError as exc:
                    entry["pdf_error"] = str(exc)
                    if args.no_text_fallback:
                        raise
                    print(f"{row.key}: PDF blocked; trying texts_and_annotations fallback")

            if args.no_text_fallback:
                entry["status"] = "missing_pdf_url"
                entry["hint"] = "Run again with --print-metadata, then pass --pdf-field FIELD if the URL is exposed under a custom field."
                report.append(entry)
                print(f"{row.key}: no PDF URL found")
                continue

            text_path = _ingest_text_fallback(
                api=api,
                pipeline=pipeline,
                memory=memory,
                row=row,
                text_key=text_key,
                text_dir=args.text_dir,
                translation=args.translation,
                make_meta=make_meta,
                graph=graph,
            )
            entry.update({"status": "ingested_api_text", **text_path})
            print(f"{row.key}: indexed API text fallback ({text_path['segments']} segments)")
        except ManifestoAPIError as exc:
            entry.update({"status": "api_error", "error": str(exc)})
            report.append(entry)
            print(f"{row.key}: API error: {exc}")
            continue
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)})
            report.append(entry)
            print(f"{row.key}: error: {exc}")
            continue
        report.append(entry)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {args.report}")


def _resolve_metadata(api: ManifestoAPI, row: ManifestoIngestionRow, pdf_field: str | None) -> dict[str, Any]:
    if row.pdf_url:
        return {"pdf_url": row.pdf_url, "text_key": row.key, "raw": {}}
    documents = api.resolve_documents([row.key], version=row.metadata_version, pdf_field=pdf_field)
    if not documents:
        return {"pdf_url": None, "text_key": row.key, "raw": {}}
    doc = documents[0]
    return {
        "pdf_url": doc.pdf_url or find_pdf_url(doc.metadata, preferred_field=pdf_field),
        "text_key": doc.key,
        "raw": doc.metadata,
    }


def _ingest_text_fallback(
    *,
    api: ManifestoAPI,
    pipeline: Any,
    memory: Any,
    row: ManifestoIngestionRow,
    text_key: str,
    text_dir: Path,
    translation: str | None,
    make_meta: Any,
    graph: Any | None = None,
) -> dict[str, Any]:
    texts = api.texts_and_annotations([text_key], version=row.metadata_version, translation=translation)
    if not texts and text_key != row.key:
        texts = api.texts_and_annotations([row.key], version=row.metadata_version, translation=translation)
    if not texts:
        raise ManifestoAPIError(f"No machine-readable text returned for {text_key}")
    api_text = texts[0]
    text_path = text_dir / row.country_iso3.upper() / f"{_safe_name(row.key)}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(api_text.text, encoding="utf-8")
    meta = make_meta(
        country_iso3=row.country_iso3,
        party_id=row.party_id,
        election_id=row.election_id,
        doc_date=row.doc_date,
        doc_type=f"{row.doc_type}_api_text",
        language=row.language,
        source_url="https://manifesto-project.wzb.eu/api/v1/texts_and_annotations",
        source_path=str(text_path),
        reliability=row.reliability,
    )
    segments = pipeline.ingest_text(api_text.text, meta)
    memory.add_documents(segments)
    graph_edges = _update_graph(graph, segments)
    return {
        "text_key": api_text.key,
        "text_path": str(text_path),
        "segments": len(segments),
        "graph_new_edges": graph_edges,
        "graph_total_edges": graph.edge_count if graph else 0,
    }


def _update_graph(graph: Any | None, segments: list[Any]) -> int:
    if graph is None:
        return 0
    new_edges = graph.ingest(segments)
    graph.save()
    return new_edges

def _load_rows(args: argparse.Namespace) -> list[ManifestoIngestionRow]:
    rows: list[ManifestoIngestionRow] = []
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                rows.append(_row_from_mapping(raw))
    if args.keys:
        if not args.country_iso3 or not args.doc_date:
            raise SystemExit("--keys requires --country-iso3 and --doc-date.")
        for key in args.keys:
            rows.append(
                ManifestoIngestionRow(
                    key=key,
                    country_iso3=args.country_iso3,
                    party_id=args.party_id,
                    election_id=args.election_id,
                    doc_date=date.fromisoformat(args.doc_date),
                    metadata_version=args.metadata_version,
                    language=args.language,
                )
            )
    return rows


def _row_from_mapping(raw: dict[str, str]) -> ManifestoIngestionRow:
    def get(name: str, default: str = "") -> str:
        return (raw.get(name) or default).strip()

    reliability_raw = get("reliability", SourceReliability.OFFICIAL.value)
    try:
        reliability = SourceReliability(reliability_raw)
    except ValueError:
        reliability = SourceReliability.UNKNOWN
    return ManifestoIngestionRow(
        key=get("key") or get("manifesto_key") or get("manifesto_id"),
        country_iso3=get("country_iso3"),
        party_id=get("party_id") or None,
        election_id=get("election_id") or None,
        doc_date=date.fromisoformat(get("doc_date") or get("election_date")),
        metadata_version=get("metadata_version") or get("version") or None,
        doc_type=get("doc_type", "manifesto"),
        language=get("language", "und"),
        reliability=reliability,
        pdf_url=get("pdf_url") or None,
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _reset_generated_data(download_dir: Path, text_dir: Path) -> None:
    for path in (download_dir, text_dir, ROOT / "data" / "manifesto_ingestion"):
        if path.exists():
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
