"""Build or update a country political graph from an existing Chroma index."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.schemas import DocumentMeta, Segment, SourceReliability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PoliticalGraph from documents already indexed in CountryMemory."
    )
    parser.add_argument("--country", required=True, help="Three-letter country ISO3 code")
    parser.add_argument("--party", help="Optional party id filter")
    parser.add_argument("--reset", action="store_true", help="Rebuild the country graph from scratch")
    args = parser.parse_args()

    from compass.country_memory import CountryMemory
    from compass.political_graph import PoliticalGraph

    memory = CountryMemory(args.country)
    graph = PoliticalGraph(args.country)
    if args.reset and graph.storage_path.exists():
        graph.storage_path.unlink()
    else:
        graph.load()

    records = memory.list_document_records(party_id=args.party, parent_only=True)
    segments, skipped = records_to_segments(records, fallback_country=args.country)
    new_edges = graph.ingest(segments)
    graph.save()

    print(f"Indexed records read: {len(records)}")
    print(f"Graph source segments: {len(segments)}")
    print(f"Skipped records without valid metadata: {skipped}")
    print(f"New graph edges: {new_edges}")
    print(f"Total graph edges: {graph.edge_count}")
    print(f"Graph written: {graph.storage_path}")


def records_to_segments(
    records: list[dict],
    fallback_country: str,
) -> tuple[list[Segment], int]:
    segments: list[Segment] = []
    skipped = 0
    for record in records:
        metadata = record.get("meta") or {}
        try:
            doc_date = date.fromisoformat(str(metadata.get("doc_date") or ""))
        except ValueError:
            skipped += 1
            continue
        reliability_value = str(metadata.get("reliability") or SourceReliability.UNKNOWN.value)
        try:
            reliability = SourceReliability(reliability_value)
        except ValueError:
            reliability = SourceReliability.UNKNOWN
        segment_id = str(record.get("segment_id") or "")
        doc_id = str(metadata.get("doc_id") or segment_id.split(":", 1)[0] or "unknown")
        meta = DocumentMeta(
            doc_id=doc_id,
            country_iso3=str(metadata.get("country_iso3") or fallback_country).upper(),
            party_id=str(metadata.get("party_id") or "") or None,
            doc_date=doc_date,
            doc_type=str(metadata.get("doc_type") or "unknown"),
            language=str(metadata.get("language") or "und"),
            reliability=reliability,
        )
        segments.append(Segment(
            segment_id=segment_id,
            doc_id=doc_id,
            text=str(record.get("text") or ""),
            meta=meta,
            parent_segment_id=str(metadata.get("parent_segment_id") or "") or None,
        ))
    return segments, skipped


if __name__ == "__main__":
    main()
