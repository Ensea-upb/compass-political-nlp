"""Build CSV manifests for automatic Manifesto ingestion.

The generated CSV is consumed by examples/run_manifesto_pdf_ingestion.py.
It can be built from the Manifesto API core dataset or from a local core CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.manifesto_api import ManifestoAPI, debug_core_decoding

HEADER = [
    "key",
    "metadata_version",
    "country_iso3",
    "party_id",
    "election_id",
    "doc_date",
    "doc_type",
    "language",
    "reliability",
    "pdf_url",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Manifesto ingestion CSV for COMPASS.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--core-version", help="Manifesto core dataset version, for example MPDS2024a")
    source.add_argument("--core-csv", type=Path, help="Local Manifesto core CSV already downloaded")
    parser.add_argument("--core-kind", default="dta", help="API get_core kind, usually dta or xlsx")
    parser.add_argument("--metadata-version", required=True, help="Corpus metadata version, for example 2024-1")
    parser.add_argument("--country-iso3", required=True, help="Output country ISO3, for example DEU")
    parser.add_argument("--country-code", help="Filter core country code, for example 41")
    parser.add_argument("--election-date", help="Filter Manifesto date, for example 200909")
    parser.add_argument("--party", action="append", help="Optional party id filter; can be repeated")
    parser.add_argument("--language", default="und")
    parser.add_argument("--election-id", help="Override output election_id; default ISO3_year")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="Print loaded core columns and sample filter values")
    parser.add_argument("--list-core-versions", action="store_true", help="Print available Manifesto core versions and exit")
    parser.add_argument("--debug-core-payload", action="store_true", help="Print a redacted summary of get_core payload and exit")
    args = parser.parse_args()

    if args.list_core_versions:
        print_core_versions()
        return
    if args.debug_core_payload:
        debug_core_payload(args)
        return

    records = _load_records(args)
    if args.inspect:
        inspect_records(records)
    rows = build_manifest_rows(
        records,
        metadata_version=args.metadata_version,
        country_iso3=args.country_iso3,
        country_code=args.country_code,
        election_date=args.election_date,
        parties=set(args.party or []),
        language=args.language,
        election_id=args.election_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if not rows:
        print("No rows matched. Re-run with --inspect and check country/date values, or try --core-kind dta.")


def print_core_versions() -> None:
    payload = ManifestoAPI().list_core_versions()
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:8000])


def debug_core_payload(args: argparse.Namespace) -> None:
    if not args.core_version:
        raise SystemExit("--debug-core-payload requires --core-version.")
    payload = ManifestoAPI().get_core_payload(args.core_version, kind=args.core_kind)
    summary = summarize_payload(payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": list(payload.keys()),
        }
        if isinstance(payload.get("content"), str):
            content = payload["content"]
            summary["content_length"] = len(content)
            summary["content_prefix"] = content[:80]
            summary["decoding"] = debug_core_decoding(content, kind="")
        for key in ("error", "message", "status", "kind", "file", "filename"):
            if key in payload:
                summary[key] = payload[key]
        return summary
    if isinstance(payload, list):
        return {"type": "list", "length": len(payload), "first_item": payload[0] if payload else None}
    return {"type": type(payload).__name__, "preview": str(payload)[:1000]}

def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.core_csv:
        with args.core_csv.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    return ManifestoAPI().get_core_records(args.core_version, kind=args.core_kind)


def build_manifest_rows(
    records: list[dict[str, Any]],
    *,
    metadata_version: str,
    country_iso3: str,
    country_code: str | None = None,
    election_date: str | None = None,
    parties: set[str] | None = None,
    language: str = "und",
    election_id: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        normalized = {str(key).lower(): value for key, value in record.items()}
        party = _clean(_first_value(normalized, "party", "party_id"))
        mp_date = _clean(_first_value(normalized, "date", "edate", "election_date"))
        country = _clean(_first_value(normalized, "country", "country_id", "countrycode"))
        if not party or not mp_date:
            continue
        if country_code and country != str(country_code):
            continue
        if election_date and _digits(mp_date) != _digits(str(election_date)):
            continue
        if parties and party not in parties:
            continue
        doc_date = doc_date_from_manifesto_date(mp_date)
        out_election_id = election_id or f"{country_iso3.upper()}_{doc_date.year}"
        rows.append(
            {
                "key": f"{party}_{mp_date}",
                "metadata_version": metadata_version,
                "country_iso3": country_iso3.upper(),
                "party_id": party,
                "election_id": out_election_id,
                "doc_date": doc_date.isoformat(),
                "doc_type": "manifesto",
                "language": language,
                "reliability": "official",
                "pdf_url": "",
            }
        )
    rows.sort(key=lambda row: (row["election_id"], row["party_id"], row["key"]))
    return rows


def inspect_records(records: list[dict[str, Any]], limit: int = 5) -> None:
    print(f"Loaded core records: {len(records)}")
    if not records:
        return
    columns = list(records[0].keys())
    print("Columns: " + ", ".join(str(col) for col in columns[:40]))
    print("Sample rows:")
    for record in records[:limit]:
        normalized = {str(key).lower(): value for key, value in record.items()}
        print(
            "  country={country} party={party} date={date}".format(
                country=_clean(_first_value(normalized, "country", "country_id", "countrycode")),
                party=_clean(_first_value(normalized, "party", "party_id")),
                date=_clean(_first_value(normalized, "date", "edate", "election_date")),
            )
        )


def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())

def doc_date_from_manifesto_date(value: str) -> date:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 8:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    if len(digits) == 6:
        return date(int(digits[:4]), int(digits[4:6]), 1)
    if len(digits) == 4:
        return date(int(digits), 1, 1)
    raise ValueError(f"Unsupported Manifesto date: {value}")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


if __name__ == "__main__":
    main()