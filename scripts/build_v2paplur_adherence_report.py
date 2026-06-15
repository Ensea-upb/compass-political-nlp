"""Generate the R-1 adherence report for v2paplur."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.adherence import run_v2paplur_adherence, v2paplur_report_markdown


def main() -> None:
    sheet, generated_tests, results = run_v2paplur_adherence()
    report = v2paplur_report_markdown(sheet, generated_tests, results)
    out = ROOT / "docs" / "adherence" / "v2paplur_R1_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    passed = sum(1 for result in results if result.passed)
    print(f"v2paplur R-1 report written: {out}")
    print(f"Curated probes passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()