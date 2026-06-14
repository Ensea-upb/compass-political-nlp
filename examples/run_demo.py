"""Run the public COMPASS demo on a synthetic manifesto."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.demo import run_demo_pipeline


def main() -> None:
    input_path = ROOT / "examples" / "sample_manifesto.txt"
    output_path = ROOT / "examples" / "sample_party_profile.json"

    profile, validation = run_demo_pipeline(input_path)
    output_path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    themes = ", ".join(theme.value for theme in profile.themes[:3])
    print(f"Document loaded: {input_path.name}")
    print(f"Detected political themes: {themes}")
    print("Generated party profile: examples/sample_party_profile.json")
    print(f"Validation status: {validation.status}")


if __name__ == "__main__":
    main()
