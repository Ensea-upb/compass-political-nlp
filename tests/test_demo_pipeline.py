from pathlib import Path

from compass import run_demo_pipeline
from compass.schemas import PoliticalTheme


def test_demo_pipeline_detects_core_themes() -> None:
    root = Path(__file__).resolve().parents[1]
    profile, validation = run_demo_pipeline(root / "examples" / "sample_manifesto.txt")

    assert validation.passed
    assert PoliticalTheme.ECONOMY in profile.themes
    assert PoliticalTheme.SOVEREIGNTY in profile.themes
    assert PoliticalTheme.DEMOCRACY in profile.themes
