from pathlib import Path

from compass.demo import DemoTheme, run_demo_pipeline


def test_demo_pipeline_detects_core_themes() -> None:
    root = Path(__file__).resolve().parents[1]
    profile, validation = run_demo_pipeline(root / "examples" / "sample_manifesto.txt")

    assert validation.passed
    assert DemoTheme.ECONOMY in profile.themes
    assert DemoTheme.SOVEREIGNTY in profile.themes
    assert DemoTheme.DEMOCRACY in profile.themes
