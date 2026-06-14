"""End-to-end public demo pipeline."""

from __future__ import annotations

from pathlib import Path

from compass.ingestion import load_text_document
from compass.reasoning import build_party_profile
from compass.retrieval import retrieve_theme_evidence
from compass.schemas import PartyProfile, ValidationReport
from compass.validation import validate_profile


def run_demo_pipeline(input_path: str | Path) -> tuple[PartyProfile, ValidationReport]:
    document = load_text_document(input_path)
    evidence = retrieve_theme_evidence(document)
    profile = build_party_profile(document, evidence)
    validation = validate_profile(profile)
    return profile, validation
