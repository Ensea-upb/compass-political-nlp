"""Synthetic public demo kept separate from the research pipeline.

This file exists only so the repository has a zero-credential quickstart. The
actual framework lives in the renamed modules extracted from ``compass_system``.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DemoTheme(str, Enum):
    ECONOMY = "economy"
    SOVEREIGNTY = "sovereignty"
    DEMOCRACY = "democracy"
    SOCIAL_POLICY = "social_policy"
    ENVIRONMENT = "environment"


class DemoDocument(BaseModel):
    name: str
    text: str
    language: str = "en"


class DemoPartyProfile(BaseModel):
    source_document: str
    themes: list[DemoTheme]
    evidence: dict[DemoTheme, list[str]]
    summary: str


class DemoValidationReport(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and all(self.checks.values())


THEME_KEYWORDS: dict[DemoTheme, tuple[str, ...]] = {
    DemoTheme.ECONOMY: ("jobs", "tax", "industry", "inflation", "investment"),
    DemoTheme.SOVEREIGNTY: ("sovereignty", "independence", "borders", "national"),
    DemoTheme.DEMOCRACY: ("democracy", "transparency", "elections", "rights"),
    DemoTheme.SOCIAL_POLICY: ("schools", "health", "housing", "youth"),
    DemoTheme.ENVIRONMENT: ("climate", "energy", "water", "land"),
}


def run_demo_pipeline(input_path: str | Path) -> tuple[DemoPartyProfile, DemoValidationReport]:
    document = _load_demo_document(input_path)
    evidence = _retrieve_demo_evidence(document)
    profile = _build_demo_profile(document, evidence)
    return profile, _validate_demo_profile(profile)


def _load_demo_document(path: str | Path) -> DemoDocument:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Document is empty: {file_path}")
    return DemoDocument(name=file_path.name, text=text)


def _retrieve_demo_evidence(document: DemoDocument) -> dict[DemoTheme, list[str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.text) if s.strip()]
    evidence: dict[DemoTheme, list[str]] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        matches = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        if matches:
            evidence[theme] = matches[:3]
    return evidence


def _build_demo_profile(
    document: DemoDocument,
    evidence: dict[DemoTheme, list[str]],
) -> DemoPartyProfile:
    themes = list(evidence)
    readable = ", ".join(theme.value.replace("_", " ") for theme in themes)
    summary = f"The document emphasizes {readable}." if readable else "No theme detected."
    return DemoPartyProfile(
        source_document=document.name,
        themes=themes,
        evidence=evidence,
        summary=summary,
    )


def _validate_demo_profile(profile: DemoPartyProfile) -> DemoValidationReport:
    checks = {
        "has_source_document": bool(profile.source_document),
        "has_detected_themes": bool(profile.themes),
        "has_evidence_for_each_theme": all(
            theme in profile.evidence and bool(profile.evidence[theme])
            for theme in profile.themes
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    return DemoValidationReport(status=status, checks=checks)
