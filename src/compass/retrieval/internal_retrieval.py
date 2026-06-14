"""Simple deterministic retrieval over a synthetic manifesto."""

from __future__ import annotations

import re

from compass.schemas import Document, PoliticalTheme

THEME_KEYWORDS: dict[PoliticalTheme, tuple[str, ...]] = {
    PoliticalTheme.ECONOMY: ("jobs", "tax", "industry", "inflation", "investment"),
    PoliticalTheme.SOVEREIGNTY: ("sovereignty", "independence", "borders", "national"),
    PoliticalTheme.DEMOCRACY: ("democracy", "transparency", "elections", "rights"),
    PoliticalTheme.SOCIAL_POLICY: ("schools", "health", "housing", "youth"),
    PoliticalTheme.ENVIRONMENT: ("climate", "energy", "water", "land"),
}


def retrieve_theme_evidence(document: Document) -> dict[PoliticalTheme, list[str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.text) if s.strip()]
    evidence: dict[PoliticalTheme, list[str]] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        matches = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        if matches:
            evidence[theme] = matches[:3]
    return evidence
