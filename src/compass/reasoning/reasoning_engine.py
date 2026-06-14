"""Explainable reasoning layer for the public demo."""

from __future__ import annotations

from compass.schemas import Document, PartyProfile, PoliticalTheme


def build_party_profile(
    document: Document,
    evidence: dict[PoliticalTheme, list[str]],
) -> PartyProfile:
    themes = list(evidence)
    readable = ", ".join(theme.value.replace("_", " ") for theme in themes)
    summary = f"The document emphasizes {readable}." if readable else "No theme detected."
    return PartyProfile(
        source_document=document.name,
        themes=themes,
        evidence=evidence,
        summary=summary,
    )
