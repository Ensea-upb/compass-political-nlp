"""Validation rules for the public demo output."""

from __future__ import annotations

from compass.schemas import PartyProfile, ValidationReport


def validate_profile(profile: PartyProfile) -> ValidationReport:
    checks = {
        "has_source_document": bool(profile.source_document),
        "has_detected_themes": bool(profile.themes),
        "has_evidence_for_each_theme": all(
            theme in profile.evidence and bool(profile.evidence[theme])
            for theme in profile.themes
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    return ValidationReport(status=status, checks=checks)
