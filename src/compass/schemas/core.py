"""Small public schemas used by the synthetic demo pipeline."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class PoliticalTheme(str, Enum):
    ECONOMY = "economy"
    SOVEREIGNTY = "sovereignty"
    DEMOCRACY = "democracy"
    SOCIAL_POLICY = "social_policy"
    ENVIRONMENT = "environment"


class Document(BaseModel):
    name: str
    text: str
    language: str = "en"
    loaded_at: date = Field(default_factory=date.today)


class PartyProfile(BaseModel):
    source_document: str
    themes: list[PoliticalTheme]
    evidence: dict[PoliticalTheme, list[str]]
    summary: str


class ValidationReport(BaseModel):
    status: str
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and all(self.checks.values())
