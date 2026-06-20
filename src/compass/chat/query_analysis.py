"""Structured question analysis for the conversational RAG retrieval plan."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from compass.config import settings

AnswerType = Literal[
    "position",
    "evidence",
    "comparison",
    "list",
    "chronology",
    "explanation",
]

_WORD_RE = re.compile(r"[^\W\d_][\w'-]*", re.UNICODE)
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_STOPWORDS = {
    "a", "about", "and", "are", "au", "aux", "avec", "ce", "ces", "cette",
    "comment", "dans", "de", "des", "does", "du", "en", "est", "et", "for",
    "give", "how", "i", "il", "la", "le", "les", "me", "of", "on", "parti",
    "party", "pour", "que", "quel", "quelle", "quelles", "qui", "quoi", "say",
    "says", "sur", "the", "this", "to", "tu", "un", "une", "what", "which",
    "with", "you",
}


class QueryAnalysis(BaseModel):
    """Validated retrieval plan produced before any corpus search."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actors: list[str] = Field(default_factory=list, max_length=4)
    themes: list[str] = Field(default_factory=list, max_length=6)
    period: str | None = Field(default=None, max_length=80)
    answer_type: AnswerType = "position"
    language: str = Field(default="und", pattern=r"^(?:[a-z]{2,3}|und)$")
    subqueries: list[str] = Field(min_length=1, max_length=5)
    method: Literal["llm", "deterministic"]

    @field_validator("actors", "themes", "subqueries")
    @classmethod
    def _clean_unique_values(cls, values: list[str]) -> list[str]:
        clean: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = " ".join(str(value).split()).strip()
            key = item.casefold()
            if item and len(item) <= 300 and key not in seen:
                seen.add(key)
                clean.append(item)
        return clean


class _LLMQueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actors: list[str] = Field(default_factory=list, max_length=4)
    themes: list[str] = Field(default_factory=list, max_length=6)
    period: str | None = Field(default=None, max_length=80)
    answer_type: AnswerType
    language: str = Field(pattern=r"^(?:[a-z]{2,3}|und)$")
    subqueries: list[str] = Field(min_length=2, max_length=5)


def analyze_question(
    question: str,
    *,
    scope: dict[str, Any],
    model_name: str,
    complete: Callable[..., str],
) -> QueryAnalysis:
    """Use the local LLM first and always retain a deterministic fallback."""
    fallback = deterministic_question_analysis(question, scope=scope)
    if not settings.chat_query_analysis_enabled:
        return fallback
    messages = _analysis_messages(question, scope)
    try:
        raw = complete(
            model_name,
            messages,
            temperature=0.0,
            max_tokens=settings.chat_query_analysis_max_tokens,
            response_format={"type": "json_object"},
        )
        payload = _LLMQueryAnalysis.model_validate(json.loads(raw))
        actors = _safe_actors(payload.actors, question, scope) or fallback.actors
        subqueries = _bounded_queries(
            question,
            payload.subqueries,
            settings.chat_query_analysis_max_subqueries,
        )
        if len(subqueries) < 2:
            subqueries = _bounded_queries(
                question,
                [*payload.subqueries, *fallback.subqueries[1:]],
                settings.chat_query_analysis_max_subqueries,
            )
        return QueryAnalysis(
            actors=actors,
            themes=payload.themes or fallback.themes,
            period=_safe_period(payload.period, question) or fallback.period,
            answer_type=payload.answer_type,
            language=payload.language,
            subqueries=subqueries,
            method="llm",
        )
    except Exception:
        return fallback


def deterministic_question_analysis(
    question: str,
    *,
    scope: dict[str, Any],
) -> QueryAnalysis:
    """Build a domain-agnostic retrieval plan without thematic dictionaries."""
    language = _detect_language(question)
    actors = _deterministic_actors(question, scope)
    themes = _salient_terms(question)
    period = _extract_period(question)
    answer_type = _answer_type(question)
    focus = " ".join([*actors, *themes]).strip()
    if language == "fr":
        complements = [
            f"{focus} position programme politique".strip(),
            f"{focus} engagements mesures preuves".strip(),
        ]
    else:
        complements = [
            f"{focus} political position manifesto".strip(),
            f"{focus} commitments policies evidence".strip(),
        ]
    return QueryAnalysis(
        actors=actors,
        themes=themes,
        period=period,
        answer_type=answer_type,
        language=language,
        subqueries=_bounded_queries(
            question,
            complements,
            settings.chat_query_analysis_max_subqueries,
        ),
        method="deterministic",
    )


def format_query_analysis(analysis: QueryAnalysis) -> str:
    """Human-readable, non-evidentiary representation for prompt inspection."""
    actors = ", ".join(analysis.actors) or "not specified"
    themes = ", ".join(analysis.themes) or "not specified"
    queries = "\n".join(f"- {query}" for query in analysis.subqueries)
    return (
        f"method: {analysis.method}\n"
        f"actors: {actors}\n"
        f"themes: {themes}\n"
        f"period: {analysis.period or 'not specified'}\n"
        f"answer_type: {analysis.answer_type}\n"
        f"language: {analysis.language}\n"
        f"retrieval_subqueries:\n{queries}"
    )


def _analysis_messages(question: str, scope: dict[str, Any]) -> list[dict[str, str]]:
    parties = [
        {
            "party_id": str(item.get("party_id") or ""),
            "name": str(item.get("name") or ""),
        }
        for item in scope.get("parties") or []
    ]
    system = (
        "You are a retrieval query planner for political documents. Do not answer the question. "
        "Return one JSON object and nothing else. Use exactly these keys: actors, themes, period, "
        "answer_type, language, subqueries. actors and themes are arrays of short strings. period is "
        "a string or null. answer_type is one of position, evidence, comparison, list, chronology, "
        "explanation. language is an ISO 639 code. Produce 2 to 4 complementary search queries. "
        "Preserve the user's concepts and named actors. Do not invent events, positions, dates, or facts."
    )
    user = (
        f"Active corpus parties: {json.dumps(parties, ensure_ascii=False)}\n"
        f"Corpus date range: {json.dumps(scope.get('document_dates') or [], ensure_ascii=False)}\n"
        f"Question: {question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _bounded_queries(question: str, candidates: list[str], limit: int) -> list[str]:
    values = [question, *candidates]
    anchor_tokens = set(_WORD_RE.findall(_normalize(question)))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = " ".join(str(value).split()).strip()
        key = query.casefold()
        query_tokens = set(_WORD_RE.findall(_normalize(query)))
        if unique and anchor_tokens and not query_tokens.intersection(anchor_tokens):
            continue
        if query and len(query) <= 500 and key not in seen:
            seen.add(key)
            unique.append(query)
        if len(unique) >= max(1, limit):
            break
    return unique


def _safe_period(period: str | None, question: str) -> str | None:
    if not period:
        return None
    period_tokens = set(re.findall(r"\d+", period))
    question_tokens = set(re.findall(r"\d+", question))
    if period_tokens and period_tokens.issubset(question_tokens):
        return period
    return None


def _scope_actor_labels(scope: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for party in scope.get("parties") or []:
        labels.extend(str(value) for value in (party.get("party_id"), party.get("name")) if value)
    return labels


def _safe_actors(actors: list[str], question: str, scope: dict[str, Any]) -> list[str]:
    question_key = _normalize(question)
    known = {_normalize(label): label for label in _scope_actor_labels(scope)}
    safe: list[str] = []
    for actor in actors:
        key = _normalize(actor)
        if key in known:
            safe.append(known[key])
        elif key and key in question_key:
            safe.append(actor)
    return list(dict.fromkeys(safe))[:4]


def _deterministic_actors(question: str, scope: dict[str, Any]) -> list[str]:
    normalized = _normalize(question)
    labels = _scope_actor_labels(scope)
    found = [label for label in labels if _normalize(label) in normalized]
    parties = scope.get("parties") or []
    references_active_party = any(
        marker in normalized
        for marker in ("ce parti", "du parti", "this party", "the party", "cette formation")
    )
    if not found and references_active_party and len(parties) == 1:
        party = parties[0]
        found = [str(party.get("name") or party.get("party_id") or "")]
    return [value for value in dict.fromkeys(found) if value][:4]


def _salient_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in _WORD_RE.findall(_normalize(question)):
        if len(token) >= 3 and token not in _STOPWORDS and token not in terms:
            terms.append(token)
    return terms[:6]


def _extract_period(question: str) -> str | None:
    dates = _DATE_RE.findall(question)
    years = [
        year for year in _YEAR_RE.findall(question)
        if not any(value.startswith(year) for value in dates)
    ]
    values = list(dict.fromkeys([*dates, *years]))
    return " to ".join(values[:2]) if values else None


def _answer_type(question: str) -> AnswerType:
    normalized = _normalize(question)
    if any(marker in normalized for marker in ("compare", "comparaison", "versus", "difference")):
        return "comparison"
    if any(marker in normalized for marker in ("preuve", "source", "passage", "evidence", "quote")):
        return "evidence"
    if any(marker in normalized for marker in ("evolution", "chronologie", "change", "before", "after")):
        return "chronology"
    if any(marker in normalized for marker in ("liste", "quels sont", "quelles sont", "priorites", "list")):
        return "list"
    if any(marker in normalized for marker in ("pourquoi", "comment", "why", "how")):
        return "explanation"
    return "position"


def _detect_language(question: str) -> str:
    normalized = _normalize(question)
    tokens = set(_WORD_RE.findall(normalized))
    markers = {
        "fr": {"avec", "comment", "dans", "est", "parti", "pour", "quel", "que", "sur"},
        "en": {"about", "does", "how", "party", "say", "the", "what", "which", "with"},
        "de": {"die", "partei", "sagt", "uber", "was", "welche", "zur"},
        "es": {"dice", "el", "partido", "que", "sobre", "cual", "posicion"},
        "pt": {"diz", "partido", "qual", "que", "sobre", "posicao"},
        "it": {"cosa", "dice", "il", "partito", "quale", "sulla"},
    }
    scores = {language: len(tokens & words) for language, words in markers.items()}
    if any(char in question.lower() for char in "àâçéèêëîïôùûü"):
        scores["fr"] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "und"


def _normalize(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().replace("'", " ").split())
