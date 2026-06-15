"""RAG-style chat engine over existing COMPASS memories.

This component is intentionally thin: it does not replace C01-C15. It sits on
top of ``CountryMemory.query_documents`` and the existing OpenAI-compatible LLM
client so the chat interface can be added without changing ingestion, retrieval,
reasoning, validation, or schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from compass.config import settings
from compass.llm_client import complete_chat


@dataclass(frozen=True)
class ChatRequest:
    """A user question scoped to one country memory and optional party."""

    question: str
    as_of: date
    party_id: str | None = None
    k: int = 8
    include_unverified: bool = False
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    """Readable source reference returned with a chat answer."""

    ref_id: str
    segment_id: str
    text: str
    doc_id: str
    country_iso3: str
    party_id: str | None
    doc_date: str | None
    doc_type: str | None
    reliability: str | None


@dataclass(frozen=True)
class ChatResponse:
    """Answer plus evidence used to produce it."""

    answer: str
    citations: list[Citation]
    model_used: str | None
    llm_used: bool
    retrieval_count: int


class ChatEngine:
    """Conversational RAG facade over a ``CountryMemory`` instance."""

    def __init__(self, memory: Any, model_name: str | None = None) -> None:
        self.memory = memory
        self.model_name = model_name or _default_chat_model()

    def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer a question using existing indexed COMPASS documents."""
        question = request.question.strip()
        if not question:
            raise ValueError("Question vide.")
        retrieved = self.memory.query_documents(
            question,
            as_of=request.as_of,
            k=request.k,
            party_id=request.party_id,
            include_unverified=request.include_unverified,
        )
        citations = build_citations(retrieved)
        if not citations:
            return ChatResponse(
                answer="Aucun passage pertinent n'a été trouvé dans la mémoire COMPASS pour cette question.",
                citations=[],
                model_used=None,
                llm_used=False,
                retrieval_count=0,
            )
        messages = build_messages(question, citations, request)
        try:
            answer = complete_chat(
                self.model_name,
                messages,
                temperature=0.0,
                max_tokens=min(settings.llm_max_tokens, 900),
            )
            if not answer:
                raise RuntimeError("Empty LLM response")
            answer = strip_appended_sources(answer)
            return ChatResponse(
                answer=answer,
                citations=citations,
                model_used=self.model_name,
                llm_used=True,
                retrieval_count=len(retrieved),
            )
        except Exception as exc:
            fallback = build_extractive_answer(question, citations, exc)
            return ChatResponse(
                answer=fallback,
                citations=citations,
                model_used=self.model_name,
                llm_used=False,
                retrieval_count=len(retrieved),
            )


def build_citations(retrieved: list[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, item in enumerate(retrieved, start=1):
        meta = item.get("meta") or {}
        citations.append(
            Citation(
                ref_id=f"S{idx}",
                segment_id=str(item.get("segment_id") or ""),
                text=str(item.get("text") or ""),
                doc_id=str(meta.get("doc_id") or ""),
                country_iso3=str(meta.get("country_iso3") or ""),
                party_id=_none_if_empty(meta.get("party_id")),
                doc_date=_none_if_empty(meta.get("doc_date")),
                doc_type=_none_if_empty(meta.get("doc_type")),
                reliability=_none_if_empty(meta.get("reliability")),
            )
        )
    return citations


def build_messages(question: str, citations: list[Citation], request: ChatRequest) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{citation.ref_id}] "
        f"party={citation.party_id or 'unknown'} date={citation.doc_date or 'unknown'} "
        f"type={citation.doc_type or 'unknown'} reliability={citation.reliability or 'unknown'}\n"
        f"{citation.text}"
        for citation in citations
    )
    language = infer_answer_language(question)
    system = (
        "You are COMPASS Chat, a research assistant for political manifesto analysis. "
        "Answer only from the provided COMPASS evidence. Cite sources inline with [S1], [S2], etc. "
        "Do not add a separate bibliography or sources section; the interface adds sources separately. "
        "If the evidence is insufficient, say so clearly. Keep the answer concise and analytical. "
        f"Answer in {language}."
    )
    scope = f"as_of={request.as_of.isoformat()}"
    if request.party_id:
        scope += f", party_id={request.party_id}"
    user = f"Scope: {scope}\n\nQuestion: {question}\n\nCOMPASS evidence:\n{context}"
    history = compact_history(request.history)
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def infer_answer_language(question: str) -> str:
    lowered = question.lower()
    french_markers = ("français", "francais", "réponds en français", "reponds en francais", "en français", "en francais")
    if any(marker in lowered for marker in french_markers):
        return "French"
    return "the user's language"


def compact_history(history: list[dict[str, str]], max_messages: int = 4, max_chars: int = 700) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for msg in history[-max_messages:]:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = strip_appended_sources(msg.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        compacted.append({"role": role, "content": content})
    return compacted


def strip_appended_sources(answer: str) -> str:
    markers = ("\nSources\n", "\n### Sources\n", "\n## Sources\n")
    stripped = answer.strip()
    for marker in markers:
        idx = stripped.find(marker)
        if idx != -1:
            return stripped[:idx].strip()
    return stripped

def build_extractive_answer(question: str, citations: list[Citation], exc: Exception) -> str:
    lines = [
        "Réponse extractive COMPASS : le modèle LLM n'a pas répondu, donc voici les passages les plus pertinents.",
        f"Question : {question}",
        "",
    ]
    for citation in citations[:5]:
        snippet = citation.text.strip()
        if len(snippet) > 500:
            snippet = snippet[:497].rstrip() + "..."
        lines.append(f"[{citation.ref_id}] {snippet}")
    lines.append("")
    lines.append(f"Note technique : fallback déclenché ({type(exc).__name__}).")
    return "\n".join(lines)


def format_citations(citations: list[Citation]) -> str:
    """Return Markdown source list for UI display."""
    if not citations:
        return "Aucune source retrouvée."
    lines = []
    for citation in citations:
        label = (
            f"[{citation.ref_id}] {citation.country_iso3 or 'UNK'} "
            f"{citation.party_id or 'party?'} {citation.doc_date or 'date?'} "
            f"{citation.doc_type or 'document'}"
        )
        lines.append(f"- {label} - `{citation.segment_id}`")
    return "\n".join(lines)


def _default_chat_model() -> str:
    if settings.judge_models:
        return settings.judge_models[0]
    return settings.hyde_model


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None