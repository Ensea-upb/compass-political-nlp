"""RAG-style chat engine over existing COMPASS memories.

This component is intentionally thin: it does not replace C01-C15. It sits on
top of ``CountryMemory.query_documents`` and the existing OpenAI-compatible LLM
client so the chat interface can be added without changing ingestion, retrieval,
reasoning, validation, or schemas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from compass.config import settings
from compass.llm_client import complete_chat

_SEGMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*:p\d{3}(?:c\d{3})?")
_MAX_PROMPT_CITATIONS = 6
_MAX_GENERAL_CONTEXT_ITEMS = 2
_MAX_EVIDENCE_TEXT_CHARS = 360
_MAX_PARENT_CONTEXT_CHARS = 240
_MAX_GENERAL_CONTEXT_CHARS = 360
_MAX_CHAT_HISTORY_MESSAGES = 2
_MAX_CHAT_HISTORY_CHARS = 350
_MAX_CHAT_OUTPUT_TOKENS = 650


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
    parent_text: str | None = None


@dataclass(frozen=True)
class GeneralContext:
    """Broader, non-cited context used to frame evidence interpretation."""

    ref_id: str
    segment_id: str
    text: str
    country_iso3: str
    party_id: str | None
    doc_date: str | None
    doc_type: str | None


@dataclass(frozen=True)
class ChatResponse:
    """Answer plus evidence used to produce it."""

    answer: str
    citations: list[Citation]
    model_used: str | None
    llm_used: bool
    retrieval_count: int
    general_context: list[GeneralContext] = field(default_factory=list)


class ChatEngine:
    """Conversational RAG facade over a ``CountryMemory`` instance."""

    def __init__(self, memory: Any, model_name: str | None = None) -> None:
        self.memory = memory
        self.model_name = model_name or _default_chat_model()

    def _answer_segment_lookup(self, question: str, segment_ids: list[str]) -> ChatResponse:
        if hasattr(self.memory, "fetch_records_by_ids"):
            records = self.memory.fetch_records_by_ids(segment_ids)
        else:
            texts = self.memory.fetch_by_ids(segment_ids)
            records = [
                {"segment_id": segment_id, "text": text, "meta": {}}
                for segment_id, text in texts.items()
            ]
        citations = build_citations(records)
        if not citations:
            return ChatResponse(
                answer="Je n'ai pas trouve ce segment exact dans l'index COMPASS.",
                citations=[],
                model_used=None,
                llm_used=False,
                retrieval_count=0,
            )
        answer_lines = ["Voici le passage exact demande dans l'index COMPASS :", ""]
        for citation in citations:
            answer_lines.append(f"[{citation.ref_id}] `{citation.segment_id}`")
            answer_lines.append(_source_label(citation))
            answer_lines.append(citation.text)
            answer_lines.append("")
        return ChatResponse(
            answer="\n".join(answer_lines).strip(),
            citations=citations,
            model_used=None,
            llm_used=False,
            retrieval_count=len(citations),
        )

    def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer a question using existing indexed COMPASS documents."""
        question = request.question.strip()
        if not question:
            raise ValueError("Question vide.")
        segment_ids = extract_segment_ids(question)
        if segment_ids:
            return self._answer_segment_lookup(question, segment_ids)
        retrieved = self.memory.query_documents(
            build_retrieval_query(question),
            as_of=request.as_of,
            k=request.k,
            party_id=request.party_id,
            include_unverified=request.include_unverified,
        )
        citations = build_citations(retrieved)
        if not citations:
            return ChatResponse(
                answer="Aucun passage pertinent n'a ete trouve dans la memoire COMPASS pour cette question.",
                citations=[],
                model_used=None,
                llm_used=False,
                retrieval_count=0,
            )
        retrieved = attach_parent_context(self.memory, retrieved)
        citations = build_citations(retrieved)
        general_context = retrieve_general_context(self.memory, question, request, retrieved)
        messages = build_messages(question, citations, request, general_context)
        try:
            answer = complete_chat(
                self.model_name,
                messages,
                temperature=0.0,
                max_tokens=min(settings.llm_max_tokens, _MAX_CHAT_OUTPUT_TOKENS),
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
                general_context=general_context,
            )
        except Exception as exc:
            fallback = build_extractive_answer(question, citations, exc)
            return ChatResponse(
                answer=fallback,
                citations=citations,
                model_used=self.model_name,
                llm_used=False,
                retrieval_count=len(retrieved),
                general_context=general_context,
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
                parent_text=_none_if_empty(item.get("parent_text")),
            )
        )
    return citations


def build_general_context_items(records: list[dict[str, Any]], limit: int = 3) -> list[GeneralContext]:
    items: list[GeneralContext] = []
    seen: set[str] = set()
    for item in records:
        segment_id = str(item.get("segment_id") or "")
        if not segment_id or segment_id in seen:
            continue
        seen.add(segment_id)
        meta = item.get("meta") or {}
        items.append(
            GeneralContext(
                ref_id=f"C{len(items) + 1}",
                segment_id=segment_id,
                text=str(item.get("text") or ""),
                country_iso3=str(meta.get("country_iso3") or ""),
                party_id=_none_if_empty(meta.get("party_id")),
                doc_date=_none_if_empty(meta.get("doc_date")),
                doc_type=_none_if_empty(meta.get("doc_type")),
            )
        )
        if len(items) >= limit:
            break
    return items


def build_messages(
    question: str,
    citations: list[Citation],
    request: ChatRequest,
    general_context: list[GeneralContext] | None = None,
) -> list[dict[str, str]]:
    prompt_citations = citations[:_MAX_PROMPT_CITATIONS]
    evidence_context = "\n\n".join(
        f"[{citation.ref_id}] "
        f"country={citation.country_iso3 or 'unknown'} party={citation.party_id or 'unknown'} "
        f"date={citation.doc_date or 'unknown'} type={citation.doc_type or 'unknown'} "
        f"reliability={citation.reliability or 'unknown'}\n"
        f"local_parent_context={_compact_parent_context(citation)}\n"
        f"{_excerpt(citation.text, max_chars=_MAX_EVIDENCE_TEXT_CHARS)}"
        for citation in prompt_citations
    )
    general_context_text = format_general_context_for_prompt(general_context or [])
    language = infer_answer_language(question)
    system = (
        "You are COMPASS Chat, a research assistant for political manifesto analysis. "
        "Answer only from the provided COMPASS evidence. Every substantive claim must end with an inline citation like [S1] or [S2]. "
        "Use the COMPASS general context only to understand the document frame; do not cite it as evidence and do not make claims that are absent from cited evidence. "
        "Do not add a separate bibliography or sources section; the interface adds sources separately. "
        "Do not infer motives, psychology, or hidden positions beyond the passages. "
        "Do not overinterpret; if a conclusion is not directly supported, phrase it cautiously or say the evidence is insufficient. "
        "Prefer short evidence-linked sentences over broad summaries. "
        f"Answer in {language}."
    )
    scope = f"as_of={request.as_of.isoformat()}"
    if request.party_id:
        scope += f", party_id={request.party_id}"
    user = (
        f"Scope: {scope}\n\n"
        f"Question: {question}\n\n"
        f"COMPASS general context (background only, not citation evidence):\n{general_context_text}\n\n"
        f"COMPASS cited evidence:\n{evidence_context}"
    )
    history = compact_history(
        request.history,
        max_messages=_MAX_CHAT_HISTORY_MESSAGES,
        max_chars=_MAX_CHAT_HISTORY_CHARS,
    )
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def attach_parent_context(memory: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add parent chunk text to retrieved child records when available."""
    parent_ids = list({
        (record.get("meta") or {}).get("parent_segment_id", "")
        for record in records
        if (record.get("meta") or {}).get("parent_segment_id")
    })
    if not parent_ids or not hasattr(memory, "fetch_by_ids"):
        return records
    try:
        parent_texts = memory.fetch_by_ids(parent_ids)
    except Exception:
        return records
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        parent_id = (item.get("meta") or {}).get("parent_segment_id", "")
        if parent_id and parent_id in parent_texts:
            item["parent_text"] = parent_texts[parent_id]
        enriched.append(item)
    return enriched


def retrieve_general_context(
    memory: Any,
    question: str,
    request: ChatRequest,
    evidence_records: list[dict[str, Any]],
) -> list[GeneralContext]:
    """Retrieve broad parent-level context without mixing it into citations."""
    try:
        records = memory.query_documents(
            build_general_context_query(question),
            as_of=request.as_of,
            k=_MAX_GENERAL_CONTEXT_ITEMS,
            party_id=request.party_id,
            include_unverified=request.include_unverified,
            include_parent_segments=True,
        )
    except TypeError:
        return build_general_context_items(evidence_records, limit=2)
    except Exception:
        return build_general_context_items(evidence_records, limit=2)
    return build_general_context_items(records, limit=_MAX_GENERAL_CONTEXT_ITEMS)


def build_general_context_query(question: str) -> str:
    return (
        f"{question} manifesto overall political program party priorities "
        "general orientation policy agenda election platform"
    )


def format_general_context_for_prompt(items: list[GeneralContext]) -> str:
    if not items:
        return "No separate general context retrieved."
    lines = []
    for item in items[:_MAX_GENERAL_CONTEXT_ITEMS]:
        lines.append(
            f"[{item.ref_id}] country={item.country_iso3 or 'unknown'} "
            f"party={item.party_id or 'unknown'} date={item.doc_date or 'unknown'} "
            f"type={item.doc_type or 'unknown'} segment={item.segment_id}\n"
            f"{_excerpt(item.text, max_chars=_MAX_GENERAL_CONTEXT_CHARS)}"
        )
    return "\n\n".join(lines)


def _compact_parent_context(citation: Citation) -> str:
    if not citation.parent_text:
        return "none"
    return _excerpt(citation.parent_text, max_chars=_MAX_PARENT_CONTEXT_CHARS)


def extract_segment_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in _SEGMENT_ID_RE.findall(text):
        if match not in seen:
            seen.add(match)
            ids.append(match)
    return ids


def infer_answer_language(question: str) -> str:
    lowered = question.lower()
    french_markers = ("francais", "français", "reponds en francais", "réponds en français", "en francais", "en français")
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
        "Reponse extractive COMPASS : le modele LLM n'a pas repondu, donc voici les passages les plus pertinents.",
        f"Question : {question}",
        "",
    ]
    for citation in citations[:5]:
        snippet = _excerpt(citation.text, max_chars=500)
        lines.append(f"[{citation.ref_id}] {snippet}")
    lines.append("")
    lines.append(f"Note technique : fallback declenche ({type(exc).__name__}).")
    return "\n".join(lines)


def format_citations(citations: list[Citation]) -> str:
    """Return Markdown source list for UI display."""
    if not citations:
        return "Aucune source retrouvee."
    lines = []
    for citation in citations:
        lines.append(f"- {_source_label(citation)}")
        lines.append(f"  segment: `{citation.segment_id}`")
        lines.append(f"  excerpt: \"{_excerpt(citation.text)}\"")
    return "\n".join(lines)


def build_retrieval_query(question: str) -> str:
    """Small query expansion for demo RAG without changing the user question."""
    lowered = question.lower()
    if any(token in lowered for token in ("economy", "economic", "emploi", "salaires", "économie", "economie")):
        return (
            f"{question} employment wages industry domestic market exports innovation "
            "taxation growth sustainable economy decent work jobs salaires emploi industrie"
        )
    return question


def _source_label(citation: Citation) -> str:
    return (
        f"[{citation.ref_id}] {citation.country_iso3 or 'UNK'} | "
        f"party={citation.party_id or 'party?'} | "
        f"date={citation.doc_date or 'date?'} | "
        f"{citation.doc_type or 'document'}"
    )


def _excerpt(text: str, max_chars: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _default_chat_model() -> str:
    if settings.judge_models:
        return settings.judge_models[0]
    return settings.hyde_model


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
