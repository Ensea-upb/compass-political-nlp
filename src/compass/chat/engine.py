"""RAG-style chat engine over existing COMPASS memories.

This component is intentionally thin: it does not replace C01-C15. It sits on
top of ``CountryMemory.query_documents`` and the existing OpenAI-compatible LLM
client so the chat interface can be added without changing ingestion, retrieval,
reasoning, validation, or schemas.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from compass.config import settings
from compass.llm_client import complete_chat

_SEGMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*:p\d{3}(?:c\d{3})?")
_MAX_GENERAL_CONTEXT_ITEMS = 1
_MAX_PARENT_CONTEXT_CHARS = 90
_MAX_GENERAL_CONTEXT_CHARS = 150
_MAX_ANALYTICAL_CONTEXT_CHARS = 420
_MAX_CHAT_HISTORY_MESSAGES = 1
_MAX_CHAT_HISTORY_CHARS = 160
_MAX_CHAT_OUTPUT_TOKENS = 350
_ANSWER_REF_RE = re.compile(r"\[(A|C\d+|S\d+)\]")
_INSUFFICIENT_MARKERS = (
    "insufficient evidence",
    "provided evidence is insufficient",
    "not enough evidence",
    "does not answer",
    "preuves insuffisantes",
    "elements insuffisants",
    "éléments insuffisants",
    "ne permet pas",
)

_ROUTE_VALIDATION_POLICIES = {
    "direct_lookup": "none",
    "corpus_scope": "none",
    "evidence_query": "strict_evidence",
    "FOLLOW_UP_SOURCES": "none",
    "OUT_OF_CORPUS": "none",
    "COMPARISON_NEEDS_MORE_CORPUS": "none",
    "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA": "none",
}
_ROUTING_MODES = {"deterministic", "llm"}

_ANALYTICAL_LENSES = {
    "democracy": {
        "triggers": {
            "democracy", "democratic", "democratie", "democratique",
            "rights", "freedom", "parliament", "election", "constitutional",
            "citizens", "participation",
        },
        "frame": (
            "For democracy questions, separate institutions, rights, participation, "
            "rule of law, electoral competition, and social-democratic or constitutional claims."
        ),
    },
    "european_integration": {
        "triggers": {
            "europe", "european", "eu", "union", "integration", "lisbon",
            "turkey", "enlargement", "brussels",
        },
        "frame": (
            "For European integration questions, distinguish support for the EU, reform of EU institutions, "
            "enlargement, sovereignty transfer, treaty politics, and practical policy cooperation."
        ),
    },
    "economy": {
        "triggers": {
            "economy", "economic", "economie", "emploi", "jobs", "wages",
            "industry", "growth", "tax", "market", "exports", "innovation",
        },
        "frame": (
            "For economic questions, distinguish goals, instruments, beneficiaries, fiscal measures, "
            "labor-market commitments, industrial policy, growth model, and distributional effects."
        ),
    },
    "migration": {
        "triggers": {
            "migration", "immigration", "asylum", "refugees", "diaspora",
            "migrants", "integration",
        },
        "frame": (
            "For migration questions, distinguish border control, asylum rights, integration policy, "
            "labor migration, citizenship, and humanitarian commitments."
        ),
    },
    "environment": {
        "triggers": {
            "environment", "ecology", "ecological", "climate", "energy",
            "sustainability", "sustainable", "renewable",
        },
        "frame": (
            "For environment questions, distinguish climate goals, energy transition, regulation, "
            "industrial adaptation, agriculture, infrastructure, and intergenerational responsibility."
        ),
    },
}


@dataclass(frozen=True)
class ChatRequest:
    """A user question scoped to one country memory and optional party."""

    question: str
    as_of: date
    party_id: str | None = None
    k: int = 8
    include_unverified: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    routing_mode: str = "deterministic"
    previous_citations: list[dict[str, Any]] = field(default_factory=list)


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
    retrieval_reason: str | None = None


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
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    route: str = "evidence_query"
    prompt_citation_count: int = 0


class AnswerContractError(RuntimeError):
    """Raised when an LLM answer violates COMPASS citation discipline."""


class ChatEngine:
    """Conversational RAG facade over a ``CountryMemory`` instance."""

    def __init__(self, memory: Any, model_name: str | None = None) -> None:
        self.memory = memory
        self.model_name = model_name or _default_chat_model()

    def _answer_segment_lookup(self, question: str, segment_ids: list[str]) -> ChatResponse:
        metadata_warning = ""
        if hasattr(self.memory, "fetch_records_by_ids"):
            records = self.memory.fetch_records_by_ids(segment_ids)
        else:
            texts = self.memory.fetch_by_ids(segment_ids)
            records = [
                {"segment_id": segment_id, "text": text, "meta": {}}
                for segment_id, text in texts.items()
            ]
            metadata_warning = (
                "\n\nAvertissement technique : les métadonnées sont absentes pour ce segment. "
                "Vérifiez l'index ou réindexez le corpus."
            )
        citations = build_citations(records)
        if not citations:
            return ChatResponse(
                answer="Je n'ai pas trouve ce segment exact dans l'index COMPASS.",
                citations=[],
                model_used=None,
                llm_used=False,
                retrieval_count=0,
                route="direct_lookup",
            )
        answer_lines = ["Voici le passage exact demande dans l'index COMPASS :", ""]
        for citation in citations:
            answer_lines.append(f"[{citation.ref_id}] `{citation.segment_id}`")
            answer_lines.append(_source_label(citation))
            answer_lines.append(citation.text)
            answer_lines.append("")
        return ChatResponse(
            answer="\n".join(answer_lines).strip() + metadata_warning,
            citations=citations,
            model_used=None,
            llm_used=False,
            retrieval_count=len(citations),
            route="direct_lookup",
            prompt_citation_count=len(citations),
        )

    def _answer_corpus_scope(self, request: ChatRequest, scope: dict[str, Any]) -> ChatResponse:
        """Describe the active corpus without retrieval or LLM generation."""
        country = scope.get("country_iso3") or "non renseigné"
        parties = _format_parties(scope.get("parties") or [])
        dates = _format_document_dates(scope.get("document_dates") or [])
        doc_types = ", ".join(scope.get("document_types") or []) or "non renseigné"
        answer = (
            "Je suis connecté à la mémoire documentaire COMPASS active pour cette session.\n\n"
            f"- Pays : {country}\n"
            f"- Partis indexés : {parties}\n"
            f"- Documents distincts : {scope.get('n_documents', 0)}\n"
            f"- Types de documents : {doc_types}\n"
            f"- Dates des documents : {dates}\n"
            f"- Borne temporelle as_of : {request.as_of.isoformat()}\n"
            "- Stockage : index vectoriel ChromaDB local\n\n"
            "La date as_of est une borne temporelle de requête. Elle n'est pas nécessairement "
            "la date officielle du scrutin.\n\n"
            "Je réponds uniquement à partir des documents déjà ingérés dans cet index ; "
            "je n'interroge pas Internet pendant la conversation."
        )
        return ChatResponse(
            answer=answer,
            citations=[],
            model_used=None,
            llm_used=False,
            retrieval_count=0,
            route="corpus_scope",
        )

    def _answer_scope_limit(
        self,
        request: ChatRequest,
        scope: dict[str, Any],
        route: str,
    ) -> ChatResponse:
        country = scope.get("country_iso3") or "un pays non renseigné"
        parties = _format_parties(scope.get("parties") or [])
        n_documents = scope.get("n_documents", 0)
        missing = {
            "OUT_OF_CORPUS": "un corpus exhaustif couvrant tous les acteurs concernés",
            "COMPARISON_NEEDS_MORE_CORPUS": "les manifestes des autres partis à comparer",
            "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA": "une base structurée de résultats et de contexte électoral",
        }[route]
        answer = (
            f"Le corpus actif contient {n_documents} document(s) distinct(s) pour {country}, "
            f"parti(s) : {parties}. Il permet d'analyser les positions documentées dans ces textes, "
            "mais pas de répondre de façon exhaustive à toute la question. "
            f"Pour compléter l'analyse, il faudrait indexer {missing}."
        )
        return ChatResponse(
            answer=answer,
            citations=[],
            model_used=None,
            llm_used=False,
            retrieval_count=0,
            route=route,
        )

    def _answer_follow_up_sources(self, request: ChatRequest) -> ChatResponse:
        citations = citations_from_payload(request.previous_citations)
        if not citations:
            answer = "Aucune source structurée n'est disponible pour la réponse précédente dans cette session."
        else:
            answer = "Voici les preuves utilisées dans la réponse précédente :\n\n" + format_citations(citations)
        return ChatResponse(
            answer=answer,
            citations=citations,
            model_used=None,
            llm_used=False,
            retrieval_count=0,
            route="FOLLOW_UP_SOURCES",
            prompt_citation_count=len(citations),
        )

    def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer a question using existing indexed COMPASS documents."""
        question = request.question.strip()
        if not question:
            raise ValueError("Question vide.")
        scope = describe_active_corpus(self.memory, request)
        route = route_chat_question(
            question,
            mode=request.routing_mode,
            model_name=self.model_name,
            party_entities=_party_entity_labels(scope),
        )
        if route == "direct_lookup":
            segment_ids = extract_segment_ids(question)
            return self._answer_segment_lookup(question, segment_ids)
        if route == "corpus_scope":
            return self._answer_corpus_scope(request, scope)
        if route == "FOLLOW_UP_SOURCES":
            return self._answer_follow_up_sources(request)
        if route in {
            "OUT_OF_CORPUS",
            "COMPARISON_NEEDS_MORE_CORPUS",
            "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA",
        }:
            return self._answer_scope_limit(request, scope, route)
        retrieved = query_evidence(
            self.memory,
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
                route="evidence_query",
            )
        retrieved = attach_parent_context(self.memory, retrieved)
        all_citations = build_citations(retrieved)
        citations = all_citations[:settings.chat_max_prompt_citations]
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
            validate_llm_answer(answer, citations, route=route)
            if settings.chat_semantic_validation_enabled:
                validate_semantic_grounding(answer, citations)
            return ChatResponse(
                answer=answer,
                citations=citations,
                model_used=self.model_name,
                llm_used=True,
                retrieval_count=len(retrieved),
                general_context=general_context,
                prompt_messages=messages,
                route=route,
                prompt_citation_count=len(citations),
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
                prompt_messages=messages,
                route=route,
                prompt_citation_count=len(citations),
            )


def query_evidence(
    memory: Any,
    question: str,
    *,
    as_of: date,
    k: int,
    party_id: str | None,
    include_unverified: bool,
) -> list[dict[str, Any]]:
    """Use the production retrieval stack: dense + BM25 when available."""
    if hasattr(memory, "query_documents_hybrid"):
        return memory.query_documents_hybrid(
            question,
            as_of=as_of,
            k=k,
            party_id=party_id,
            include_unverified=include_unverified,
        )
    return memory.query_documents(
        question,
        as_of=as_of,
        k=k,
        party_id=party_id,
        include_unverified=include_unverified,
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
                retrieval_reason=_none_if_empty(item.get("retrieval_reason")),
            )
        )
    return citations


def citation_to_payload(citation: Citation) -> dict[str, Any]:
    """Serialize a citation for browser/session state without parsing prose."""
    return {
        "ref_id": citation.ref_id,
        "segment_id": citation.segment_id,
        "text": citation.text,
        "doc_id": citation.doc_id,
        "country_iso3": citation.country_iso3,
        "party_id": citation.party_id,
        "doc_date": citation.doc_date,
        "doc_type": citation.doc_type,
        "reliability": citation.reliability,
        "parent_text": citation.parent_text,
        "retrieval_reason": citation.retrieval_reason,
    }


def citations_from_payload(items: list[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    for item in items:
        try:
            citations.append(Citation(
                ref_id=str(item.get("ref_id") or f"S{len(citations) + 1}"),
                segment_id=str(item.get("segment_id") or ""),
                text=str(item.get("text") or ""),
                doc_id=str(item.get("doc_id") or ""),
                country_iso3=str(item.get("country_iso3") or ""),
                party_id=_none_if_empty(item.get("party_id")),
                doc_date=_none_if_empty(item.get("doc_date")),
                doc_type=_none_if_empty(item.get("doc_type")),
                reliability=_none_if_empty(item.get("reliability")),
                parent_text=_none_if_empty(item.get("parent_text")),
                retrieval_reason=_none_if_empty(item.get("retrieval_reason")),
            ))
        except (AttributeError, TypeError, ValueError):
            continue
    return citations


def describe_active_corpus(memory: Any, request: ChatRequest) -> dict[str, Any]:
    """Read the active corpus profile from memory, never from demo constants."""
    scope: dict[str, Any] = {}
    if hasattr(memory, "describe_corpus"):
        try:
            scope = memory.describe_corpus(as_of=request.as_of, party_id=request.party_id) or {}
        except TypeError:
            scope = memory.describe_corpus() or {}
        except Exception:
            scope = {}
    return {
        "country_iso3": str(
            scope.get("country_iso3") or getattr(memory, "country", "") or ""
        ).upper(),
        "n_documents": int(scope.get("n_documents") or 0),
        "parties": list(scope.get("parties") or []),
        "document_dates": sorted({str(value) for value in scope.get("document_dates") or [] if value}),
        "document_types": sorted({str(value) for value in scope.get("document_types") or [] if value}),
    }


def _party_entity_labels(scope: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for party in scope.get("parties") or []:
        if isinstance(party, dict):
            labels.extend(str(party.get(key) or "") for key in ("party_id", "name"))
        else:
            labels.append(str(party))
    return [label for label in labels if label]


def _format_parties(parties: list[Any]) -> str:
    labels: list[str] = []
    for party in parties:
        if isinstance(party, dict):
            party_id = str(party.get("party_id") or "").strip()
            name = str(party.get("name") or "").strip()
            if party_id and name:
                labels.append(f"{party_id} ({name})")
            elif party_id or name:
                labels.append(party_id or name)
        elif str(party).strip():
            labels.append(str(party).strip())
    return ", ".join(labels) if labels else "non renseigné dans les métadonnées"


def _format_document_dates(document_dates: list[str]) -> str:
    if not document_dates:
        return "non renseignées"
    if len(document_dates) == 1:
        return document_dates[0]
    return f"du {document_dates[0]} au {document_dates[-1]}"


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
    prompt_citations = citations[:settings.chat_max_prompt_citations]
    evidence_context = "\n\n".join(
        f"[{citation.ref_id}] "
        f"country={citation.country_iso3 or 'unknown'} party={citation.party_id or 'unknown'} "
        f"date={citation.doc_date or 'unknown'} type={citation.doc_type or 'unknown'} "
        f"reliability={citation.reliability or 'unknown'}\n"
        f"retrieval_reason={citation.retrieval_reason or 'not available'}\n"
        f"local_parent_context={_compact_parent_context(citation)}\n"
        f"{_excerpt(citation.text, max_chars=settings.chat_max_evidence_text_chars)}"
        for citation in prompt_citations
    )
    general_context_text = format_general_context_for_prompt(general_context or [])
    analytical_context_text = build_analytical_context(question)
    language = infer_answer_language(question)
    system = (
        "You are COMPASS Chat, a research assistant for political manifesto analysis. "
        "Your task is evidence-grounded analysis, not open-ended political commentary. "
        "Use only the material in this prompt. Do not use outside knowledge, memory, assumptions, or likely facts. "
        "There are three inputs: ANALYTICAL_CONTEXT, GENERAL_CONTEXT, and CITED_EVIDENCE. "
        "ANALYTICAL_CONTEXT gives a political-science reading frame; it is not factual evidence and must never be cited. "
        "GENERAL_CONTEXT gives document-level orientation only; it is not proof and must never be cited. "
        "CITED_EVIDENCE contains the only passages that may support claims. "
        "Every substantive political claim must end with an inline citation like [S1] or [S2]. "
        "Never cite [A], [C1], [C2], or any analytical/general-context label. "
        "If the cited evidence does not answer the question, say that the provided evidence is insufficient and explain what is missing. "
        "If a user premise is not supported by the cited evidence, correct it cautiously instead of accepting it. "
        "Do not treat absence of evidence as evidence of absence. If the cited evidence mentions only one actor, "
        "party, policy, country, or event, you may state that only this item appears in the provided evidence. "
        "You must not conclude that no other items existed unless the evidence explicitly states the list is exhaustive. "
        "Do not add a separate bibliography or sources section; the interface adds sources separately. "
        "Do not infer motives, psychology, or hidden positions beyond the passages. "
        "Do not overinterpret; if a conclusion is not directly supported, phrase it cautiously or say the evidence is insufficient. "
        "Prefer short evidence-linked sentences over broad summaries. "
        "Use only source ids shown in CITED_EVIDENCE; if a claim lacks an [S] source, omit it. "
        "Before answering, internally verify that each sentence with a political claim has at least one [S] citation. "
        f"Answer in {language}."
    )
    scope = f"as_of={request.as_of.isoformat()}"
    if request.party_id:
        scope += f", party_id={request.party_id}"
    user = (
        f"Scope: {scope}\n\n"
        f"Question: {question}\n\n"
        "ANALYTICAL_CONTEXT - conceptual reading frame, never cite this block:\n"
        f"{analytical_context_text}\n\n"
        "GENERAL_CONTEXT - background only, never cite this block:\n"
        f"{general_context_text}\n\n"
        "CITED_EVIDENCE - the only claim-supporting evidence:\n"
        f"{evidence_context}\n\n"
        "Answer contract:\n"
        "- Answer the question directly.\n"
        "- Use [S1], [S2], etc. for every claim.\n"
        "- Never invent source ids. Use only the [S] ids present above.\n"
        "- Use ANALYTICAL_CONTEXT only to understand the political concept being asked about.\n"
        "- Do not cite [A], [C1], or other analytical/general-context labels.\n"
        "- If no [S] passage supports the answer, say evidence is insufficient.\n"
        "- If evidence is weak or absent, say so explicitly."
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
        if hasattr(memory, "query_documents_hybrid"):
            records = memory.query_documents_hybrid(
                build_general_context_query(question),
                as_of=request.as_of,
                k=_MAX_GENERAL_CONTEXT_ITEMS,
                party_id=request.party_id,
                include_unverified=request.include_unverified,
                include_parent_segments=True,
            )
        else:
            records = memory.query_documents(
                build_general_context_query(question),
                as_of=request.as_of,
                k=_MAX_GENERAL_CONTEXT_ITEMS,
                party_id=request.party_id,
                include_unverified=request.include_unverified,
                include_parent_segments=True,
            )
    except TypeError:
        try:
            records = memory.query_documents(
                build_general_context_query(question),
                as_of=request.as_of,
                k=_MAX_GENERAL_CONTEXT_ITEMS,
                party_id=request.party_id,
                include_unverified=request.include_unverified,
                include_parent_segments=True,
            )
        except Exception:
            return build_general_context_items(evidence_records, limit=2)
    except Exception:
        return build_general_context_items(evidence_records, limit=2)
    return build_general_context_items(records, limit=_MAX_GENERAL_CONTEXT_ITEMS)


def build_general_context_query(question: str) -> str:
    return (
        f"{question} manifesto overall political program party priorities "
        "general orientation policy agenda election platform"
    )


def build_analytical_context(question: str) -> str:
    """Return a compact non-cited political-science frame for the prompt.

    This is not factual knowledge about the party or election. It tells the LLM
    what analytical dimensions to look for in the cited evidence, which reduces
    overinterpretation without making a second LLM call.
    """
    query_tokens = set(re.findall(r"[\w']+", question.lower()))
    lines = [
        "[A] Analytical frame, not evidence:",
        "- Interpret the political text under the given date and party scope.",
        "- Separate declared position, policy instrument, target group, value, and institutional implication.",
        "- Distinguish explicit claims from indirect inference.",
        "- Do not import external facts, party reputation, or theory as evidence.",
    ]
    matched = [
        lens["frame"]
        for lens in _ANALYTICAL_LENSES.values()
        if query_tokens & lens["triggers"]
    ]
    if matched:
        lines.append("- Question-specific lens: " + " ".join(matched[:2]))
    else:
        lines.append(
            "- Question-specific lens: identify the political concept in the question, "
            "then look for explicit claims, policy tools, beneficiaries, and limits in the cited passages."
        )
    return _excerpt("\n".join(lines), max_chars=_MAX_ANALYTICAL_CONTEXT_CHARS)


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


def route_chat_question(
    question: str,
    mode: str = "deterministic",
    model_name: str | None = None,
    party_entities: list[str] | None = None,
) -> str:
    """Route a question using the selected mode, with deterministic fallback."""
    normalized_mode = str(mode or "deterministic").strip().lower()
    if normalized_mode not in _ROUTING_MODES:
        raise ValueError(f"Unsupported routing mode: {mode}")
    if normalized_mode == "llm":
        return _route_chat_question_llm(
            question,
            model_name or _default_chat_model(),
            party_entities=party_entities,
        )
    return _route_chat_question_deterministic(question, party_entities=party_entities)


def _route_chat_question_deterministic(
    question: str,
    party_entities: list[str] | None = None,
) -> str:
    if extract_segment_ids(question):
        return "direct_lookup"
    normalized = _normalize_intent_text(question)
    if _is_source_followup_intent(normalized):
        return "FOLLOW_UP_SOURCES"
    corpus_patterns = (
        "quel corpus",
        "a quel corpus",
        "connecte a quel corpus",
        "connecte au corpus",
        "quelle base de donnees",
        "quelles donnees utilises tu",
        "quelles donnees utilise le systeme",
        "what corpus",
        "which corpus",
        "which dataset",
        "what data are you connected to",
    )
    if any(pattern in normalized for pattern in corpus_patterns):
        return "corpus_scope"
    if _is_cross_party_comparison(question, normalized, party_entities or []):
        return "COMPARISON_NEEDS_MORE_CORPUS"
    election_context_patterns = (
        "resultats electoraux", "resultat electoral", "qui a gagne",
        "qui gouvernait", "nombre de sieges", "combien de sieges",
        "electoral results", "who won", "who governed", "seat count", "turnout",
    )
    if any(pattern in normalized for pattern in election_context_patterns):
        return "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA"
    exhaustive_patterns = (
        "tous les partis", "toutes les parties", "liste complete des",
        "liste exhaustive", "all the parties", "all parties", "complete list of",
    )
    if any(pattern in normalized for pattern in exhaustive_patterns):
        return "OUT_OF_CORPUS"
    return "evidence_query"


def _route_chat_question_llm(
    question: str,
    model_name: str,
    party_entities: list[str] | None = None,
) -> str:
    """Ask the configured LLM for one route label; fail back deterministically."""
    fallback = _route_chat_question_deterministic(question, party_entities=party_entities)
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user request into exactly one COMPASS route. "
                "Return only one label and no explanation: direct_lookup, corpus_scope, evidence_query, "
                "FOLLOW_UP_SOURCES, OUT_OF_CORPUS, COMPARISON_NEEDS_MORE_CORPUS, or "
                "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA. "
                "direct_lookup means the user requests an explicit segment id. "
                "corpus_scope means the user asks which corpus, dataset, country, party, date scope, or storage is active. "
                "FOLLOW_UP_SOURCES asks for sources used in the previous answer. "
                "OUT_OF_CORPUS requests an exhaustive list not established by the active documentary corpus. "
                "COMPARISON_NEEDS_MORE_CORPUS compares at least two named political parties; comparing two themes is evidence_query. "
                "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA asks who governed, who won, turnout, seats, or election results. "
                "evidence_query means the user asks about political content or evidence in indexed documents."
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        raw_decision = complete_chat(
            model_name,
            messages,
            temperature=0.0,
            max_tokens=12,
        ).strip()
    except Exception:
        return fallback
    decision = next(
        (route for route in _ROUTE_VALIDATION_POLICIES if route.lower() == raw_decision.lower()),
        raw_decision,
    )
    if decision in _ROUTE_VALIDATION_POLICIES:
        return decision
    return fallback


def _is_source_followup_intent(normalized: str) -> bool:
    markers = (
        "quelles sont tes sources", "quelles sont les sources", "sources exactes",
        "passages cites", "preuves utilisees", "what are your sources",
        "exact sources", "cited passages", "evidence used",
    )
    if len(normalized) > 120:
        return False
    if any(marker in normalized for marker in markers):
        return True
    asks_for_sources = "source" in normalized or "preuve" in normalized
    followup_wording = any(word in normalized for word in ("quelles", "exact", "your answer", "ta reponse"))
    return asks_for_sources and followup_wording


def _is_cross_party_comparison(
    question: str,
    normalized: str,
    party_entities: list[str],
) -> bool:
    comparison_markers = ("compare", "comparer", "comparaison", "versus", " vs ")
    if not any(marker in f" {normalized} " for marker in comparison_markers):
        return False
    relative_markers = (
        "par rapport aux autres partis", "avec les autres partis",
        "compared with other parties", "versus other parties",
    )
    if any(marker in normalized for marker in relative_markers):
        return True

    found: set[str] = set()
    for entity in party_entities:
        normalized_entity = _normalize_intent_text(entity)
        if normalized_entity and normalized_entity in normalized:
            found.add(normalized_entity)
    for acronym in re.findall(r"\b[A-Z][A-Z0-9.-]{1,11}\b", question):
        found.add(acronym.lower())
    for explicit in re.findall(
        r"\b(?:parti|party)\s+([A-Za-z0-9][A-Za-z0-9_.-]{1,30})",
        question,
        flags=re.IGNORECASE,
    ):
        found.add(explicit.lower())
    return len(found) >= 2


def _normalize_intent_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().replace("'", " ").split())


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


def validate_llm_answer(
    answer: str,
    citations: list[Citation],
    route: str = "evidence_query",
) -> None:
    """Reject LLM answers that escape the evidence contract.

    The prompt is necessary but not sufficient. This deterministic validator
    prevents the UI from displaying answers that cite analytical/general
    context, invent source ids, or make a substantive answer with no cited
    evidence.
    """
    text = answer.strip()
    if not text:
        raise AnswerContractError("empty answer")

    policy = validation_policy_for_route(route)
    if policy == "none":
        return
    if policy != "strict_evidence":
        raise AnswerContractError(f"unknown validation policy: {policy}")

    refs = _ANSWER_REF_RE.findall(text)
    forbidden = [ref for ref in refs if ref == "A" or ref.startswith("C")]
    if forbidden:
        raise AnswerContractError("answer cited analytical/general context")

    valid_source_ids = {citation.ref_id for citation in citations[:settings.chat_max_prompt_citations]}
    source_refs = [ref for ref in refs if ref.startswith("S")]
    unknown = sorted({ref for ref in source_refs if ref not in valid_source_ids})
    if unknown:
        raise AnswerContractError(f"answer cited unknown source ids: {', '.join(unknown)}")

    if not source_refs and not _is_insufficiency_answer(text):
        raise AnswerContractError("substantive answer without cited evidence")


def _is_insufficiency_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _INSUFFICIENT_MARKERS)


def validate_semantic_grounding(answer: str, citations: list[Citation]) -> None:
    """Optionally verify cited claim sentences with the shared NLI model.

    This guard is deliberately opt-in: generic multilingual NLI can reject
    valid analytical paraphrases. When enabled, a failed implication raises
    ``AnswerContractError`` and the normal extractive fallback takes over.
    """
    from compass.nlp_models import nli_pipeline

    by_ref = {citation.ref_id: citation for citation in citations}
    classifier = nli_pipeline()
    claim_sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    checked = 0
    for sentence in claim_sentences:
        refs = [ref for ref in _ANSWER_REF_RE.findall(sentence) if ref.startswith("S")]
        if not refs:
            continue
        claim = _ANSWER_REF_RE.sub("", sentence).strip()
        if not claim:
            continue
        checked += 1
        supported = False
        for ref in refs:
            citation = by_ref.get(ref)
            if citation is None:
                continue
            result = classifier({"text": citation.text, "text_pair": claim})
            if isinstance(result, list):
                result = result[0] if result else {}
            label = str((result or {}).get("label") or "").lower()
            score = float((result or {}).get("score") or 0.0)
            if "entail" in label and score >= settings.chat_nli_entailment_threshold:
                supported = True
                break
        if not supported:
            raise AnswerContractError("semantic grounding check failed")
    if not checked and not _is_insufficiency_answer(answer):
        raise AnswerContractError("semantic grounding found no cited claim")


def validation_policy_for_route(route: str) -> str:
    """Return the answer-validation contract attached to a chat route."""
    if route not in _ROUTE_VALIDATION_POLICIES:
        raise AnswerContractError(f"unknown chat route: {route}")
    return _ROUTE_VALIDATION_POLICIES[route]


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
