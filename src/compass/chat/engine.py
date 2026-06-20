"""RAG-style chat engine over existing COMPASS memories.

This component is intentionally thin: it does not replace C01-C15. It sits on
top of ``CountryMemory.query_documents`` and the existing OpenAI-compatible LLM
client so the chat interface can be added without changing ingestion, retrieval,
reasoning, validation, or schemas.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from compass.config import settings
from compass.chat.query_analysis import (
    QueryAnalysis,
    analyze_question,
)
from compass.chat.expert_retrieval import (
    RetrievalBundle,
    record_in_scope,
    retrieve_expert,
)
from compass.llm_client import complete_chat

_SEGMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*:p\d{3}(?:c\d{3})?")
_SCIENTIFIC_ANALYSIS_RE = re.compile(
    r"^\s*/?(?:analyse|analyze|coder|score)\s+([A-Za-z0-9_.-]+)",
    flags=re.IGNORECASE,
)
_SCIENTIFIC_VALIDATION_RE = re.compile(
    r"^\s*/?(?:valider|validate)(?:\s+([A-Za-z0-9_.-]+))?\s*$",
    flags=re.IGNORECASE,
)
_SCIENTIFIC_CONTAMINATION_RE = re.compile(
    r"^\s*/?(?:contamination|probe-contamination)\s+([A-Za-z0-9_.-]+)\s*$",
    flags=re.IGNORECASE,
)
_MAX_RELATIONAL_CONTEXT_ITEMS = 6
_MAX_ANALYTICAL_CONTEXT_CHARS = 420
_MAX_CHAT_HISTORY_MESSAGES = 1
_MAX_CHAT_HISTORY_CHARS = 160
_MAX_CHAT_OUTPUT_TOKENS = 350
_ANSWER_REF_RE = re.compile(r"\[(A|C\d+|R\d+|S\d+)\]")
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
    "SCIENTIFIC_ANALYSIS": "none",
    "SCIENTIFIC_VALIDATION": "none",
    "SCIENTIFIC_VARIABLES": "none",
    "SCIENTIFIC_CONTAMINATION": "none",
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
    election_id: str | None = None
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
    section_title: str | None = None
    parent_text: str | None = None
    retrieval_reason: str | None = None
    evidence_role: str = "primary"


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
    section_title: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    """Answer plus evidence used to produce it."""

    answer: str
    citations: list[Citation]
    model_used: str | None
    llm_used: bool
    retrieval_count: int
    general_context: list[GeneralContext] = field(default_factory=list)
    graph_context: list[dict[str, Any]] = field(default_factory=list)
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    route: str = "evidence_query"
    prompt_citation_count: int = 0
    query_analysis: dict[str, Any] = field(default_factory=dict)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    validation_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PromptBudget:
    evidence_chars: int
    parent_chars: int
    general_chars: int
    graph_chars: int
    target_chars: int


class AnswerContractError(RuntimeError):
    """Raised when an LLM answer violates COMPASS citation discipline."""


class RepairExhaustedError(AnswerContractError):
    def __init__(self, message: str, trace: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class ClaimValidation:
    claim: str
    refs: list[str]
    status: str
    best_label: str | None = None
    best_score: float = 0.0


class ChatEngine:
    """Conversational RAG facade over a ``CountryMemory`` instance."""

    def __init__(
        self,
        memory: Any,
        model_name: str | None = None,
        graph: Any | None = None,
        scientific_service: Any | None = None,
    ) -> None:
        self.memory = memory
        self.model_name = model_name or _default_chat_model()
        self.graph = graph
        self.scientific_service = scientific_service

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
        citations = build_citations(records, enforce_minimum=False)
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

    def _answer_scientific_analysis(
        self,
        request: ChatRequest,
        scope: dict[str, Any],
    ) -> ChatResponse:
        variable_id = extract_scientific_variable(request.question)
        if self.scientific_service is None:
            return _scientific_unavailable_response(
                "Le service scientifique n'est pas configuré pour cette interface.",
                route="SCIENTIFIC_ANALYSIS",
            )
        if not variable_id:
            return _scientific_unavailable_response(
                "Précisez une variable : /analyse <variable_id>.",
                route="SCIENTIFIC_ANALYSIS",
            )
        try:
            result = self.scientific_service.analyze(
                variable_id,
                country_iso3=str(scope.get("country_iso3") or getattr(self.memory, "country", "")),
                party_id=request.party_id,
                election_id=request.election_id,
                as_of=request.as_of,
            )
        except Exception as exc:
            return _scientific_unavailable_response(
                f"Analyse scientifique indisponible : {exc}",
                route="SCIENTIFIC_ANALYSIS",
            )
        citations = citations_from_final_answer(result)
        return ChatResponse(
            answer=format_scientific_answer(
                result,
                citations,
                trace_path=getattr(self.scientific_service, "last_trace_path", None),
            ),
            citations=citations,
            model_used=None,
            llm_used=bool(getattr(result, "inferred", [])),
            retrieval_count=len(result.main_evidence) + len(result.counter_evidence),
            graph_context=list(getattr(result, "graph_context", []) or []),
            route="SCIENTIFIC_ANALYSIS",
            prompt_citation_count=len(citations),
        )

    def _answer_scientific_validation(self, request: ChatRequest) -> ChatResponse:
        if self.scientific_service is None:
            return _scientific_unavailable_response(
                "Le service scientifique n'est pas configuré pour cette interface.",
                route="SCIENTIFIC_VALIDATION",
            )
        variable_id = extract_scientific_variable(request.question)
        try:
            report = self.scientific_service.validate_cached(variable_id)
            answer = format_validation_report(report)
        except Exception as exc:
            answer = f"Validation scientifique indisponible : {exc}"
        return ChatResponse(
            answer=answer,
            citations=[],
            model_used=None,
            llm_used=False,
            retrieval_count=0,
            route="SCIENTIFIC_VALIDATION",
        )

    def _answer_scientific_variables(self) -> ChatResponse:
        if self.scientific_service is None:
            variables: list[str] = []
        else:
            try:
                variables = self.scientific_service.available_variables()
            except Exception:
                variables = []
        answer = (
            "Variables scientifiques disponibles :\n\n- " + "\n- ".join(variables)
            if variables
            else "Aucune variable scientifique n'est disponible dans le registre actif."
        )
        return ChatResponse(
            answer=answer,
            citations=[],
            model_used=None,
            llm_used=False,
            retrieval_count=0,
            route="SCIENTIFIC_VARIABLES",
        )

    def _answer_scientific_contamination(self, request: ChatRequest) -> ChatResponse:
        variable_id = extract_scientific_variable(request.question)
        if self.scientific_service is None or not variable_id:
            return _scientific_unavailable_response(
                "Commande attendue : /contamination <variable_id>.",
                route="SCIENTIFIC_CONTAMINATION",
            )
        try:
            results = self.scientific_service.contamination_check(
                variable_id,
                party_id=request.party_id,
                election_year=request.as_of.year,
            )
            lines = ["Sonde de contamination C15", ""]
            for result in results:
                lines.append(
                    f"- {result['model']} : claims_knowledge={result['claims_knowledge']} "
                    f"| réponse brute={result['raw']}"
                )
            answer = "\n".join(lines)
        except Exception as exc:
            answer = f"Sonde de contamination indisponible : {exc}"
        return ChatResponse(
            answer=answer,
            citations=[],
            model_used=None,
            llm_used=True,
            retrieval_count=0,
            route="SCIENTIFIC_CONTAMINATION",
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
        if route == "SCIENTIFIC_ANALYSIS":
            return self._answer_scientific_analysis(request, scope)
        if route == "SCIENTIFIC_VALIDATION":
            return self._answer_scientific_validation(request)
        if route == "SCIENTIFIC_VARIABLES":
            return self._answer_scientific_variables()
        if route == "SCIENTIFIC_CONTAMINATION":
            return self._answer_scientific_contamination(request)
        if route in {
            "OUT_OF_CORPUS",
            "COMPARISON_NEEDS_MORE_CORPUS",
            "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA",
        }:
            return self._answer_scope_limit(request, scope, route)
        question_analysis = analyze_question(
            question,
            scope=scope,
            model_name=self.model_name,
            complete=complete_chat,
        )
        retrieval = retrieve_expert(
            self.memory,
            question_analysis,
            query_function=query_evidence,
            as_of=request.as_of,
            country_iso3=str(scope.get("country_iso3") or ""),
            k=request.k,
            party_id=request.party_id,
            election_id=request.election_id,
            include_unverified=request.include_unverified,
        )
        prompt_records = retrieval.prompt_records(settings.chat_max_prompt_citations)
        prompt_records = attach_parent_context(self.memory, prompt_records)
        citations = build_citations(prompt_records)
        if not citations:
            return ChatResponse(
                answer="Aucun passage pertinent n'a ete trouve dans la memoire COMPASS pour cette question.",
                citations=[],
                model_used=None,
                llm_used=False,
                retrieval_count=retrieval.total_candidates,
                route="evidence_query",
                query_analysis=question_analysis.model_dump(),
                retrieval_trace=retrieval.trace,
            )
        if retrieval.sufficiency < settings.chat_retrieval_min_sufficiency:
            return ChatResponse(
                answer=(
                    "Les preuves récupérées sont insuffisantes pour produire une réponse "
                    "politique fiable dans le périmètre actif."
                ),
                citations=citations,
                model_used=None,
                llm_used=False,
                retrieval_count=retrieval.total_candidates,
                route="evidence_query",
                prompt_citation_count=len(citations),
                query_analysis=question_analysis.model_dump(),
                retrieval_trace=retrieval.trace,
            )
        general_context = retrieve_general_context(
            self.memory,
            question,
            request,
            prompt_records,
            question_analysis,
        )
        graph_context = retrieve_graph_context(self.graph, question, request, scope)
        messages = build_messages(
            question,
            citations,
            request,
            general_context,
            graph_context,
            question_analysis,
            retrieval,
        )
        try:
            answer, validation_trace, final_messages = generate_validated_answer(
                model_name=self.model_name,
                messages=messages,
                citations=citations,
                route=route,
                complete=complete_chat,
            )
            return ChatResponse(
                answer=answer,
                citations=citations,
                model_used=self.model_name,
                llm_used=True,
                retrieval_count=retrieval.total_candidates,
                general_context=general_context,
                graph_context=graph_context,
                prompt_messages=final_messages,
                route=route,
                prompt_citation_count=len(citations),
                query_analysis=question_analysis.model_dump(),
                retrieval_trace=retrieval.trace,
                validation_trace=validation_trace,
            )
        except Exception as exc:
            fallback = build_extractive_answer(question, citations, exc)
            validation_trace = list(getattr(exc, "trace", []))
            return ChatResponse(
                answer=fallback,
                citations=citations,
                model_used=self.model_name,
                llm_used=False,
                retrieval_count=retrieval.total_candidates,
                general_context=general_context,
                graph_context=graph_context,
                prompt_messages=messages,
                route=route,
                prompt_citation_count=len(citations),
                query_analysis=question_analysis.model_dump(),
                retrieval_trace=retrieval.trace,
                validation_trace=validation_trace,
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


def build_citations(
    retrieved: list[dict[str, Any]],
    *,
    enforce_minimum: bool = True,
) -> list[Citation]:
    child_parent_ids = {
        str((item.get("meta") or {}).get("parent_segment_id") or "")
        for item in retrieved
        if (item.get("meta") or {}).get("parent_segment_id")
    }
    child_parent_fingerprints = {
        _text_fingerprint(str(item.get("parent_text") or ""))
        for item in retrieved
        if item.get("parent_text")
    }
    citations: list[Citation] = []
    seen_segments: set[str] = set()
    seen_texts: set[str] = set()
    for item in retrieved:
        meta = item.get("meta") or {}
        segment_id = str(item.get("segment_id") or "")
        text = str(item.get("text") or "").strip()
        parent_text = str(item.get("parent_text") or "").strip()
        text_fingerprint = _text_fingerprint(text)
        parent_id = str(meta.get("parent_segment_id") or "")
        if not segment_id or not text_fingerprint:
            continue
        if segment_id in seen_segments or text_fingerprint in seen_texts:
            continue
        if (
            not parent_id
            and (
                segment_id in child_parent_ids
                or text_fingerprint in child_parent_fingerprints
            )
        ):
            continue
        if enforce_minimum and (
            _word_count(text) < settings.chat_min_citable_words
            and _word_count(parent_text) < settings.chat_min_citable_words
        ):
            continue
        seen_segments.add(segment_id)
        seen_texts.add(text_fingerprint)
        citations.append(
            Citation(
                ref_id=f"S{len(citations) + 1}",
                segment_id=segment_id,
                text=text,
                doc_id=str(meta.get("doc_id") or ""),
                country_iso3=str(meta.get("country_iso3") or ""),
                party_id=_none_if_empty(meta.get("party_id")),
                doc_date=_none_if_empty(meta.get("doc_date")),
                doc_type=_none_if_empty(meta.get("doc_type")),
                section_title=_none_if_empty(meta.get("section_title")),
                reliability=_none_if_empty(meta.get("reliability")),
                parent_text=_none_if_empty(parent_text),
                retrieval_reason=_none_if_empty(item.get("retrieval_reason")),
                evidence_role=str(item.get("evidence_role") or "primary"),
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
        "section_title": citation.section_title,
        "parent_text": citation.parent_text,
        "retrieval_reason": citation.retrieval_reason,
        "evidence_role": citation.evidence_role,
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
                section_title=_none_if_empty(item.get("section_title")),
                parent_text=_none_if_empty(item.get("parent_text")),
                retrieval_reason=_none_if_empty(item.get("retrieval_reason")),
                evidence_role=str(item.get("evidence_role") or "primary"),
            ))
        except (AttributeError, TypeError, ValueError):
            continue
    return citations


def citations_from_final_answer(answer: Any) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    evidence = list(answer.main_evidence) + list(answer.counter_evidence)
    for item in evidence:
        segment = item.segment
        if segment.segment_id in seen:
            continue
        seen.add(segment.segment_id)
        meta = segment.meta
        reliability = getattr(meta.reliability, "value", meta.reliability)
        citations.append(Citation(
            ref_id=f"S{len(citations) + 1}",
            segment_id=segment.segment_id,
            text=segment.text,
            doc_id=segment.doc_id,
            country_iso3=meta.country_iso3,
            party_id=meta.party_id,
            doc_date=meta.doc_date.isoformat(),
            doc_type=meta.doc_type,
            reliability=str(reliability),
            parent_text=segment.parent_text,
            retrieval_reason=f"scientific_pipeline:{item.qualification_method}",
        ))
    return citations


def format_scientific_answer(
    answer: Any,
    citations: list[Citation],
    trace_path: str | None = None,
) -> str:
    by_segment = {citation.segment_id: citation.ref_id for citation in citations}
    lines = [
        "Analyse scientifique COMPASS",
        "",
        f"- Variable : {answer.variable_id}",
        f"- Statut : {'abstention' if answer.abstained else 'résultat produit'}",
        f"- Score : {answer.score if answer.score is not None else 'non attribué'}",
        f"- Confiance : {answer.confidence if answer.confidence is not None else 'non calculée'}",
        f"- Attribution NLI vérifiée : {'oui' if answer.attribution_checked else 'non'}",
        f"- Incertitude résiduelle : {answer.residual_uncertainty or 'non renseignée'}",
    ]
    if trace_path:
        lines.append(f"- Trace C15 : {trace_path}")
    if answer.main_evidence:
        lines.extend(["", "Preuves principales :"])
        for item in answer.main_evidence[:6]:
            ref = by_segment.get(item.segment.segment_id, "S?")
            lines.append(f"- [{ref}] {_excerpt(item.segment.text, max_chars=300)}")
    if answer.counter_evidence:
        lines.extend(["", "Contre-preuves :"])
        for item in answer.counter_evidence[:4]:
            ref = by_segment.get(item.segment.segment_id, "S?")
            lines.append(f"- [{ref}] {_excerpt(item.segment.text, max_chars=300)}")
    return "\n".join(lines)


def format_validation_report(report: Any) -> str:
    return (
        "Validation externe C14\n\n"
        f"- Strate : {report.stratum}\n"
        f"- Cas évalués : {report.n_cases}\n"
        f"- Abstentions : {report.n_abstentions}\n"
        f"- MAE : {report.mae:.4f}\n"
        f"- Corrélation de Spearman : {report.spearman:.4f}\n"
        f"- Couverture des intervalles : {report.interval_coverage:.4f}\n"
        f"- ECE : {report.ece:.4f}\n"
        f"- Taux d'attribution vérifiée : {report.attribution_rate:.4f}"
    )


def _scientific_unavailable_response(message: str, route: str) -> ChatResponse:
    return ChatResponse(
        answer=message,
        citations=[],
        model_used=None,
        llm_used=False,
        retrieval_count=0,
        route=route,
    )


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
    seen: set[str] = set()
    seen_texts: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in records:
        segment_id = str(item.get("segment_id") or "")
        text = str(item.get("parent_text") or item.get("text") or "")
        fingerprint = _text_fingerprint(text)
        if (
            not segment_id
            or not fingerprint
            or segment_id in seen
            or fingerprint in seen_texts
        ):
            continue
        seen.add(segment_id)
        seen_texts.add(fingerprint)
        unique.append(item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_sections: set[str] = set()
    for item in unique:
        section = str((item.get("meta") or {}).get("section_title") or "").casefold()
        if not section or section in seen_sections:
            continue
        seen_sections.add(section)
        selected.append(item)
        selected_ids.add(str(item.get("segment_id")))
        if len(selected) >= limit:
            break
    for item in unique:
        if len(selected) >= limit:
            break
        if str(item.get("segment_id")) not in selected_ids:
            selected.append(item)
            selected_ids.add(str(item.get("segment_id")))

    return [
        _general_context_from_record(item, index)
        for index, item in enumerate(selected, start=1)
    ]


def _general_context_from_record(item: dict[str, Any], index: int) -> GeneralContext:
    meta = item.get("meta") or {}
    parent_id = str(meta.get("parent_segment_id") or "")
    return GeneralContext(
        ref_id=f"C{index}",
        segment_id=parent_id or str(item.get("segment_id") or ""),
        text=str(item.get("parent_text") or item.get("text") or ""),
        country_iso3=str(meta.get("country_iso3") or ""),
        party_id=_none_if_empty(meta.get("party_id")),
        doc_date=_none_if_empty(meta.get("doc_date")),
        doc_type=_none_if_empty(meta.get("doc_type")),
        section_title=_none_if_empty(meta.get("section_title")),
    )


def build_messages(
    question: str,
    citations: list[Citation],
    request: ChatRequest,
    general_context: list[GeneralContext] | None = None,
    graph_context: list[dict[str, Any]] | None = None,
    question_analysis: QueryAnalysis | None = None,
    retrieval: RetrievalBundle | None = None,
) -> list[dict[str, str]]:
    prompt_citations = citations[:settings.chat_max_prompt_citations]
    general_context = _exclude_evidence_from_general_context(
        general_context or [], prompt_citations,
    )
    budget = compute_prompt_budget(
        len(prompt_citations),
        len(general_context),
        len(graph_context or []),
    )
    evidence_context = format_evidence_for_prompt(prompt_citations, budget)
    general_context_text = format_general_context_for_prompt(
        general_context, max_chars=budget.general_chars,
    )
    relational_context_text = format_graph_context_for_prompt(
        graph_context or [], max_chars=budget.graph_chars,
    )
    analytical_context_text = build_analytical_context(question)
    language = infer_answer_language(question)
    system = (
        "You are COMPASS Chat, a research assistant for political manifesto analysis. "
        "Your task is evidence-grounded analysis, not open-ended political commentary. "
        "Use only the material in this prompt. Do not use outside knowledge, memory, assumptions, or likely facts. "
        "There are four evidence-framing inputs: ANALYTICAL_CONTEXT, GENERAL_CONTEXT, RELATIONAL_CONTEXT, and CITED_EVIDENCE. "
        "ANALYTICAL_CONTEXT gives a political-science reading frame; it is not factual evidence and must never be cited. "
        "GENERAL_CONTEXT gives document-level orientation only; it is not proof and must never be cited. "
        "RELATIONAL_CONTEXT contains inferred entity co-occurrences from the political graph. It is not verified fact, "
        "must never be cited, and must be ignored when it conflicts with CITED_EVIDENCE. "
        "CITED_EVIDENCE contains PRIMARY_EVIDENCE, NUANCE_EVIDENCE, and COUNTER_EVIDENCE_CANDIDATES. "
        "Each [S] source refers to both its evidence segment and the local_parent_context supplied with it. "
        "All are citable passages, but a counter-evidence candidate must be described as contradictory only when its text actually conflicts. "
        "Every substantive political claim must end with an inline citation like [S1] or [S2]. "
        "Never cite [A], [C1], [C2], [R1], or any analytical/general/relational-context label. "
        "If the cited evidence does not answer the question, say that the provided evidence is insufficient and explain what is missing. "
        "If a user premise is not supported by the cited evidence, correct it cautiously instead of accepting it. "
        "Do not treat absence of evidence as evidence of absence. If the cited evidence mentions only one actor, "
        "party, policy, country, or event, you may state that only this item appears in the provided evidence. "
        "You must not conclude that no other items existed unless the evidence explicitly states the list is exhaustive. "
        "Do not add a separate bibliography or sources section; the interface adds sources separately. "
        "Do not infer motives, psychology, or hidden positions beyond the passages. "
        "Do not overinterpret; if a conclusion is not directly supported, phrase it cautiously or say the evidence is insufficient. "
        "Distinguish direct declaration from synthesis: say 'the passage explicitly states' only when one passage says it directly; "
        "for a synthesis use cautious wording such as 'taken together, the cited passages suggest'. "
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
        "RELATIONAL_CONTEXT - inferred graph co-occurrences only, never cite this block:\n"
        f"{relational_context_text}\n\n"
        "CITED_EVIDENCE - the only claim-supporting evidence:\n"
        f"{evidence_context}\n\n"
        "Answer contract:\n"
        "- Start with a direct answer, then organize the supporting points clearly.\n"
        "- Use [S1], [S2], etc. for every claim.\n"
        "- An [S] source covers its evidence segment and its supplied local_parent_context.\n"
        "- Never invent source ids. Use only the [S] ids present above.\n"
        "- Use ANALYTICAL_CONTEXT only to understand the political concept being asked about.\n"
        "- Label direct declarations as explicit and multi-source interpretations as cautious synthesis.\n"
        "- Do not cite [A], [C1], [R1], or other analytical/general/relational-context labels.\n"
        "- If no [S] passage supports the answer, say evidence is insufficient.\n"
        "- If evidence is weak or absent, say so explicitly.\n"
        "- Do not repeat or restate the question text in your answer; begin directly with the substantive claim.\n\n"
        "Citation style, illustrated on an unrelated topic (do not reuse this wording or copy any question text):\n"
        "Correct: The party calls for lower payroll taxes on low incomes [S1]. Taken together, the cited "
        "passages suggest this is framed as a fairness measure [S1][S2].\n"
        "Incorrect (missing citations): The party calls for lower payroll taxes on low incomes. "
        "This is framed as a fairness measure."
    )
    history = compact_history(
        _previous_history_only(request.history, question),
        max_messages=_MAX_CHAT_HISTORY_MESSAGES,
        max_chars=_MAX_CHAT_HISTORY_CHARS,
    )
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def _exclude_evidence_from_general_context(
    general_context: list[GeneralContext],
    citations: list[Citation],
) -> list[GeneralContext]:
    evidence_ids = {citation.segment_id for citation in citations}
    evidence_texts = {
        fingerprint
        for citation in citations
        for fingerprint in (
            _text_fingerprint(citation.text),
            _text_fingerprint(citation.parent_text or ""),
        )
        if fingerprint
    }
    selected: list[GeneralContext] = []
    seen_texts: set[str] = set()
    for item in general_context:
        fingerprint = _text_fingerprint(item.text)
        if (
            not fingerprint
            or item.segment_id in evidence_ids
            or fingerprint in evidence_texts
            or fingerprint in seen_texts
        ):
            continue
        seen_texts.add(fingerprint)
        selected.append(GeneralContext(
            ref_id=f"C{len(selected) + 1}",
            segment_id=item.segment_id,
            text=item.text,
            country_iso3=item.country_iso3,
            party_id=item.party_id,
            doc_date=item.doc_date,
            doc_type=item.doc_type,
            section_title=item.section_title,
        ))
    return selected


def _previous_history_only(
    history: list[dict[str, str]],
    current_question: str,
) -> list[dict[str, str]]:
    if not history:
        return []
    last = history[-1]
    if (
        str(last.get("role") or "").lower() == "user"
        and " ".join(str(last.get("content") or "").split()).casefold()
        == " ".join(current_question.split()).casefold()
    ):
        return history[:-1]
    return history


def attach_parent_context(memory: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add parent chunk text to retrieved child records when available."""
    parent_ids = list({
        (record.get("meta") or {}).get("parent_segment_id", "")
        for record in records
        if (record.get("meta") or {}).get("parent_segment_id")
    })
    if not parent_ids:
        return records
    try:
        if hasattr(memory, "fetch_records_by_ids"):
            parent_records = {
                item["segment_id"]: item
                for item in memory.fetch_records_by_ids(parent_ids)
            }
        elif hasattr(memory, "fetch_by_ids"):
            parent_records = {
                segment_id: {"segment_id": segment_id, "text": text, "meta": {}}
                for segment_id, text in memory.fetch_by_ids(parent_ids).items()
            }
        else:
            return records
    except Exception:
        return records
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        meta = item.get("meta") or {}
        parent_id = meta.get("parent_segment_id", "")
        parent = parent_records.get(parent_id)
        if parent and _same_scope_metadata(meta, parent.get("meta") or {}):
            item["parent_text"] = parent.get("text") or ""
        enriched.append(item)
    return enriched


def _same_scope_metadata(child: dict[str, Any], parent: dict[str, Any]) -> bool:
    for key in ("doc_id", "country_iso3", "party_id", "election_id", "language"):
        left = str(child.get(key) or "")
        right = str(parent.get(key) or "")
        if left and right and left != right:
            return False
    return True


def retrieve_general_context(
    memory: Any,
    question: str,
    request: ChatRequest,
    evidence_records: list[dict[str, Any]],
    analysis: QueryAnalysis,
) -> list[GeneralContext]:
    """Select broad and topically distinct parent chunks as non-cited context."""
    records: list[dict[str, Any]] = []
    queries = build_general_context_queries(question, analysis)
    for query in queries:
        try:
            if hasattr(memory, "query_parent_documents_hybrid"):
                found = memory.query_parent_documents_hybrid(
                    query,
                    as_of=request.as_of,
                    k=settings.chat_general_context_items * 2,
                    party_id=request.party_id,
                    include_unverified=request.include_unverified,
                )
            elif hasattr(memory, "query_documents_hybrid"):
                found = memory.query_documents_hybrid(
                    query,
                    as_of=request.as_of,
                    k=settings.chat_general_context_items * 2,
                    party_id=request.party_id,
                    include_unverified=request.include_unverified,
                    include_parent_segments=True,
                )
            else:
                try:
                    found = memory.query_documents(
                        query,
                        as_of=request.as_of,
                        k=settings.chat_general_context_items * 2,
                        party_id=request.party_id,
                        include_unverified=request.include_unverified,
                        include_parent_segments=True,
                    )
                except TypeError:
                    found = memory.query_documents(
                        query,
                        as_of=request.as_of,
                        k=settings.chat_general_context_items * 2,
                        party_id=request.party_id,
                        include_unverified=request.include_unverified,
                    )
        except Exception:
            continue
        records.extend(found)
    country = str(getattr(memory, "country", "") or "")
    scoped = [
        record for record in records
        if record_in_scope(
            record,
            as_of=request.as_of,
            country_iso3=country,
            party_id=request.party_id,
            election_id=request.election_id,
            include_unverified=request.include_unverified,
        )
    ]
    if not scoped:
        scoped = evidence_records
    return build_general_context_items(
        scoped,
        limit=settings.chat_general_context_items,
    )


def build_general_context_queries(question: str, analysis: QueryAnalysis) -> list[str]:
    focus = " ".join(analysis.themes[:3]).strip()
    queries = [
        f"{question} overall manifesto orientation values priorities",
        f"{focus} broader program context objectives instruments".strip(),
    ]
    return list(dict.fromkeys(query for query in queries if query.strip()))


def retrieve_graph_context(
    graph: Any | None,
    question: str,
    request: ChatRequest,
    scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return temporal party relations only for explicitly relational questions."""
    if graph is None or not _is_relational_question(question):
        return []
    party_id = request.party_id or _single_scope_party_id(scope)
    if not party_id:
        return []
    try:
        return list(graph.query_party(
            party_id=party_id,
            as_of=request.as_of,
            k_hops=2,
            top_k=_MAX_RELATIONAL_CONTEXT_ITEMS,
        ))
    except Exception:
        return []


def _is_relational_question(question: str) -> bool:
    normalized = _normalize_intent_text(question)
    markers = (
        "alliance", "coalition", "partenaire", "relation", "oppose", "opposition",
        "rival", "adversaire", "fusion", "scission", "acteur", "avec qui",
        "ally", "allies", "partner", "relationship", "opponent", "merger",
        "split", "actor", "who does", "who is",
    )
    return any(marker in normalized for marker in markers)


def _single_scope_party_id(scope: dict[str, Any]) -> str | None:
    parties = scope.get("parties") or []
    if len(parties) != 1 or not isinstance(parties[0], dict):
        return None
    return _none_if_empty(parties[0].get("party_id"))


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


def compute_prompt_budget(
    citation_count: int,
    general_count: int,
    graph_count: int,
    extra_overhead_chars: int = 0,
) -> PromptBudget:
    target_chars = int(
        max(800, settings.chat_llm_context_window - settings.chat_prompt_reserved_output_tokens)
        * settings.chat_prompt_chars_per_token
    )
    content_chars = max(900, target_chars - 5200 - extra_overhead_chars)
    evidence_total = int(content_chars * 0.45)
    parent_total = int(content_chars * 0.22)
    general_total = int(content_chars * 0.25)
    graph_total = max(100, content_chars - evidence_total - parent_total - general_total)
    return PromptBudget(
        evidence_chars=min(
            settings.chat_max_evidence_text_chars,
            max(140, evidence_total // max(1, citation_count)),
        ),
        parent_chars=min(
            settings.chat_max_parent_context_chars,
            max(80, parent_total // max(1, citation_count)),
        ),
        general_chars=min(
            settings.chat_max_general_context_chars,
            max(120, general_total // max(1, general_count)),
        ),
        graph_chars=max(80, graph_total // max(1, graph_count)),
        target_chars=target_chars,
    )


def format_evidence_for_prompt(
    citations: list[Citation],
    budget: PromptBudget,
) -> str:
    labels = {
        "primary": "PRIMARY_EVIDENCE",
        "nuance": "NUANCE_EVIDENCE",
        "counter": "COUNTER_EVIDENCE_CANDIDATES",
    }
    blocks: list[str] = []
    for role in ("primary", "nuance", "counter"):
        role_citations = [citation for citation in citations if citation.evidence_role == role]
        if not role_citations:
            continue
        passages = "\n\n".join(
            f"[{citation.ref_id}] "
            f"country={citation.country_iso3 or 'unknown'} party={citation.party_id or 'unknown'} "
            f"date={citation.doc_date or 'unknown'} type={citation.doc_type or 'unknown'} "
            f"reliability={citation.reliability or 'unknown'}\n"
            f"retrieval_reason={citation.retrieval_reason or 'not available'}\n"
            f"local_parent_context={_compact_parent_context(citation, budget.parent_chars)}\n"
            f"{_excerpt(citation.text, max_chars=budget.evidence_chars)}"
            for citation in role_citations
        )
        blocks.append(f"{labels[role]}:\n{passages}")
    return "\n\n".join(blocks) or "No citable evidence retrieved."


def format_retrieval_trace(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "No retrieval trace available."
    lines: list[str] = []
    for step in trace:
        stage = str(step.get("stage") or "step")
        details = ", ".join(
            f"{key}={value}"
            for key, value in step.items()
            if key != "stage" and key not in {"selected_ids"}
        )
        lines.append(f"- {stage}: {details}"[:320])
    rendered = "\n".join(lines)
    limit = settings.chat_max_retrieval_trace_chars
    if len(rendered) > limit:
        return rendered[: limit - 28].rstrip() + "\n- trace: truncated for budget"
    return rendered


def format_general_context_for_prompt(
    items: list[GeneralContext],
    max_chars: int | None = None,
) -> str:
    if not items:
        return "No separate general context retrieved."
    lines = []
    text_limit = max_chars or settings.chat_max_general_context_chars
    for item in items[:settings.chat_general_context_items]:
        lines.append(
            f"[{item.ref_id}] country={item.country_iso3 or 'unknown'} "
            f"party={item.party_id or 'unknown'} date={item.doc_date or 'unknown'} "
            f"type={item.doc_type or 'unknown'} segment={item.segment_id} "
            f"section={item.section_title or 'unknown'}\n"
            f"{_excerpt(item.text, max_chars=text_limit)}"
        )
    return "\n\n".join(lines)


def format_graph_context_for_prompt(
    items: list[dict[str, Any]],
    max_chars: int = 260,
) -> str:
    if not items:
        return "No relevant political-graph context retrieved."
    lines = []
    for index, item in enumerate(items[:_MAX_RELATIONAL_CONTEXT_ITEMS], start=1):
        summary = str(item.get("summary") or "").strip()
        if summary:
            lines.append(
                f"[R{index}] [INFERRED, NOT EVIDENCE] "
                f"{_excerpt(summary, max_chars=max_chars)}"
            )
    return "\n".join(lines) or "No relevant political-graph context retrieved."


def _compact_parent_context(citation: Citation, max_chars: int | None = None) -> str:
    if not citation.parent_text:
        return "none"
    return _excerpt(
        citation.parent_text,
        max_chars=max_chars or settings.chat_max_parent_context_chars,
    )


def extract_segment_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in _SEGMENT_ID_RE.findall(text):
        if match not in seen:
            seen.add(match)
            ids.append(match)
    return ids


def extract_scientific_variable(text: str) -> str | None:
    for pattern in (
        _SCIENTIFIC_ANALYSIS_RE,
        _SCIENTIFIC_VALIDATION_RE,
        _SCIENTIFIC_CONTAMINATION_RE,
    ):
        match = pattern.match(text or "")
        if match and match.group(1):
            return match.group(1)
    return None


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
    normalized = _normalize_intent_text(question)
    if normalized in {"/variables", "variables", "variables scientifiques"}:
        return "SCIENTIFIC_VARIABLES"
    if _SCIENTIFIC_CONTAMINATION_RE.match(question):
        return "SCIENTIFIC_CONTAMINATION"
    if _SCIENTIFIC_VALIDATION_RE.match(question):
        return "SCIENTIFIC_VALIDATION"
    if _SCIENTIFIC_ANALYSIS_RE.match(question):
        return "SCIENTIFIC_ANALYSIS"
    if extract_segment_ids(question):
        return "direct_lookup"
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
                "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA, SCIENTIFIC_ANALYSIS, "
                "SCIENTIFIC_VALIDATION, SCIENTIFIC_VARIABLES, or SCIENTIFIC_CONTAMINATION. "
                "direct_lookup means the user requests an explicit segment id. "
                "corpus_scope means the user asks which corpus, dataset, country, party, date scope, or storage is active. "
                "FOLLOW_UP_SOURCES asks for sources used in the previous answer. "
                "OUT_OF_CORPUS requests an exhaustive list not established by the active documentary corpus. "
                "COMPARISON_NEEDS_MORE_CORPUS compares at least two named political parties; comparing two themes is evidence_query. "
                "ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA asks who governed, who won, turnout, seats, or election results. "
                "SCIENTIFIC_ANALYSIS runs an explicit /analyse <variable_id> command. "
                "SCIENTIFIC_VALIDATION runs /valider on results produced in the session. "
                "SCIENTIFIC_VARIABLES lists registered variables. "
                "SCIENTIFIC_CONTAMINATION runs an explicit /contamination <variable_id> audit command. "
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


_ECHOED_LABEL_RE = re.compile(
    r"^\s*(question|réponse|reponse|answer)\s*:\s*.*?(?:\n+|$)",
    flags=re.IGNORECASE,
)


def strip_echoed_question_label(answer: str) -> str:
    """Drop a leading 'Question:'/'Réponse:' line some local models echo from the prompt."""
    return _ECHOED_LABEL_RE.sub("", answer.strip(), count=1).strip()


def generate_validated_answer(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    citations: list[Citation],
    route: str,
    complete: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    """Generate, validate, and repair with the exact same evidence package."""
    trace: list[dict[str, Any]] = []
    current_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(settings.chat_repair_max_attempts + 1):
        raw = ""
        try:
            raw = complete(
                model_name,
                current_messages,
                temperature=0.0,
                max_tokens=min(settings.llm_max_tokens, _MAX_CHAT_OUTPUT_TOKENS),
            )
            if not raw:
                raise AnswerContractError("empty answer")
            answer = strip_echoed_question_label(strip_appended_sources(raw))
            try:
                validate_llm_answer(answer, citations, route=route)
            except Exception as syntax_exc:
                trace.append({
                    "stage": "syntax",
                    "attempt": attempt,
                    "status": "rejected",
                    "error": str(syntax_exc),
                })
                raise
            trace.append({
                "stage": "syntax",
                "attempt": attempt,
                "status": "accepted",
            })
            if settings.chat_semantic_validation_enabled:
                claims = evaluate_semantic_grounding(answer, citations)
                trace.extend({
                    "stage": "nli_claim",
                    "attempt": attempt,
                    "status": claim.status,
                    "claim": claim.claim,
                    "refs": claim.refs,
                    "best_label": claim.best_label,
                    "best_score": round(claim.best_score, 4),
                } for claim in claims)
                rejected = [claim for claim in claims if claim.status == "rejected"]
                if rejected:
                    raise AnswerContractError(
                        f"{len(rejected)} claim(s) failed semantic grounding"
                    )
            trace.append({
                "stage": "answer",
                "attempt": attempt,
                "status": "accepted",
            })
            return answer, trace, current_messages
        except Exception as exc:
            last_error = exc
            trace.append({
                "stage": "answer",
                "attempt": attempt,
                "status": "rejected",
                "error": str(exc),
            })
            if attempt >= settings.chat_repair_max_attempts:
                break
            current_messages = build_repair_messages(
                messages,
                raw,
                trace,
                citations,
            )
    raise RepairExhaustedError(
        f"answer validation failed after repair: {last_error}",
        trace,
    ) from last_error


def build_repair_messages(
    original_messages: list[dict[str, str]],
    rejected_answer: str,
    trace: list[dict[str, Any]],
    citations: list[Citation],
) -> list[dict[str, str]]:
    allowed = ", ".join(citation.ref_id for citation in citations)
    failures = [
        str(step.get("error") or step.get("claim") or "validation failure")
        for step in trace
        if step.get("status") == "rejected"
    ][-8:]
    instruction = (
        "Your draft failed COMPASS validation. Rewrite it, using exactly the same evidence already "
        "present in the previous user message. Do not retrieve, add, or assume new information. "
        f"Allowed source ids: {allowed}. Every substantive political claim needs an inline source id. "
        "Use 'the passage explicitly states' only for a direct statement. For a synthesis, write "
        "'taken together, the cited passages suggest' and cite all supporting sources. If support is "
        "insufficient, abstain explicitly. Validation failures:\n- "
        + "\n- ".join(failures)
    )
    return [
        *original_messages,
        {"role": "assistant", "content": rejected_answer},
        {"role": "user", "content": instruction},
    ]


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
    forbidden = [
        ref for ref in refs
        if ref == "A" or ref.startswith("C") or ref.startswith("R")
    ]
    if forbidden:
        raise AnswerContractError("answer cited analytical/general context")

    valid_source_ids = {citation.ref_id for citation in citations[:settings.chat_max_prompt_citations]}
    source_refs = [ref for ref in refs if ref.startswith("S")]
    unknown = sorted({ref for ref in source_refs if ref not in valid_source_ids})
    if unknown:
        raise AnswerContractError(f"answer cited unknown source ids: {', '.join(unknown)}")

    if not source_refs and not _is_insufficiency_answer(text):
        raise AnswerContractError("substantive answer without cited evidence")
    uncited = [
        sentence for sentence in _substantive_sentences(text)
        if not any(ref.startswith("S") for ref in _ANSWER_REF_RE.findall(sentence))
    ]
    if uncited and not _is_insufficiency_answer(text):
        raise AnswerContractError(
            "uncited substantive claim: " + _ANSWER_REF_RE.sub("", uncited[0])[:180]
        )


def _is_insufficiency_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _INSUFFICIENT_MARKERS)


def _substantive_sentences(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    return [
        sentence.strip(" -*#")
        for sentence in sentences
        if len(sentence.strip(" -*#").split()) >= 4
        and not sentence.strip().endswith(":")
    ]


def evaluate_semantic_grounding(
    answer: str,
    citations: list[Citation],
) -> list[ClaimValidation]:
    """Return phrase-level NLI decisions without hiding rejected claims."""
    from compass.nlp_models import nli_pipeline

    by_ref = {citation.ref_id: citation for citation in citations}
    classifier = nli_pipeline()
    decisions: list[ClaimValidation] = []
    for sentence in _substantive_sentences(answer):
        refs = [ref for ref in _ANSWER_REF_RE.findall(sentence) if ref.startswith("S")]
        if not refs:
            continue
        claim = _ANSWER_REF_RE.sub("", sentence).strip()
        if not claim:
            continue
        best_label: str | None = None
        best_score = 0.0
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
            if score > best_score:
                best_label, best_score = label, score
            if "entail" in label and score >= settings.chat_nli_entailment_threshold:
                supported = True
        if not supported and len(refs) > 1:
            combined = "\n".join(
                by_ref[ref].text for ref in refs if ref in by_ref
            )
            if combined:
                result = classifier({"text": combined, "text_pair": claim})
                if isinstance(result, list):
                    result = result[0] if result else {}
                label = str((result or {}).get("label") or "").lower()
                score = float((result or {}).get("score") or 0.0)
                if score > best_score:
                    best_label, best_score = label, score
                supported = (
                    "entail" in label
                    and score >= settings.chat_nli_entailment_threshold
                )
        decisions.append(ClaimValidation(
            claim=claim,
            refs=refs,
            status="accepted" if supported else "rejected",
            best_label=best_label,
            best_score=best_score,
        ))
    if not decisions and not _is_insufficiency_answer(answer):
        raise AnswerContractError("semantic grounding found no cited claim")
    return decisions


def validate_semantic_grounding(answer: str, citations: list[Citation]) -> None:
    decisions = evaluate_semantic_grounding(answer, citations)
    if any(decision.status == "rejected" for decision in decisions):
        raise AnswerContractError("semantic grounding check failed")


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


def _text_fingerprint(text: str) -> str:
    normalized = " ".join(str(text or "").casefold().split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+", str(text or ""), flags=re.UNICODE))


def _default_chat_model() -> str:
    if settings.judge_models:
        return settings.judge_models[0]
    return settings.hyde_model


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
