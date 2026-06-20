"""Schémas de données partagés entre les composants.

Colonne vertébrale — pas un composant. Tout échange inter-composants passe
par ces types : c'est le contrat d'interface qui permet l'assemblage modulaire.

Révision post-audit (2026-06-12) :
    - P0-2 : statut temporel explicite (TemporalStatus, eligible_for_historical_reasoning)
      — une page sans date vérifiable ne peut plus passer pour une preuve antérieure ;
    - P0-4 : types de sortie V-Party (OutputType) — le questionnaire n'est pas
      uniquement scalaire (ordinal, nominal, multi-sélection, dérivé...) ;
    - JudgeAnswer/FinalAnswer acceptent labels (multi-sélection) et score optionnel.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- enums
class EvidenceRegime(str, Enum):
    """Régime de preuve d'un élément — hérité de coding_status (Phase 3.01)."""

    DECLARED = "declared"
    OBSERVED = "observed"
    INFERRED = "inferred"


class VariableMethod(str, Enum):
    """Méthode de traitement d'une variable — clé de routage du moteur C10."""

    STRUCTURED_QUERY = "structured_query"
    DETERMINISTIC_RULE = "deterministic_rule"
    NLP_CLASSIFIER = "nlp_classifier"
    LLM_GUIDED = "llm_guided"
    COMPOSITE = "composite"


class OutputType(str, Enum):
    """Type de sortie de la variable (P0-4) — dicte agrégation ET métrique."""

    SCALAR_NUMERIC = "scalar_numeric"        # ex. part de voix
    ORDINAL = "ordinal"                      # ex. v2paplur 0-4
    NOMINAL = "nominal"                      # catégorie unique
    MULTI_LABEL = "multi_label"              # ex. groupes de soutien
    FREE_TEXT = "free_text"
    DERIVED_DISTRIBUTION = "derived_distribution"  # indices dérivés (propagation)


class SufficiencyVerdict(str, Enum):
    SUFFICIENT = "sufficient"
    SEARCH_MORE = "search_more"
    ABSTAIN = "abstain"


class TemporalStatus(str, Enum):
    """Fiabilité de la datation d'un document (P0-2)."""

    VERIFIED = "verified"      # date fournie par l'opérateur ou extraite (htmldate)
    ESTIMATED = "estimated"    # date plausible mais non confirmée
    UNKNOWN = "unknown"        # aucune date vérifiable


class SourceReliability(str, Enum):
    OFFICIAL = "official"
    ACADEMIC = "academic"
    INDEPENDENT_PRESS = "independent_press"
    GOVERNMENT_PRESS = "government_press"
    PARTISAN = "partisan"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- documents
class DocumentMeta(BaseModel):
    """Métadonnées obligatoires d'un document ingéré (C01).

    Règle P0-2 : ``doc_date`` est la date utilisée pour le filtrage temporel ;
    elle n'est OPPOSABLE que si ``eligible_for_historical_reasoning`` est vrai
    (statut VERIFIED). Un document UNKNOWN peut orienter une enquête mais ne
    constitue jamais une preuve d'antériorité.
    """

    doc_id: str
    country_iso3: str
    party_id: str | None = None
    doc_date: date
    publication_date: date | None = None
    retrieval_date: date | None = None
    temporal_status: TemporalStatus = TemporalStatus.ESTIMATED
    eligible_for_historical_reasoning: bool = True
    doc_type: str
    language: str
    source_url: str | None = None
    source_path: str | None = None
    page: int | None = None
    election_id: str | None = None
    reliability: SourceReliability = SourceReliability.UNKNOWN
    sha256: str | None = None

    def compute_hash(self, text: str) -> str:
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.sha256


class Segment(BaseModel):
    segment_id: str
    doc_id: str
    text: str
    meta: DocumentMeta
    # Chunking hiérarchique (parent-child, Gap 1) :
    # - None  → segment parent (bloc thématique ~400 chars)
    # - str   → segment enfant (phrase), pointe vers son parent
    parent_segment_id: str | None = None
    # Texte du parent injecté à la volée au moment du re-ranking (C06).
    # Non persisté dans ChromaDB — rempli par InternalRetriever.
    parent_text: str | None = None
    # Provenance structurelle conservee pendant toute la chaine d'indexation.
    chunk_index: int = 0
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    section_title: str | None = None


# --------------------------------------------------------------------------- registre
class VariableSheet(BaseModel):
    """Fiche du registre opérationnel V-Party (C05) — une par variable."""

    variable_id: str
    question: str
    definition: str
    scale: dict[int | str, str]
    method: VariableMethod
    output_type: OutputType = OutputType.ORDINAL   # P0-4
    allowed_labels: list[str] = Field(default_factory=list)  # multi_label/nominal
    evidence_regimes: list[EvidenceRegime]
    required_sources: list[str]
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    ambiguous_cases: list[str] = Field(default_factory=list)
    decision_rules: list[str] = Field(default_factory=list)
    adherence_passed: bool = False


# --------------------------------------------------------------------------- cas
class CaseKey(BaseModel):
    country_iso3: str
    party_id: str
    election_id: str
    election_date: date


class EvidenceItem(BaseModel):
    segment: Segment
    regime: EvidenceRegime
    supports: bool
    relevance: float = Field(ge=0.0, le=1.0)
    qualification_method: str = "unqualified"   # P0-5 : trace de QUI a qualifié


class Diagnosis(BaseModel):
    case: CaseKey
    variable_id: str
    convergent: list[EvidenceItem] = Field(default_factory=list)
    contradictory: list[EvidenceItem] = Field(default_factory=list)
    contradictions_detail: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    decisive: list[str] = Field(default_factory=list)
    dominant_language: str = "und"               # P1-6 : routage par langue
    # Gap 3 — résumés relationnels du graphe de connaissances (C02b).
    # Régime INFÉRÉ ; injectés dans le prompt du juge comme contexte global.
    graph_context: list[dict] = Field(default_factory=list)


class JudgeAnswer(BaseModel):
    judge_id: str
    model_name: str
    score: float | None = None                   # None pour multi_label/free_text
    labels: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AggregatedJudgment(BaseModel):
    score: float | None = None
    labels: list[str] = Field(default_factory=list)
    disagreement: float
    method: str
    individual: list[JudgeAnswer]


class FinalAnswer(BaseModel):
    case: CaseKey
    variable_id: str
    output_type: OutputType = OutputType.ORDINAL
    score: float | None
    labels: list[str] = Field(default_factory=list)
    confidence: float | None
    abstained: bool = False
    main_evidence: list[EvidenceItem] = Field(default_factory=list)
    counter_evidence: list[EvidenceItem] = Field(default_factory=list)
    declared: list[str] = Field(default_factory=list)
    observed: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    residual_uncertainty: str = ""
    sources: list[str] = Field(default_factory=list)
    attribution_checked: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

