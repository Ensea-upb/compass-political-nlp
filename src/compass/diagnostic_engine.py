"""C09 — Qualification des preuves + moteur de diagnostic.

Révision post-audit (2026-06-12) — P0-5 : le maillon manquant est ajouté.
Avant le diagnostic, chaque passage récupéré est QUALIFIÉ :
    passage → pertinence → régime de preuve → polarité (soutien/contradiction/
    neutre) → EvidenceItem tracé (qualification_method).
Le diagnostic ne reçoit plus de preuves pré-interprétées par l'appelant.

ÉTAT DE L'ART RÉUTILISÉ :
    - Polarité et contradictions : NLI multilingue (mDeBERTa-v3 XNLI) — le
      passage est confronté à une hypothèse construite depuis la fiche C05.
      Approche NLI validée sur textes politiques (Laurer et al. 2024).
    - Typologie des conflits : Xu et al. 2024 (inter-contexte).

CUSTOM (justifié) : l'hypothèse de qualification construite depuis la fiche,
le typage heuristique du régime (doc_type) marqué comme tel, la structure du
rapport. La qualification v0 DOIT être validée sur un petit corpus humain
avant le pilote (exigence P0-5 de l'audit) — voir ASSEMBLAGE étape 7.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter

from transformers import pipeline as hf_pipeline

from compass.config import settings
from compass.schemas import (CaseKey, Diagnosis, DocumentMeta, EvidenceItem,
                     EvidenceRegime, Segment, SourceReliability, VariableSheet)

logger = logging.getLogger(__name__)

_LOW_TRUST = {SourceReliability.PARTISAN, SourceReliability.GOVERNMENT_PRESS,
              SourceReliability.UNKNOWN}

# Typage heuristique v0 du régime de preuve par type de document.
# Marqué 'heuristic_doctype_v0' dans qualification_method — à confronter à
# une annotation humaine (ASSEMBLAGE étape 7) avant toute montée en charge.
_REGIME_BY_DOCTYPE: dict[str, EvidenceRegime] = {
    "manifeste": EvidenceRegime.DECLARED,
    "discours": EvidenceRegime.DECLARED,
    "communique": EvidenceRegime.DECLARED,
    "presse": EvidenceRegime.OBSERVED,
    "rapport": EvidenceRegime.OBSERVED,
    "observation_electorale": EvidenceRegime.OBSERVED,
    "web_actif": EvidenceRegime.OBSERVED,
}


class EvidenceQualifier:
    """P0-5 — qualifie les passages bruts en preuves typées et polarisées."""

    def __init__(self, entail_threshold: float = 0.75,
                 contra_threshold: float = 0.75) -> None:
        self._nli = hf_pipeline("text-classification", model=settings.nli_model, **settings.hf_pipeline_kwargs())
        self._entail = entail_threshold
        self._contra = contra_threshold

    def qualify(self, passages: list[dict], sheet: VariableSheet) -> list[EvidenceItem]:
        """Transforme les passages du retrieval en EvidenceItem qualifiés.

        Polarité par NLI contre une hypothèse tirée de la fiche :
        entailment fort → soutien ; contradiction forte → contre-preuve ;
        neutre/ambigu → écarté (journalisé, pas silencieux).
        """
        hypothesis = self._build_hypothesis(sheet)
        items: list[EvidenceItem] = []
        dropped = 0
        for p in passages:
            res = self._nli({"text": p["text"], "text_pair": hypothesis})
            label, score = res["label"].lower(), float(res["score"])
            if label == "entailment" and score >= self._entail:
                supports = True
            elif label == "contradiction" and score >= self._contra:
                supports = False
            else:
                dropped += 1
                continue
            seg = _segment_from_index(p)
            regime = _REGIME_BY_DOCTYPE.get(
                seg.meta.doc_type.replace("_undated", ""), EvidenceRegime.INFERRED)
            items.append(EvidenceItem(
                segment=seg, regime=regime, supports=supports,
                relevance=min(max(p.get("relevance", 0.5), 0.0), 1.0),
                qualification_method="nli_polarity+heuristic_doctype_v0",
            ))
        logger.info("Qualification %s : %d retenus, %d neutres/ambigus écartés",
                    sheet.variable_id, len(items), dropped)
        return items

    @staticmethod
    def _build_hypothesis(sheet: VariableSheet) -> str:
        """Hypothèse NLI dérivée de la fiche — jamais d'un texte libre."""
        return (f"Ce passage montre que le parti satisfait le critère suivant : "
                f"{sheet.question} ({sheet.definition[:200]})")


class DiagnosisEngine:
    """Confronte les preuves qualifiées : convergences, contradictions, manques."""

    def __init__(self, max_pairs: int = 60) -> None:
        self._nli = hf_pipeline("text-classification", model=settings.nli_model, **settings.hf_pipeline_kwargs())
        self._max_pairs = max_pairs

    def diagnose(self, case: CaseKey, sheet: VariableSheet,
                 evidence: list[EvidenceItem]) -> Diagnosis:
        """Produit le rapport de diagnostic pour une variable d'un cas."""
        diag = Diagnosis(case=case, variable_id=sheet.variable_id)
        diag.convergent = [e for e in evidence if e.supports]
        diag.contradictory = [e for e in evidence if not e.supports]

        langs = Counter(e.segment.meta.language for e in evidence)
        diag.dominant_language = langs.most_common(1)[0][0] if langs else "und"

        pairs = list(itertools.combinations(evidence, 2))[: self._max_pairs]
        for a, b in pairs:
            label = self._nli({"text": a.segment.text, "text_pair": b.segment.text})
            if label["label"].lower() == "contradiction" and label["score"] > 0.8:
                diag.contradictions_detail.append(
                    f"[{a.segment.meta.doc_type}] « {a.segment.text[:80]}… » "
                    f"CONTREDIT [{b.segment.meta.doc_type}] « {b.segment.text[:80]}… »"
                )

        covered = {e.regime for e in evidence}
        for regime in sheet.evidence_regimes:
            if regime not in covered:
                diag.missing.append(f"aucune preuve de régime « {regime.value} »")

        for e in evidence:
            reliable = e.segment.meta.reliability not in _LOW_TRUST
            if e.regime == EvidenceRegime.OBSERVED and reliable and e.relevance > 0.7:
                diag.decisive.append(e.segment.segment_id)

        logger.info(
            "Diagnostic %s : %d pour, %d contre, %d contradictions, %d manques",
            sheet.variable_id, len(diag.convergent), len(diag.contradictory),
            len(diag.contradictions_detail), len(diag.missing),
        )
        return diag


def _segment_from_index(p: dict) -> Segment:
    """Reconstruit un Segment depuis les métadonnées d'index ChromaDB."""
    from datetime import date as _date

    m = p["meta"]
    return Segment(
        segment_id=p["segment_id"], doc_id=m.get("doc_id", "?"), text=p["text"],
        meta=DocumentMeta(
            doc_id=m.get("doc_id", "?"),
            country_iso3=m.get("country_iso3", "???"),
            party_id=m.get("party_id") or None,
            doc_date=_date.fromisoformat(m["doc_date"]) if m.get("doc_date") else _date.min,
            doc_type=m.get("doc_type", "?"),
            language=m.get("language", "und"),
            reliability=SourceReliability(m.get("reliability", "unknown")),
        ),
    )

