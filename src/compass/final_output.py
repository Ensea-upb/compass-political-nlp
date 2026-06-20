"""C13 — Sortie finale : score + justification + incertitude, auditables.

ÉTAT DE L'ART RÉUTILISÉ :
    - pydantic : le format de sortie est le schéma ``FinalAnswer`` (bloc 13
      de l'architecture), sérialisable JSON — auditable par construction.
    - Vérification d'attribution : le MÊME modèle NLI que C09 (réutilisation),
      appliqué cette fois en mode « la preuve implique-t-elle l'affirmation ? »
      — cadre AIS (Rashkin et al. 2023) : une citation n'est une justification
      que si elle SOUTIENT réellement l'affirmation (recommandation R-5).

CUSTOM (justifié) : l'assemblage du dossier de sortie (mise en forme) —
trivial et métier.
"""

from __future__ import annotations

import logging

from compass.nlp_models import nli_pipeline

from compass.aggregation import combined_confidence
from compass.schemas import (AggregatedJudgment, CaseKey, Diagnosis, EvidenceItem,
                     FinalAnswer, VariableSheet)

logger = logging.getLogger(__name__)


class AnswerComposer:
    """Compose la réponse finale et vérifie la fidélité des justifications."""

    def __init__(self, entailment_threshold: float = 0.7) -> None:
        self._nli = nli_pipeline()
        self._threshold = entailment_threshold

    def compose(self, case: CaseKey, sheet: VariableSheet, diagnosis: Diagnosis,
                judgment: AggregatedJudgment, sufficiency_proba: float) -> FinalAnswer:
        """Construit la réponse auditable au format du bloc 13."""
        scale_values = [float(k) for k in sheet.scale if str(k).lstrip("-").replace(".", "").isdigit()]
        scale_range = (max(scale_values) - min(scale_values)) if scale_values else 1.0

        answer = FinalAnswer(
            case=case, variable_id=sheet.variable_id,
            score=judgment.score,
            confidence=combined_confidence(judgment.disagreement, sufficiency_proba,
                                           scale_range),
            main_evidence=diagnosis.convergent,
            counter_evidence=diagnosis.contradictory,
            declared=[e.segment.text for e in diagnosis.convergent
                      if e.regime.value == "declared"],
            observed=[e.segment.text for e in diagnosis.convergent
                      if e.regime.value == "observed"],
            inferred=[a.rationale for a in judgment.individual],
            residual_uncertainty="; ".join(diagnosis.missing) or "aucun manque identifié",
            sources=sorted({e.segment.meta.source_url or e.segment.meta.source_path or e.segment.doc_id
                            for e in diagnosis.convergent + diagnosis.contradictory}),
        )
        answer.attribution_checked = self._check_attribution(answer)
        return answer

    def abstain(self, case: CaseKey, sheet: VariableSheet,
                diagnosis: Diagnosis, sufficiency_proba: float) -> FinalAnswer:
        """Réponse d'abstention — un produit scientifique, pas un échec.

        Le verdict « information insuffisante » alimente la carte de
        l'observabilité réelle par variable × pays × période (apport candidat
        identifié dans l'audit, §3.2 de 03_apports.md).
        """
        return FinalAnswer(
            case=case, variable_id=sheet.variable_id, score=None,
            confidence=None, abstained=True,
            main_evidence=diagnosis.convergent,
            counter_evidence=diagnosis.contradictory,
            declared=[e.segment.text for e in diagnosis.convergent
                      if e.regime.value == "declared"],
            observed=[e.segment.text for e in diagnosis.convergent
                      if e.regime.value == "observed"],
            residual_uncertainty=f"Abstention (p_suffisance={sufficiency_proba:.2f}) ; "
                                 f"manques : {'; '.join(diagnosis.missing) or 'non spécifiés'}",
            sources=sorted({
                e.segment.meta.source_url or e.segment.meta.source_path or e.segment.doc_id
                for e in diagnosis.convergent + diagnosis.contradictory
            }),
        )

    # ------------------------------------------------------------------ AIS / R-5
    def _check_attribution(self, answer: FinalAnswer) -> bool:
        """Chaque preuve principale soutient-elle réellement le score proposé ?

        Test NLI preuve -> affirmation. Si une part trop faible des preuves
        est jugée « entailment », la réponse est marquée non vérifiée — elle
        part en revue humaine (C14), pas à la corbeille.
        """
        if not answer.main_evidence or answer.score is None:
            return False
        claim = (f"Le parti mérite le score {answer.score} pour "
                 f"{answer.variable_id} au sens du codebook V-Party.")
        supported = 0
        for ev in answer.main_evidence[:8]:
            res = self._nli({"text": ev.segment.text, "text_pair": claim})
            if res["label"].lower() == "entailment" and res["score"] >= self._threshold:
                supported += 1
        ratio = supported / min(len(answer.main_evidence), 8)
        logger.info("Attribution %s : %.0f%% de preuves supportives",
                    answer.variable_id, 100 * ratio)
        return ratio >= 0.5

