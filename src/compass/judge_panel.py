"""C11 — Juges artificiels multiples : pluralité des évaluations.

ÉTAT DE L'ART RÉUTILISÉ :
    - litellm : interface unifiée vers des modèles de bases DIFFÉRENTES
      (GPT, Mistral, Claude…) — condition d'indépendance documentée :
      Weidmann et al. (2026) montrent que les biais directionnels sont
      propres à chaque modèle et s'atténuent par combinaison de modèles
      opposés ; Le Mens & Gallego (2025) valident le gain du moyennage.
    - Le moteur C10 est réutilisé tel quel : un juge = (modèle, variante
      de prompt) appliqué au MÊME diagnostic.

CHOIX D'ARCHITECTURE EXPLICITE (trade-off documenté dans l'audit) :
    Les juges PARTAGENT le dossier et le diagnostic (coût maîtrisé) et ne
    diffèrent que par le raisonnement final. Conséquence assumée : leurs
    erreurs restent corrélées via les preuves communes — le désaccord mesuré
    est donc une BORNE INFÉRIEURE de l'incertitude. L'indépendance effective
    est mesurée (``error_correlation``) sur les cas de contrôle du pilote.

CUSTOM (justifié) : l'orchestration (~40 lignes) — aucun framework ne
s'impose pour 3 appels parallèles ; en ajouter un serait de la complexité
sans gain (anti-pattern).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from compass.reasoning_engine import ReasoningEngine
from compass.config import settings
from compass.schemas import Diagnosis, JudgeAnswer, VariableMethod, VariableSheet

logger = logging.getLogger(__name__)

_PROMPT_VARIANTS = ["standard", "behavior_first"]


class JudgePanel:
    """Panel de juges hétérogènes appliqués au même cas."""

    def __init__(self, engine: ReasoningEngine,
                 models: list[str] | None = None) -> None:
        self._engine = engine
        self._models = models or settings.judge_models

    def evaluate(self, sheet: VariableSheet, diagnosis: Diagnosis) -> list[JudgeAnswer]:
        """Fait évaluer le cas par tous les juges.

        Les méthodes déterministes (structured/rule) n'ont qu'UN juge — la
        pluralité n'a de sens que pour le jugement, pas pour une lecture SQL.
        """
        if sheet.method in (VariableMethod.STRUCTURED_QUERY,
                            VariableMethod.DETERMINISTIC_RULE,
                            VariableMethod.NLP_CLASSIFIER):
            # P1-5 : pas de panel pour les méthodes déterministes NI pour le
            # classifieur unique — un faux panel produirait un désaccord nul
            # artificiel. Le classifieur est UN juge ; le désaccord viendra
            # de sa confrontation aux juges LLM dans COMPOSITE, pas de copies.
            return [self._engine.answer(sheet, diagnosis, model_name="n/a")]

        # P1-5 : un appel par modèle, variantes de prompt en rotation —
        # aucun modèle n'est compté deux fois (poids égaux dans l'agrégation).
        jobs = [(m, _PROMPT_VARIANTS[i % len(_PROMPT_VARIANTS)])
                for i, m in enumerate(self._models)]
        answers: list[JudgeAnswer] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._engine.answer, sheet, diagnosis, m, v): (m, v)
                       for m, v in jobs}
            for fut in as_completed(futures):
                model, variant = futures[fut]
                try:
                    answers.append(fut.result())
                except (ValueError, KeyError, RuntimeError) as exc:
                    logger.error("Juge %s/%s en échec : %s", model, variant, exc)
        logger.info("%s : %d/%d juges ont répondu", sheet.variable_id,
                    len(answers), len(jobs))
        return answers


def error_correlation(panel_scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Corrélation des erreurs entre juges sur des cas de contrôle.

    Args:
        panel_scores: matrice (n_cas, n_juges).
        truth: scores de référence V-Party (n_cas,).

    Returns:
        Matrice de corrélation des erreurs (n_juges, n_juges). Des valeurs
        proches de 1 hors diagonale = juges redondants : le désaccord
        sous-estime l'incertitude et le panel doit être diversifié.
    """
    errors = panel_scores - truth.reshape(-1, 1)
    return np.corrcoef(errors, rowvar=False)

