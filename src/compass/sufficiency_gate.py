"""C07 — Test de suffisance informationnelle, traité en prédiction sélective.

ÉTAT DE L'ART RÉUTILISÉ :
    - Cadre théorique : prédiction sélective / rejet (Chow 1970 ; Geifman &
      El-Yaniv 2017) ; « sufficient context » (Joren et al. 2025).
    - Implémentation : scikit-learn — régression logistique + calibration
      (CalibratedClassifierCV) sur des traits simples. AUCUN classifieur
      maison : sklearn fournit l'estimateur, la calibration et les courbes.
    - Garde-fou littérature : ne JAMAIS se fier à l'autoévaluation brute du
      LLM (les modèles répondent plutôt que s'abstenir — Joren et al. 2025).

CUSTOM (justifié) : l'extraction des traits (couverture des régimes de preuve
requis, nombre/fiabilité/diversité des passages) — métier pur ; et le seuil,
qui sera CALIBRÉ sur la courbe risque-couverture du pilote (R-2), pas choisi
à la main.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from compass.config import settings
from compass.schemas import SufficiencyVerdict, VariableSheet

logger = logging.getLogger(__name__)

_FEATURES = ["n_passages", "mean_relevance", "max_relevance", "n_doc_types",
             "n_reliable_sources", "regimes_covered_ratio"]


def extract_features(passages: list[dict], sheet: VariableSheet) -> np.ndarray:
    """Traits observables de suffisance pour une variable donnée.

    Le trait clé est ``regimes_covered_ratio`` : la part des régimes de preuve
    exigés par la fiche (déclaré/observé) effectivement couverts par au moins
    un passage — un manifeste seul ne suffit pas pour le clientélisme.
    """
    if not passages:
        return np.zeros((1, len(_FEATURES)))
    rel = [p.get("relevance", 0.0) for p in passages]
    doc_types = {p["meta"].get("doc_type", "?") for p in passages}
    reliable = [p for p in passages
                if p["meta"].get("reliability") in ("official", "academic",
                                                    "independent_press")]
    # proxy de couverture : un type de document par régime requis
    needed = max(len(sheet.evidence_regimes), 1)
    covered = min(len(doc_types), needed) / needed
    feats = [len(passages), float(np.mean(rel)), float(np.max(rel)),
             len(doc_types), len(reliable), covered]
    return np.asarray(feats, dtype=float).reshape(1, -1)


class SufficiencyGate:
    """Classifieur calibré « les preuves suffisent-elles ? » + politique de décision."""

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold or settings.sufficiency_threshold
        self._model: CalibratedClassifierCV | None = None

    # ------------------------------------------------------------------ entraînement
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Entraîne sur des cas étiquetés du pilote.

        Étiquette y : 1 si un humain a jugé les preuves suffisantes pour coder,
        0 sinon. La calibration sigmoïde rend la probabilité interprétable.
        """
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        self._model.fit(x, y)
        logger.info("SufficiencyGate entraîné sur %d cas", len(y))

    def save(self, path: str) -> None:
        joblib.dump((self._model, self._threshold), path)

    def load(self, path: str) -> None:
        self._model, self._threshold = joblib.load(path)

    # ------------------------------------------------------------------ décision
    def decide(self, passages: list[dict], sheet: VariableSheet,
               searches_done: int) -> tuple[SufficiencyVerdict, float]:
        """Verdict : poursuivre, chercher davantage, ou s'abstenir.

        La boucle est BORNÉE : au-delà de ``search_max_iterations`` recherches,
        l'insuffisance devient une abstention définitive (signal scientifique
        en soi — carte de l'observabilité réelle).
        """
        if self._model is None:
            # Mode amorçage (avant pilote) : heuristique conservatrice documentée.
            proba = float(extract_features(passages, sheet)[0, -1])  # couverture
            logger.warning("SufficiencyGate non entraîné : heuristique d'amorçage.")
        else:
            proba = float(self._model.predict_proba(
                extract_features(passages, sheet))[0, 1])

        if proba >= self._threshold:
            return SufficiencyVerdict.SUFFICIENT, proba
        if searches_done < settings.search_max_iterations:
            return SufficiencyVerdict.SEARCH_MORE, proba
        return SufficiencyVerdict.ABSTAIN, proba

