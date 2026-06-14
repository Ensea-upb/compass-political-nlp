"""C12 — Agrégation des juges et mesure du désaccord.

ÉTAT DE L'ART RÉUTILISÉ :
    - numpy/scipy : moyennes, médianes, dispersion — rien à réinventer.
    - krippendorff : alpha de Krippendorff, LA mesure standard d'accord
      inter-codeurs en science politique (utilisée pour le CMP).
    - Pour l'option IRT (la méthode V-Dem, Pemstein et al. 2018) : py-irt /
      PyMC en extension — PAS implémentée from scratch ; volontairement
      différée car elle exige beaucoup de cas par juge pour être estimée.

POINT OUVERT (architecture, bloc 12) : la méthode d'agrégation est un
PARAMÈTRE à tester empiriquement sur le pilote, pas une décision a priori —
d'où l'interface unique ``aggregate(method=...)``.

CUSTOM (justifié) : la décomposition de l'incertitude en deux composantes
(preuve vs jugement) — exigence de l'audit, absente des libs : un panel
unanime sur des preuves pauvres n'est PAS un cas certain.
"""

from __future__ import annotations

import logging
import statistics

import krippendorff
import numpy as np

from compass.schemas import AggregatedJudgment, JudgeAnswer

logger = logging.getLogger(__name__)

_METHODS = ("mean", "median", "majority")


def aggregate(answers: list[JudgeAnswer], method: str = "median") -> AggregatedJudgment:
    """Combine les réponses des juges et quantifie leur divergence.

    Args:
        answers: réponses individuelles (≥ 1).
        method: 'mean' | 'median' | 'majority' — choisie empiriquement (pilote).

    Returns:
        Score agrégé + désaccord (écart-type inter-juges).

    Raises:
        ValueError: méthode inconnue ou panel vide.
    """
    if not answers:
        raise ValueError("Panel vide : rien à agréger.")
    if method not in _METHODS:
        raise ValueError(f"Méthode inconnue : {method} (choix : {_METHODS})")

    scores = [a.score for a in answers if a.score is not None and not _is_nan(a.score)]
    if not scores:
        raise ValueError("Aucun score exploitable dans le panel.")

    if method == "mean":
        final = float(np.mean(scores))
    elif method == "median":
        final = float(np.median(scores))
    else:  # majority — pour échelles ordinales discrètes
        final = float(statistics.mode([round(s) for s in scores]))

    disagreement = float(np.std(scores)) if len(scores) > 1 else 0.0
    logger.info("Agrégation (%s) : %.2f, désaccord %.2f (n=%d)",
                method, final, disagreement, len(scores))
    return AggregatedJudgment(score=final, disagreement=disagreement,
                              method=method, individual=answers)


def panel_alpha(panel_scores: np.ndarray) -> float:
    """Alpha de Krippendorff du panel sur un lot de cas (fiabilité inter-juges).

    Args:
        panel_scores: matrice (n_juges, n_cas), NaN admis.

    Returns:
        Alpha ordinal — comparable aux standards humains du champ (κ ≥ 0.60
        retenu comme seuil dans Partie 3 §4.1).
    """
    return float(krippendorff.alpha(reliability_data=panel_scores,
                                    level_of_measurement="ordinal"))


def combined_confidence(disagreement: float, sufficiency_proba: float,
                        scale_range: float) -> float:
    """Confiance finale = f(désaccord des juges, suffisance des preuves).

    Deux composantes MULTIPLICATIVES (audit : un panel unanime sur preuves
    pauvres n'est pas un cas certain) :
        - composante jugement : 1 - désaccord normalisé par l'étendue d'échelle ;
        - composante preuve  : probabilité de suffisance calibrée (C07).

    Cette formule est un POINT DE DÉPART à recalibrer sur le pilote (R-3) —
    la calibration finale se mesure, ne se décrète pas.
    """
    judgment_part = max(0.0, 1.0 - (disagreement / max(scale_range, 1e-9)))
    return round(judgment_part * sufficiency_proba, 3)


def aggregate_multilabel(answers: list[JudgeAnswer],
                         min_votes: int = 2) -> AggregatedJudgment:
    """Agrégation des variables multi-sélection (P0-4) : vote par label.

    Un label est retenu s'il est proposé par au moins ``min_votes`` juges.
    Le désaccord est 1 - Jaccard moyen entre les ensembles des juges.
    """
    if not answers:
        raise ValueError("Panel vide : rien à agréger.")
    sets = [set(a.labels) for a in answers if a.labels]
    if not sets:
        return AggregatedJudgment(score=None, labels=[], disagreement=1.0,
                                  method="multilabel_vote", individual=answers)
    counts: dict[str, int] = {}
    for s in sets:
        for lab in s:
            counts[lab] = counts.get(lab, 0) + 1
    retained = sorted(lab for lab, c in counts.items() if c >= min(min_votes, len(sets)))
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    if pairs:
        jacc = sum(len(a & b) / max(len(a | b), 1) for a, b in pairs) / len(pairs)
    else:
        jacc = 1.0
    return AggregatedJudgment(score=None, labels=retained,
                              disagreement=round(1.0 - jacc, 3),
                              method="multilabel_vote", individual=answers)


def _is_nan(x: float) -> bool:
    return x != x

