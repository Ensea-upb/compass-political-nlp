"""C14 — Validation externe, revue humaine et recalibration.

ÉTAT DE L'ART RÉUTILISÉ :
    - scikit-learn : MAE/corrélations, calibration_curve.
    - scipy : corrélation de Spearman (échelles ordinales).
    - L'étalon : scores V-Party officiels importés en C03 (vparty_scores),
      AVEC leurs intervalles — V-Party n'est pas une vérité ponctuelle
      (Pemstein et al. 2018 ; audit AM-11) : on évalue aussi la couverture
      des intervalles, pas seulement l'écart aux points.
    - Splits : leave-country-out et leave-period-out — pratique standard
      de robustesse hors distribution (Ovadia et al. 2019 ; Kamath et al. 2020).

CUSTOM (justifié) : l'ECE (expected calibration error) en ~10 lignes numpy —
formule standard publiée (Guo et al. 2017), pas d'algorithme nouveau ; et la
stratification par langue (R-6), exigence propre au terrain Afrique/Asie.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compass.schemas import FinalAnswer

logger = logging.getLogger(__name__)


class EvaluationVault:
    """P0-3 — coffre de l'étalon V-Party, PHYSIQUEMENT séparé de la production.

    Base SQLite dédiée (settings.vault_path). Aucun composant C01–C13 ne doit
    importer cette classe : seuls C14 (validation) et C15 (sonde contamination)
    y accèdent. La mémoire de production (C03) REFUSE la table vparty_scores.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS vparty_scores ("
        "party_id TEXT NOT NULL, election_id TEXT NOT NULL, "
        "variable_id TEXT NOT NULL, score REAL, ci_low REAL, ci_high REAL, "
        "PRIMARY KEY (party_id, election_id, variable_id))"
    )

    def __init__(self, vault_path: Path | None = None) -> None:
        from compass.config import settings as _settings

        self._conn = sqlite3.connect(vault_path or _settings.vault_path)
        self._conn.execute(self._SCHEMA)

    def import_csv(self, csv_path: Path, column_map: dict[str, str]) -> int:
        """Importe les scores officiels V-Party (téléchargés depuis v-dem.net)."""
        df = pd.read_csv(csv_path).rename(columns=column_map)
        cols = ["party_id", "election_id", "variable_id", "score", "ci_low", "ci_high"]
        df = df[[c for c in cols if c in df.columns]]
        df.to_sql("vparty_scores", self._conn, if_exists="append", index=False)
        logger.info("Vault : %d scores V-Party importés", len(df))
        return len(df)

    def truth(self) -> pd.DataFrame:
        """Étalon complet — exclusivement pour Validator et la sonde C15."""
        return pd.read_sql_query("SELECT * FROM vparty_scores", self._conn)


@dataclass
class ValidationReport:
    """Les quatre critères du bloc 14, par strate."""

    stratum: str
    n_cases: int
    n_abstentions: int
    mae: float
    spearman: float
    interval_coverage: float      # part des scores dans [ci_low, ci_high] V-Party
    ece: float                    # calibration de l'incertitude
    attribution_rate: float       # part des réponses à justification vérifiée


class Validator:
    """Compare les sorties système aux scores experts, par strate."""

    def __init__(self, truth: pd.DataFrame) -> None:
        """
        Args:
            truth: scores V-Party (party_id, election_id, variable_id,
                   score, ci_low, ci_high) — import C03, jamais resaisis.
        """
        self._truth = truth.set_index(["party_id", "election_id", "variable_id"])

    def evaluate(self, answers: list[FinalAnswer], stratum: str = "all") -> ValidationReport:
        """Produit le rapport 4 critères sur un lot de réponses."""
        rows = []
        abstentions = 0
        for a in answers:
            if a.abstained or a.score is None:
                abstentions += 1
                continue
            key = (a.case.party_id, a.case.election_id, a.variable_id)
            if key not in self._truth.index:
                continue
            t = self._truth.loc[key]
            rows.append({
                "pred": a.score, "truth": float(t["score"]),
                "in_interval": float(t["ci_low"]) <= a.score <= float(t["ci_high"]),
                "confidence": a.confidence or 0.0,
                "correct": abs(a.score - float(t["score"])) <= 0.5,
                "attributed": a.attribution_checked,
            })
        if not rows:
            raise ValueError("Aucun cas évaluable — vérifier la jointure avec l'étalon.")
        df = pd.DataFrame(rows)
        rho, _ = spearmanr(df["pred"], df["truth"])
        report = ValidationReport(
            stratum=stratum, n_cases=len(df), n_abstentions=abstentions,
            mae=float((df["pred"] - df["truth"]).abs().mean()),
            spearman=float(rho),
            interval_coverage=float(df["in_interval"].mean()),
            ece=expected_calibration_error(df["confidence"].to_numpy(),
                                           df["correct"].to_numpy()),
            attribution_rate=float(df["attributed"].mean()),
        )
        logger.info("Validation [%s] : MAE=%.3f ρ=%.3f couverture=%.0f%% ECE=%.3f",
                    stratum, report.mae, report.spearman,
                    100 * report.interval_coverage, report.ece)
        return report

    @staticmethod
    def make_splits(cases: pd.DataFrame, by: str = "country_iso3") -> dict[str, pd.Index]:
        """Splits leave-one-out par pays (ou par période) — robustesse OOD."""
        return {f"holdout_{v}": cases.index[cases[by] == v]
                for v in cases[by].unique()}


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE standard (Guo et al. 2017) : |confiance moyenne - exactitude| pondéré."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)

