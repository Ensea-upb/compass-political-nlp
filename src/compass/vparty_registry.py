"""C05 — Registre opérationnel V-Party : une fiche par variable, chargée depuis YAML.

ÉTAT DE L'ART RÉUTILISÉ :
    - pydantic (validation des fiches — schéma ``VariableSheet``).
    - PyYAML (les fiches sont des fichiers YAML lisibles et versionnables).
    - Le CONTENU des fiches vient du codebook V-Party officiel (Lührmann et al.
      2020) — on transcrit, on n'invente pas de définitions.

CUSTOM (justifié) : la gate d'adhérence (``check_adherence``) — recommandation
R-1 de l'audit (Halterman & Keith 2025 : les LLM ne suivent pas spontanément
les codebooks ; sensibilité à l'ordre des catégories, aux labels). Aucune lib
n'implémente ce test : c'est un protocole, pas un algorithme.

RÈGLE DURE : une fiche dont ``adherence_passed`` est False ne peut pas être
servie en production — le registre lève une exception.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import yaml

from compass.config import settings
from compass.schemas import VariableSheet

logger = logging.getLogger(__name__)


class AdherenceError(RuntimeError):
    """Levée quand une fiche non qualifiée est demandée en production."""


class VPartyRegistry:
    """Charge, valide et sert les fiches de variables."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._dir = registry_dir or settings.registry_dir
        self._sheets: dict[str, VariableSheet] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self._dir.glob("*.yaml")):
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            sheet = VariableSheet(**raw)
            self._sheets[sheet.variable_id] = sheet
        logger.info("Registre : %d fiches chargées depuis %s", len(self._sheets), self._dir)

    # ------------------------------------------------------------------ service
    def get(self, variable_id: str, production: bool = True) -> VariableSheet:
        """Retourne une fiche ; refuse les fiches non qualifiées en production.

        Raises:
            KeyError: variable inconnue.
            AdherenceError: fiche non passée par la gate R-1 (production=True).
        """
        sheet = self._sheets[variable_id]
        if production and not sheet.adherence_passed:
            raise AdherenceError(
                f"{variable_id} : fiche non qualifiée (gate d'adhérence R-1 non "
                "passée). Exécuter check_adherence() et figer le résultat."
            )
        return sheet

    def list_ids(self) -> list[str]:
        return sorted(self._sheets)

    # ------------------------------------------------------------------ gate R-1
    def build_adherence_tests(self, variable_id: str, seed: int = 42) -> list[dict]:
        """Génère les tests comportementaux d'adhérence pour une fiche.

        Trois familles (d'après les échecs documentés par Halterman & Keith) :
            1. permutation : l'ordre des ancres de l'échelle est mélangé —
               le juge doit produire le même score ;
            2. paraphrase : la définition est reformulée — même score attendu ;
            3. cas limites : chaque critère d'inclusion/exclusion devient un
               mini-cas dont la réponse correcte est connue par construction.

        Returns:
            Liste de tests ``{kind, payload, expected}`` à exécuter contre les
            juges (C11) AVANT toute mise en production de la fiche. Le verdict
            (taux de réussite ≥ seuil) est consigné manuellement dans le YAML
            (``adherence_passed: true``) — décision humaine, tracée.
        """
        sheet = self._sheets[variable_id]
        rng = random.Random(seed)
        tests: list[dict] = []

        shuffled = list(sheet.scale.items())
        rng.shuffle(shuffled)
        tests.append({"kind": "scale_permutation", "payload": dict(shuffled),
                      "expected": "score identique à l'ordre canonique"})
        tests.append({"kind": "definition_paraphrase",
                      "payload": f"PARAPHRASER puis réutiliser : {sheet.definition}",
                      "expected": "score identique à la définition canonique"})
        for crit in sheet.inclusion_criteria:
            tests.append({"kind": "inclusion_probe", "payload": crit,
                          "expected": "la preuve correspondante DOIT être retenue"})
        for crit in sheet.exclusion_criteria:
            tests.append({"kind": "exclusion_probe", "payload": crit,
                          "expected": "la preuve correspondante DOIT être écartée"})
        return tests

