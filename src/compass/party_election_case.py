"""C04 — Dossier parti × élection : le cas instruit.

ÉTAT DE L'ART RÉUTILISÉ : aucun outil dédié n'existe pour « constituer un dossier
parti-élection » — mais le composant n'invente rien : il ORCHESTRE C02 et C03
(qui, eux, reposent sur ChromaDB/SQLite). C'est le cas assumé de « custom
nécessaire » : pure logique métier, zéro algorithmique réinventée.

CONTENU DU DOSSIER (architecture, bloc 4) : contexte national pré-électoral,
trajectoire récente du parti, programme, discours, presse, comportements
observables, données structurées pertinentes — borné par la date de l'élection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from compass.general_memory import GeneralMemory
from compass.country_memory import CountryMemory
from compass.schemas import CaseKey

if TYPE_CHECKING:
    from compass.political_graph import PoliticalGraph

logger = logging.getLogger(__name__)


@dataclass
class CaseFile:
    """Dossier instruit pour une unité pays × parti × élection."""

    case: CaseKey
    national_context: list[dict] = field(default_factory=list)
    party_trajectory: list[dict] = field(default_factory=list)
    party_documents: list[dict] = field(default_factory=list)
    structured_facts: dict[str, pd.DataFrame] = field(default_factory=dict)
    general_background: list[dict] = field(default_factory=list)
    # Gap 3 — Graphe de connaissances politiques (C02b) :
    # résumés relationnels du voisinage du parti, filtrés temporellement.
    graph_context: list[dict] = field(default_factory=list)


class CaseFileBuilder:
    """Assemble un ``CaseFile`` depuis les mémoires, sous contrainte temporelle."""

    def __init__(self, general: GeneralMemory, country: CountryMemory,
                 context_window_years: int = 5,
                 graph: "PoliticalGraph | None" = None) -> None:
        self._general = general
        self._country = country
        self._window = timedelta(days=365 * context_window_years)
        self._graph = graph  # optionnel : non disponible = pas de graph_context

    def build(self, case: CaseKey) -> CaseFile:
        """Construit le dossier — uniquement des éléments antérieurs à l'élection.

        Args:
            case: identifiant pays × parti × élection.

        Returns:
            Dossier prêt pour le retrieval ciblé (C06).
        """
        as_of = case.election_date
        dossier = CaseFile(case=case)

        dossier.national_context = self._country.query_documents(
            question="situation politique nationale, contexte pré-électoral, crise, "
                     "réforme constitutionnelle, sécurité",
            as_of=as_of, k=10,
        )
        dossier.party_trajectory = self._country.query_documents(
            question="histoire du parti, coalitions, scissions, changement de dirigeant",
            as_of=as_of, k=10, party_id=case.party_id,
        )
        dossier.party_documents = self._country.query_documents(
            question="programme électoral, discours de campagne, communiqués",
            as_of=as_of, k=20, party_id=case.party_id,
        )
        dossier.structured_facts = {
            "results": self._country.query_structured(
                "SELECT r.* FROM results r JOIN elections e USING(election_id) "
                "WHERE r.party_id = ? AND e.election_date < ?",
                (case.party_id, as_of.isoformat()),
            ),
            "events": self._country.query_structured(
                "SELECT * FROM events WHERE party_id = ? AND event_date < ? "
                "ORDER BY event_date DESC",
                (case.party_id, as_of.isoformat()),
            ),
        }
        dossier.general_background = self._general.query(
            "grilles d'analyse des partis : pluralisme, clientélisme, organisation", k=6
        )
        # --- Gap 3 : contexte relationnel depuis le graphe de connaissances ---
        if self._graph is not None:
            dossier.graph_context = self._graph.query_party(
                party_id=case.party_id,
                as_of=as_of,
                k_hops=2,
                top_k=10,
            )
            logger.debug(
                "Graph context %s/%s : %d résumés relationnels",
                case.party_id, case.election_id, len(dossier.graph_context),
            )

        logger.info(
            "Dossier %s/%s/%s : %d docs parti, %d contexte",
            case.country_iso3, case.party_id, case.election_id,
            len(dossier.party_documents), len(dossier.national_context),
        )
        return dossier

