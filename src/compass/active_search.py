"""C08 — Moteur de recherche active : enquêter quand la mémoire ne suffit pas.

ÉTAT DE L'ART RÉUTILISÉ (choix vérifiés par recherche web le 2026-06-11,
cf. CHOIX_COMPOSANTS.md §5) :
    - Recherche web : ddgs (DuckDuckGo Search, sans clé) par défaut ; Tavily
      (API dédiée aux agents, clé via variable d'environnement) si disponible.
      Patron « agent qui enquête » : WebGPT (Nakano et al. 2021), ReAct
      (Yao et al. 2023) — on reprend la boucle requête → lecture → ingestion.
    - Ingestion ET DATATION des pages : via C01 (trafilatura + htmldate) —
      la date de publication réelle est extraite à l'ingestion ; une page
      sans date vérifiable est marquée ``_undated`` et pénalisée au
      diagnostic, jamais présumée antérieure à l'élection.

CUSTOM (justifié) : le contrôle — budget de requêtes, typage de fiabilité des
domaines (C15). La littérature fournit le patron, pas la discipline ; or la
recherche active non bornée détruit coût et reproductibilité (audit, AM-10).
"""

from __future__ import annotations

import logging
from datetime import date

from ddgs import DDGS

from compass.document_pipeline import DocumentPipeline, make_meta
from compass.config import settings
from compass.schemas import CaseKey, Segment, SourceReliability, VariableSheet

logger = logging.getLogger(__name__)

# Typage de fiabilité par domaine — table à enrichir pays par pays (C15).
_DOMAIN_RELIABILITY: dict[str, SourceReliability] = {
    "v-dem.net": SourceReliability.ACADEMIC,
    "jstor.org": SourceReliability.ACADEMIC,
    "eisa.org": SourceReliability.OFFICIAL,        # observation électorale
    "eces.eu": SourceReliability.OFFICIAL,
    "jeuneafrique.com": SourceReliability.INDEPENDENT_PRESS,
    "rfi.fr": SourceReliability.INDEPENDENT_PRESS,
}


class ActiveSearchEngine:
    """Recherche web ciblée, bornée, datée et typée en fiabilité."""

    def __init__(self, pipeline: DocumentPipeline) -> None:
        self._pipeline = pipeline
        self._queries_used = 0

    def investigate(self, case: CaseKey, sheet: VariableSheet,
                    missing: list[str]) -> list[Segment]:
        """Cherche des preuves pour les manques identifiés par le diagnostic.

        Args:
            case: unité pays × parti × élection (borne temporelle).
            sheet: fiche de la variable (oriente les requêtes).
            missing: manques explicites (sortie C07/C09).

        Returns:
            Nouveaux segments ingérés via C01 — datés par htmldate, hashés, typés.
        """
        new_segments: list[Segment] = []
        for gap in missing:
            if self._queries_used >= settings.search_max_queries:
                logger.warning("Budget de requêtes épuisé (%d) — arrêt.",
                               settings.search_max_queries)
                break
            query = f"{case.country_iso3} {case.party_id} {gap} avant {case.election_date.year}"
            self._queries_used += 1
            results = list(DDGS().text(query, max_results=5))
            for res in results:
                url = res.get("href", "")
                try:
                    meta = make_meta(
                        country_iso3=case.country_iso3,
                        doc_date=self._bootstrap_date(case.election_date),
                        doc_type="web_actif",
                        party_id=case.party_id,
                        election_id=case.election_id,
                        reliability=self._classify_domain(url),
                    )
                    segs = self._pipeline.ingest_url(url, meta)
                except (ValueError, OSError) as exc:
                    logger.info("Page écartée (%s) : %s", url, exc)
                    continue
                new_segments.extend(segs)
        logger.info("Recherche active : %d segments ingérés (%d requêtes utilisées)",
                    len(new_segments), self._queries_used)
        return new_segments

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _classify_domain(url: str) -> SourceReliability:
        """P1-8 : appariement sur l'hôte exact (ou sous-domaine), jamais par
        inclusion de chaîne — 'rfi.fr.attaquant.com' ne doit pas passer."""
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        for domain, rel in _DOMAIN_RELIABILITY.items():
            if host == domain or host.endswith("." + domain):
                return rel
        return SourceReliability.UNKNOWN

    @staticmethod
    def _bootstrap_date(election_date: date) -> date:
        """Date d'amorçage passée à C01 — qui la REMPLACE par la date htmldate.

        La datation réelle se fait à l'ingestion (C01) : si htmldate trouve une
        date de publication, elle écrase cette valeur ; sinon le document est
        marqué ``_undated`` et pénalisé par le diagnostic. Cette valeur ne sert
        donc jamais de preuve d'antériorité.
        """
        return election_date

