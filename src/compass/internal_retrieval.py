"""C06 — Retrieval interne : retrouver les preuves pertinentes pour UNE variable.

ÉTAT DE L'ART RÉUTILISÉ (choix vérifiés par recherche web le 2026-06-11,
cf. CHOIX_COMPOSANTS.md §2) :
    - Recherche dense : ChromaDB (déjà en place via C02/C03).
    - Recherche lexicale : rank-bm25 (BM25Okapi) — l'hybride dense+lexical est
      le standard documenté des surveys RAG (Gao et al. 2023), utile quand le
      style rhétorique varie (le lexical rattrape ce que le sémantique manque).
    - Re-ranking : BGE-reranker-v2-m3 — meilleur défaut qualité/latence/licence
      des comparatifs 2026 pour corpus multilingues ; remplace l'ancien
      mmarco-mMiniLM (2022). Alternative d'ablation : jina-reranker-v3.

CUSTOM (justifié) : la formulation des requêtes à partir de la fiche C05
(la variable dicte quoi chercher) — logique métier de quelques lignes.
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from compass.party_election_case import CaseFile
from compass.config import settings
from compass.schemas import VariableSheet

logger = logging.getLogger(__name__)


class InternalRetriever:
    """Sélectionne, dans le dossier, les passages pertinents pour une variable."""

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k
        self._reranker = CrossEncoder(settings.reranker_model)

    def retrieve(self, dossier: CaseFile, sheet: VariableSheet) -> list[dict]:
        """Retrieval hybride (BM25 + dense déjà fait en amont) puis re-ranking.

        Args:
            dossier: dossier parti×élection (C04) — le périmètre de recherche.
            sheet: fiche de la variable (C05) — dicte la requête.

        Returns:
            Passages triés par pertinence décroissante, avec score de re-ranking.
        """
        query = self._build_query(sheet)
        pool = (dossier.party_documents + dossier.party_trajectory
                + dossier.national_context)
        if not pool:
            return []

        # BM25 sur le pool (le dense a déjà présélectionné via C03)
        corpus_tokens = [p["text"].lower().split() for p in pool]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query.lower().split())
        candidates = [p for _, p in sorted(zip(scores, pool),
                                           key=lambda x: x[0], reverse=True)][:30]

        # Re-ranking cross-encoder : pertinence fine requête × passage
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self._reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: float(x[0]),
                        reverse=True)
        out = []
        for score, cand in ranked[: self._top_k]:
            cand = dict(cand)
            cand["relevance"] = float(score)
            out.append(cand)
        logger.info("Retrieval %s : %d passages retenus", sheet.variable_id, len(out))
        return out

    @staticmethod
    def _build_query(sheet: VariableSheet) -> str:
        """La requête vient de la fiche : question + définition + sources requises."""
        return " ".join([sheet.question, sheet.definition, " ".join(sheet.required_sources)])

