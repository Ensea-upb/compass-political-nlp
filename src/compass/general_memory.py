"""C02 — Mémoire générale comparative : les connaissances portables entre pays.

ÉTAT DE L'ART RÉUTILISÉ :
    - Stockage + recherche vectorielle : ChromaDB (persistant, métadonnées filtrables) —
      base vectorielle open source la plus utilisée pour le RAG (Lewis et al. 2020).
    - Embeddings : sentence-transformers multilingue (Reimers & Gurevych 2019) —
      le choix multilingue est imposé par la cible Afrique/Asie (Licht 2023).

CUSTOM (justifié) : rien — ce composant est une configuration de ChromaDB
(une collection dédiée 'general') avec un contrat d'API minimal. Le contenu
(théories, typologies, codebook V-Party) est versé via C01.

PRINCIPE D'ARCHITECTURE : cette mémoire ne contient AUCUN fait daté propre à un
pays. Comme ``DocumentMeta.country_iso3`` est obligatoire, les connaissances
portables utilisent la portée technique ``GEN`` (ou ``GLOBAL``), jamais un code
pays réel. Les faits pays vont dans C03.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from compass.config import settings
from compass.schemas import Segment

logger = logging.getLogger(__name__)


class GeneralMemory:
    """Collection vectorielle des connaissances comparatives générales."""

    COLLECTION = "compass_general"

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._embed = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model, **({"device": settings.hf_model_device()} if settings.hf_model_device() else {}))
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=self._embed
        )

    def add(self, segments: list[Segment]) -> None:
        """Indexe des segments de connaissance générale (théories, codebook…).

        Raises:
            ValueError: si un segment porte un pays — interdit dans cette mémoire.
        """
        for seg in segments:
            general_scope = str(seg.meta.country_iso3 or "").upper() in {"GEN", "GLOBAL"}
            if not general_scope or seg.meta.party_id is not None:
                raise ValueError(
                    f"Segment {seg.segment_id} lié à un pays ou un parti : il relève de la "
                    "mémoire pays (C03), pas de la mémoire générale."
                )
        self._col.upsert(
            ids=[s.segment_id for s in segments],
            documents=[s.text for s in segments],
            metadatas=[{"doc_type": s.meta.doc_type, "language": s.meta.language,
                        "doc_id": s.doc_id} for s in segments],
        )
        logger.info("Mémoire générale : %d segments indexés", len(segments))

    def query(self, question: str, k: int = 8) -> list[dict]:
        """Recherche les k passages généraux les plus pertinents."""
        res = self._col.query(query_texts=[question], n_results=k)
        return [
            {"segment_id": i, "text": d, "meta": m}
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
        ]

