"""C03 — Mémoire pays hybride et historisée.

Deux couches : STRUCTURÉE (SQLite) + DOCUMENTAIRE (ChromaDB, datée).

Révision post-audit (2026-06-12) :
    - P0-1 : le filtre temporel Chroma utilise désormais un ENTIER
      (``doc_date_ord`` = date.toordinal()) — la documentation officielle
      Chroma ne garantit les opérateurs $lte/$gte que sur des nombres.
      La chaîne ISO est conservée pour la lecture humaine uniquement.
    - P0-2 : seuls les documents ``eligible_for_historical_reasoning`` (statut
      temporel VERIFIED) passent le filtre historique ; les documents UNKNOWN
      sont accessibles uniquement via ``query_documents(..., include_unverified=True)``
      pour ORIENTER une enquête, jamais comme preuve.
    - P0-3 : la table ``vparty_scores`` (étalon) est RETIRÉE de cette base de
      production — elle vit dans le vault d'évaluation (c14_validation).
      La production gagne une table ``compass_scores`` : les sorties du
      système lui-même, seules entrées admissibles des formules dérivées (C10).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from compass.config import settings
from compass.schemas import Segment

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS elections (
    election_id TEXT PRIMARY KEY, country_iso3 TEXT NOT NULL,
    election_date TEXT NOT NULL, election_type TEXT
);
CREATE TABLE IF NOT EXISTS parties (
    party_id TEXT PRIMARY KEY, country_iso3 TEXT NOT NULL,
    name TEXT NOT NULL, pf_id TEXT, founded TEXT, dissolved TEXT
);
CREATE TABLE IF NOT EXISTS results (
    election_id TEXT NOT NULL, party_id TEXT NOT NULL,
    vote_share REAL, seats INTEGER, seats_total INTEGER,
    PRIMARY KEY (election_id, party_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY, country_iso3 TEXT NOT NULL,
    event_date TEXT NOT NULL, party_id TEXT, event_type TEXT, description TEXT
);
-- P0-3 : sorties produites par COMPASS lui-même (jamais l'étalon V-Party).
CREATE TABLE IF NOT EXISTS compass_scores (
    party_id TEXT NOT NULL, election_id TEXT NOT NULL,
    variable_id TEXT NOT NULL, score REAL, confidence REAL,
    PRIMARY KEY (party_id, election_id, variable_id)
);
"""


class CountryMemory:
    """Mémoire d'un pays : couche structurée (SQL) + couche documentaire (vecteurs)."""

    def __init__(self, country_iso3: str) -> None:
        self.country = country_iso3.upper()
        self._conn = sqlite3.connect(settings.sqlite_path)
        self._conn.executescript(_SCHEMA)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        embed = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model, **({"device": settings.hf_model_device()} if settings.hf_model_device() else {}))
        self._col = client.get_or_create_collection(
            name=f"compass_country_{self.country.lower()}", embedding_function=embed
        )

    # ---------------------------------------------------------------- structurée
    def import_csv(self, table: str, csv_path: Path, column_map: dict[str, str]) -> int:
        """Importe une base existante (résultats, partis, événements) — REUSE_DIRECT.

        N'accepte PAS la table vparty_scores : l'étalon va dans le vault (C14).

        Raises:
            ValueError: si l'on tente d'importer l'étalon dans la production.
        """
        if table == "vparty_scores":
            raise ValueError(
                "P0-3 : les scores V-Party sont l'ÉTALON — ils vont dans le "
                "vault d'évaluation (c14_validation.EvaluationVault), jamais "
                "dans la mémoire de production."
            )
        df = pd.read_csv(csv_path).rename(columns=column_map)
        df = df[[c for c in df.columns if c in self._table_columns(table)]]
        df.to_sql(table, self._conn, if_exists="append", index=False)
        logger.info("Import %s : %d lignes depuis %s", table, len(df), csv_path.name)
        return len(df)

    def store_compass_score(self, party_id: str, election_id: str,
                            variable_id: str, score: float, confidence: float) -> None:
        """Enregistre une sortie validée du système (entrée des indices dérivés)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO compass_scores VALUES (?,?,?,?,?)",
            (party_id, election_id, variable_id, score, confidence),
        )
        self._conn.commit()

    def query_structured(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Lecture SQL libre — utilisée par le moteur C10 (STRUCTURED_QUERY)."""
        return pd.read_sql_query(sql, self._conn, params=params)

    def _table_columns(self, table: str) -> list[str]:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

    # ---------------------------------------------------------------- documentaire
    def add_documents(self, segments: list[Segment]) -> None:
        """Indexe des segments documentaires datés pour ce pays."""
        mismatched = [s for s in segments if s.meta.country_iso3.upper() != self.country]
        if mismatched:
            raise ValueError(f"{len(mismatched)} segments d'un autre pays — refusés.")
        self._col.upsert(
            ids=[s.segment_id for s in segments],
            documents=[s.text for s in segments],
            metadatas=[{
                "doc_date": s.meta.doc_date.isoformat(),          # lecture humaine
                "doc_date_ord": s.meta.doc_date.toordinal(),      # P0-1 : filtre numérique
                "temporal_ok": 1 if s.meta.eligible_for_historical_reasoning else 0,  # P0-2
                "temporal_status": s.meta.temporal_status.value,
                "doc_type": s.meta.doc_type,
                "party_id": s.meta.party_id or "",
                "reliability": s.meta.reliability.value,
                "language": s.meta.language,
                "doc_id": s.doc_id,
                "country_iso3": s.meta.country_iso3,
                # Gap 1 — chunking hiérarchique : lien enfant → parent
                "parent_segment_id": s.parent_segment_id or "",
            } for s in segments],
        )

    def fetch_by_ids(self, segment_ids: list[str]) -> dict[str, str]:
        """Récupère le texte brut de segments par leurs IDs — utilisé par C06
        pour injecter le texte parent lors du re-ranking (Gap 1).

        Args:
            segment_ids: liste d'IDs à récupérer (peut contenir des IDs absents).

        Returns:
            Dictionnaire {segment_id: text} pour les IDs trouvés.
        """
        if not segment_ids:
            return {}
        res = self._col.get(ids=segment_ids, include=["documents"])
        return {sid: doc for sid, doc in zip(res["ids"], res["documents"])}

    def query_documents(
        self, question: str, as_of: date, k: int = 12,
        party_id: str | None = None, include_unverified: bool = False,
    ) -> list[dict]:
        """Recherche documentaire SOUS CONTRAINTE TEMPORELLE STRICTE.

        Args:
            as_of: borne temporelle obligatoire (date de l'élection).
            include_unverified: True UNIQUEMENT pour orienter une enquête
                (C08) — jamais pour produire des preuves (P0-2).
        """
        clauses: list[dict] = [{"doc_date_ord": {"$lte": as_of.toordinal()}}]
        if not include_unverified:
            clauses.append({"temporal_ok": 1})
        if party_id:
            clauses.append({"party_id": party_id})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._col.query(query_texts=[question], n_results=k, where=where)
        return [
            {"segment_id": i, "text": d, "meta": m}
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
        ]

