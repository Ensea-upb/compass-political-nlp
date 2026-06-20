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
import re
import sqlite3
from datetime import date
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight installs
    BM25Okapi = None

from compass.config import settings
from compass.schemas import Segment

logger = logging.getLogger(__name__)

_RERANKER = None
_RERANKER_UNAVAILABLE = False

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
                "election_id": s.meta.election_id or "",
                # Gap 1 — chunking hiérarchique : lien enfant → parent
                "parent_segment_id": s.parent_segment_id or "",
                "segment_level": "child" if s.parent_segment_id else "parent",
                "chunk_index": s.chunk_index,
                "paragraph_start": s.paragraph_start if s.paragraph_start is not None else -1,
                "paragraph_end": s.paragraph_end if s.paragraph_end is not None else -1,
                "section_title": s.section_title or "",
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

    def fetch_records_by_ids(self, segment_ids: list[str]) -> list[dict]:
        """Récupère texte et métadonnées de segments exacts.

        Utilisé par COMPASS Chat pour afficher un lookup direct avec les mêmes
        métadonnées lisibles que le retrieval normal.
        """
        if not segment_ids:
            return []
        res = self._col.get(ids=segment_ids, include=["documents", "metadatas"])
        metadatas = res.get("metadatas") or [{} for _ in res["ids"]]
        return [
            {"segment_id": sid, "text": doc, "meta": meta or {}}
            for sid, doc, meta in zip(res["ids"], res["documents"], metadatas)
        ]

    def describe_corpus(
        self,
        as_of: date | None = None,
        party_id: str | None = None,
    ) -> dict:
        """Return a runtime profile of documents actually present in the index.

        The chat uses this method for scope answers and warnings. No country,
        party, document count, or date is inferred from demo configuration.
        """
        clauses: list[dict] = []
        if as_of is not None:
            clauses.append({"doc_date_ord": {"$lte": as_of.toordinal()}})
        if party_id:
            clauses.append({"party_id": party_id})
        where = None
        if clauses:
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        kwargs = {"include": ["metadatas"]}
        if where is not None:
            kwargs["where"] = where
        try:
            result = self._col.get(**kwargs)
        except Exception:
            result = self._col.get(include=["metadatas"])

        records: list[tuple[str, dict]] = []
        for segment_id, metadata in zip(
            result.get("ids") or [],
            result.get("metadatas") or [],
        ):
            meta = metadata or {}
            if as_of is not None and str(meta.get("doc_date") or "") > as_of.isoformat():
                continue
            if party_id and str(meta.get("party_id") or "") != party_id:
                continue
            records.append((str(segment_id), meta))

        party_ids = sorted({
            str(meta.get("party_id") or "").strip()
            for _, meta in records
            if str(meta.get("party_id") or "").strip()
        })
        party_names: dict[str, str] = {}
        if party_ids:
            placeholders = ",".join("?" for _ in party_ids)
            rows = self._conn.execute(
                f"SELECT party_id, name FROM parties WHERE party_id IN ({placeholders})",
                tuple(party_ids),
            ).fetchall()
            party_names = {str(row[0]): str(row[1]) for row in rows if row[1]}

        doc_ids = {
            str(meta.get("doc_id") or "").strip() or segment_id.split(":", 1)[0]
            for segment_id, meta in records
        }
        return {
            "country_iso3": self.country,
            "n_documents": len(doc_ids),
            "parties": [
                {"party_id": value, "name": party_names.get(value)}
                for value in party_ids
            ],
            "document_dates": sorted({
                str(meta.get("doc_date")) for _, meta in records if meta.get("doc_date")
            }),
            "document_types": sorted({
                str(meta.get("doc_type")) for _, meta in records if meta.get("doc_type")
            }),
        }

    def list_document_records(
        self,
        party_id: str | None = None,
        parent_only: bool = False,
    ) -> list[dict]:
        """List indexed documentary records for maintenance and graph backfills."""
        clauses: list[dict] = []
        if party_id:
            clauses.append({"party_id": party_id})
        if parent_only:
            clauses.append({"segment_level": "parent"})
        where = None
        if clauses:
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if where is not None:
            kwargs["where"] = where
        try:
            result = self._col.get(**kwargs)
        except Exception:
            # Indexes predating hierarchical chunk metadata have no
            # ``segment_level`` field. Return their records and let the graph
            # treat them as flat source segments.
            fallback_where = {"party_id": party_id} if party_id else None
            fallback_kwargs: dict = {"include": ["documents", "metadatas"]}
            if fallback_where is not None:
                fallback_kwargs["where"] = fallback_where
            result = self._col.get(**fallback_kwargs)
        metadatas = result.get("metadatas") or [{} for _ in result.get("ids", [])]
        return [
            {"segment_id": sid, "text": document, "meta": metadata or {}}
            for sid, document, metadata in zip(
                result.get("ids", []),
                result.get("documents", []),
                metadatas,
            )
        ]

    def query_documents(
        self, question: str, as_of: date, k: int = 12,
        party_id: str | None = None, include_unverified: bool = False,
        include_parent_segments: bool = False,
    ) -> list[dict]:
        """Recherche documentaire SOUS CONTRAINTE TEMPORELLE STRICTE.

        Args:
            as_of: borne temporelle obligatoire (date de l'élection).
            include_unverified: True UNIQUEMENT pour orienter une enquête
                (C08) — jamais pour produire des preuves (P0-2).
            include_parent_segments: True uniquement pour diagnostics techniques.
                Le retrieval normal interroge les enfants et récupère les parents
                ensuite par ``fetch_by_ids`` pour éviter de citer des blocs longs
                comme preuves directes.
        """
        clauses: list[dict] = [{"doc_date_ord": {"$lte": as_of.toordinal()}}]
        if not include_unverified:
            clauses.append({"temporal_ok": 1})
        if party_id:
            clauses.append({"party_id": party_id})
        if not include_parent_segments:
            clauses.append({"segment_level": "child"})
        res = self._query_chroma(question, k, clauses)
        if not _has_results(res) and not include_parent_segments:
            # Backward compatibility with indexes created before segment_level.
            fallback = [c for c in clauses if "segment_level" not in c]
            res = self._query_chroma(question, k, fallback)
        return [
            {"segment_id": i, "text": d, "meta": m}
            for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
        ]

    def query_documents_hybrid(
        self, question: str, as_of: date, k: int = 12,
        party_id: str | None = None, include_unverified: bool = False,
        include_parent_segments: bool = False,
    ) -> list[dict]:
        """Dense + BM25 retrieval followed by optional cross-encoder reranking.

        This is the default retrieval shape expected by COMPASS Chat: dense
        semantic ordering gives recall, BM25 reinforces exact political terms.
        The final ranking is delegated to a cross-encoder over
        ``question`` x ``parent_context + child_segment`` so the model sees
        the local manifesto section without turning parent blocks into cited
        evidence.
        """
        rerank_pool_size = max(k, settings.rerank_pool_size, k * 3)
        clauses = self._document_clauses(
            as_of=as_of,
            party_id=party_id,
            include_unverified=include_unverified,
            include_parent_segments=include_parent_segments,
        )
        dense = self.query_documents(
            question,
            as_of=as_of,
            k=max(rerank_pool_size, k * 4, 24),
            party_id=party_id,
            include_unverified=include_unverified,
            include_parent_segments=include_parent_segments,
        )
        lexical_pool = self._get_records(clauses, limit=max(rerank_pool_size * 8, 200))
        if not lexical_pool:
            return _cross_encoder_rerank(question, self._inject_parent_text(dense), k)
        candidates = _dedupe_records(dense + lexical_pool)
        hybrid_pool = _hybrid_rank(question, candidates, dense, k=rerank_pool_size)
        return _cross_encoder_rerank(question, self._inject_parent_text(hybrid_pool), k)

    def query_parent_documents_hybrid(
        self,
        question: str,
        as_of: date,
        k: int = 6,
        party_id: str | None = None,
        include_unverified: bool = False,
    ) -> list[dict]:
        """Hybrid retrieval restricted to parent chunks for general context."""
        clauses = self._document_clauses(
            as_of=as_of,
            party_id=party_id,
            include_unverified=include_unverified,
            include_parent_segments=True,
        )
        clauses.append({"segment_level": "parent"})
        dense_result = self._query_chroma(question, max(k * 4, 16), clauses)
        dense = [
            {"segment_id": sid, "text": document, "meta": metadata or {}}
            for sid, document, metadata in zip(
                dense_result.get("ids", [[]])[0],
                dense_result.get("documents", [[]])[0],
                dense_result.get("metadatas", [[]])[0],
            )
        ] if _has_results(dense_result) else []
        lexical_pool = self._get_records(clauses, limit=max(k * 16, 80))
        candidates = _dedupe_records(dense + lexical_pool)
        hybrid_pool = _hybrid_rank(question, candidates, dense, k=max(k * 3, 12))
        return _cross_encoder_rerank(question, hybrid_pool, k)

    def _query_chroma(self, question: str, k: int, clauses: list[dict]) -> dict:
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        return self._col.query(query_texts=[question], n_results=k, where=where)

    def _document_clauses(
        self,
        *,
        as_of: date,
        party_id: str | None,
        include_unverified: bool,
        include_parent_segments: bool,
    ) -> list[dict]:
        clauses: list[dict] = [{"doc_date_ord": {"$lte": as_of.toordinal()}}]
        if not include_unverified:
            clauses.append({"temporal_ok": 1})
        if party_id:
            clauses.append({"party_id": party_id})
        if not include_parent_segments:
            clauses.append({"segment_level": "child"})
        return clauses

    def _get_records(self, clauses: list[dict], limit: int) -> list[dict]:
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        try:
            res = self._col.get(where=where, include=["documents", "metadatas"], limit=limit)
        except TypeError:
            try:
                res = self._col.get(where=where, include=["documents", "metadatas"])
            except Exception:
                return []
        except Exception:
            return []
        metadatas = res.get("metadatas") or [{} for _ in res.get("ids", [])]
        return [
            {"segment_id": sid, "text": doc, "meta": meta or {}}
            for sid, doc, meta in zip(res.get("ids", []), res.get("documents", []), metadatas)
        ]

    def _inject_parent_text(self, records: list[dict]) -> list[dict]:
        """Attach parent manifesto context to child records before reranking."""
        parent_ids = sorted({
            str((record.get("meta") or {}).get("parent_segment_id") or "")
            for record in records
            if (record.get("meta") or {}).get("parent_segment_id")
        })
        if not parent_ids:
            return records
        parent_records = {
            record["segment_id"]: record
            for record in self.fetch_records_by_ids(parent_ids)
        }
        out: list[dict] = []
        for record in records:
            enriched = dict(record)
            meta = record.get("meta") or {}
            parent_id = str(meta.get("parent_segment_id") or "")
            parent = parent_records.get(parent_id)
            if parent and _same_document_scope(meta, parent.get("meta") or {}):
                enriched["parent_text"] = parent.get("text") or ""
            out.append(enriched)
        return out


def _has_results(result: dict) -> bool:
    ids = result.get("ids") or []
    return bool(ids and ids[0])


def _hybrid_rank(question: str, candidates: list[dict], dense: list[dict], k: int) -> list[dict]:
    query_tokens = _tokens(question)
    dense_rank = {item["segment_id"]: rank for rank, item in enumerate(dense, start=1)}
    bm25_rank: dict[str, int] = {}
    if query_tokens and candidates and BM25Okapi is not None:
        scores = BM25Okapi([_tokens(item.get("text", "")) for item in candidates]).get_scores(query_tokens)
        order = sorted(range(len(candidates)), key=lambda idx: float(scores[idx]), reverse=True)
        bm25_rank = {candidates[idx]["segment_id"]: rank for rank, idx in enumerate(order, start=1)}
    elif query_tokens and candidates:
        scores = [_lexical_overlap(query_tokens, item.get("text", "")) for item in candidates]
        order = sorted(range(len(candidates)), key=lambda idx: float(scores[idx]), reverse=True)
        bm25_rank = {candidates[idx]["segment_id"]: rank for rank, idx in enumerate(order, start=1)}
    ranked = []
    for item in candidates:
        sid = item["segment_id"]
        score = 0.0
        if sid in dense_rank:
            score += 1.0 / (60 + dense_rank[sid])
        if sid in bm25_rank:
            score += 1.25 / (60 + bm25_rank[sid])
        enriched = dict(item)
        enriched["hybrid_score"] = score
        enriched["dense_rank"] = dense_rank.get(sid)
        enriched["bm25_rank"] = bm25_rank.get(sid)
        enriched["retrieval_reason"] = _retrieval_reason(
            dense_rank.get(sid),
            bm25_rank.get(sid),
        )
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: float(item.get("hybrid_score") or 0.0), reverse=True)[:k]


def _cross_encoder_rerank(question: str, records: list[dict], k: int) -> list[dict]:
    if not getattr(settings, "rerank_enabled", True) or len(records) <= 1:
        return records[:k]
    try:
        scores = _cross_encoder_scores(question, records)
    except Exception as exc:  # pragma: no cover - depends on local model availability
        logger.warning("Cross-encoder reranking disabled for this query: %s", exc)
        return records[:k]

    ranked: list[dict] = []
    for record, score in zip(records, scores):
        enriched = dict(record)
        enriched["rerank_score"] = float(score)
        enriched["retrieval_reason"] = _append_reason(
            str(enriched.get("retrieval_reason") or ""),
            f"cross_encoder_score={float(score):.4f}",
        )
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)[:k]


def _cross_encoder_scores(question: str, records: list[dict]):
    reranker = _get_reranker()
    pairs = [(question, _rerank_text(record)) for record in records]
    return reranker.predict(pairs)


def _get_reranker():
    global _RERANKER, _RERANKER_UNAVAILABLE
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_UNAVAILABLE:
        raise RuntimeError("cross-encoder unavailable")
    try:
        from sentence_transformers import CrossEncoder

        kwargs = {"device": settings.hf_model_device()} if settings.hf_model_device() else {}
        _RERANKER = CrossEncoder(settings.reranker_model, **kwargs)
        return _RERANKER
    except Exception as exc:
        _RERANKER_UNAVAILABLE = True
        raise RuntimeError(f"could not load cross-encoder {settings.reranker_model}: {exc}") from exc


def _rerank_text(record: dict) -> str:
    parent = str(record.get("parent_text") or "").strip()
    child = str(record.get("text") or "").strip()
    if parent and child and child not in parent:
        return f"{parent}\n\nEvidence segment:\n{child}"
    return parent or child


def _append_reason(existing: str, reason: str) -> str:
    if not existing:
        return reason
    return f"{existing} | {reason}"


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for record in records:
        sid = str(record.get("segment_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(record)
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def _lexical_overlap(query_tokens: list[str], text: str) -> float:
    text_tokens = set(_tokens(text))
    if not text_tokens:
        return 0.0
    return float(sum(1 for token in query_tokens if token in text_tokens))


def _retrieval_reason(dense_rank: int | None, bm25_rank: int | None) -> str:
    parts: list[str] = []
    if dense_rank is not None:
        parts.append(f"dense_rank={dense_rank}")
    if bm25_rank is not None:
        parts.append(f"bm25_rank={bm25_rank}")
    return " | ".join(parts)


def _same_document_scope(child: dict, parent: dict) -> bool:
    """Prevent a malformed parent link from crossing corpus boundaries."""
    for key in ("doc_id", "country_iso3", "party_id", "election_id", "language"):
        child_value = str(child.get(key) or "")
        parent_value = str(parent.get(key) or "")
        if child_value and parent_value and child_value != parent_value:
            return False
    return str(parent.get("segment_level") or "parent") == "parent"

