"""Expert multi-lane retrieval orchestration for COMPASS Chat."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from compass.chat.query_analysis import QueryAnalysis
from compass.config import settings

QueryFunction = Callable[..., list[dict[str, Any]]]


@dataclass
class RetrievalBundle:
    primary: list[dict[str, Any]] = field(default_factory=list)
    nuances: list[dict[str, Any]] = field(default_factory=list)
    counter: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_candidates: int = 0
    sufficiency: float = 0.0

    def prompt_records(self, limit: int) -> list[dict[str, Any]]:
        """Allocate prompt evidence across lanes, then fill remaining slots."""
        if limit <= 0:
            return []
        quotas = {
            "primary": max(1, limit - 2),
            "nuance": 1 if limit >= 3 else 0,
            "counter": 1 if limit >= 4 else 0,
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        lanes = (
            ("primary", self.primary),
            ("nuance", self.nuances),
            ("counter", self.counter),
        )
        for role, records in lanes:
            for record in records[:quotas[role]]:
                _append_unique(selected, seen, record)
        for _, records in lanes:
            for record in records:
                if len(selected) >= limit:
                    break
                _append_unique(selected, seen, record)
        return selected[:limit]


def retrieve_expert(
    memory: Any,
    analysis: QueryAnalysis,
    *,
    query_function: QueryFunction,
    as_of: date,
    country_iso3: str,
    party_id: str | None,
    election_id: str | None,
    include_unverified: bool,
    k: int,
) -> RetrievalBundle:
    """Run primary, nuance and counter-evidence retrieval under strict scope."""
    lane_queries = build_lane_queries(analysis)
    trace: list[dict[str, Any]] = [{
        "stage": "scope",
        "country_iso3": country_iso3,
        "party_id": party_id,
        "election_id": election_id,
        "as_of": as_of.isoformat(),
        "include_unverified": include_unverified,
    }]
    excluded_ids: set[str] = set()
    lane_outputs: dict[str, list[dict[str, Any]]] = {}
    total_candidates = 0
    for lane in ("primary", "nuance", "counter"):
        records, lane_trace, candidate_count = _retrieve_lane(
            memory,
            lane_queries[lane],
            lane=lane,
            query_function=query_function,
            as_of=as_of,
            country_iso3=country_iso3,
            party_id=party_id,
            election_id=election_id,
            include_unverified=include_unverified,
            k=max(k, settings.chat_retrieval_lane_k),
            excluded_ids=excluded_ids,
        )
        lane_outputs[lane] = records
        excluded_ids.update(str(record.get("segment_id")) for record in records)
        total_candidates += candidate_count
        trace.extend(lane_trace)

    primary = lane_outputs["primary"]
    query_coverage = {
        index
        for record in primary
        for index in record.get("matched_query_indices", [])
    }
    coverage = len(query_coverage) / max(1, len(lane_queries["primary"]))
    volume = min(1.0, len(primary) / 2.0)
    sufficiency = round(0.6 * coverage + 0.4 * volume, 4)
    trace.append({
        "stage": "sufficiency",
        "primary_count": len(primary),
        "primary_query_coverage": round(coverage, 4),
        "score": sufficiency,
        "threshold": settings.chat_retrieval_min_sufficiency,
    })
    return RetrievalBundle(
        primary=primary,
        nuances=lane_outputs["nuance"],
        counter=lane_outputs["counter"],
        trace=trace,
        total_candidates=total_candidates,
        sufficiency=sufficiency,
    )


def build_lane_queries(analysis: QueryAnalysis) -> dict[str, list[str]]:
    focus = " ".join([*analysis.actors, *analysis.themes]).strip()
    if analysis.language == "fr":
        nuance = f"{focus} conditions limites exceptions nuances ambiguïtés".strip()
        counter = f"{focus} opposition rejet critique contradiction cependant".strip()
    else:
        nuance = f"{focus} conditions limits exceptions qualifications ambiguity".strip()
        counter = f"{focus} opposition rejection criticism contradiction however".strip()
    return {
        "primary": analysis.subqueries,
        "nuance": [nuance] if focus else analysis.subqueries[-1:],
        "counter": [counter] if focus else analysis.subqueries[-1:],
    }


def _retrieve_lane(
    memory: Any,
    queries: list[str],
    *,
    lane: str,
    query_function: QueryFunction,
    as_of: date,
    country_iso3: str,
    party_id: str | None,
    election_id: str | None,
    include_unverified: bool,
    k: int,
    excluded_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    matches: dict[str, list[int]] = {}
    trace: list[dict[str, Any]] = []
    raw_count = 0
    rejected_scope = 0
    for query_index, query in enumerate(queries, start=1):
        records = query_function(
            memory,
            query,
            as_of=as_of,
            k=k,
            party_id=party_id,
            include_unverified=include_unverified,
        )
        raw_count += len(records)
        accepted = 0
        for rank, record in enumerate(records, start=1):
            if not record_in_scope(
                record,
                as_of=as_of,
                country_iso3=country_iso3,
                party_id=party_id,
                election_id=election_id,
                include_unverified=include_unverified,
            ):
                rejected_scope += 1
                continue
            segment_id = str(record.get("segment_id") or "")
            if not segment_id or segment_id in excluded_ids:
                continue
            accepted += 1
            current = fused.get(segment_id)
            if current is None or _record_quality(record) > _record_quality(current):
                fused[segment_id] = dict(record)
            scores[segment_id] = scores.get(segment_id, 0.0) + 1.0 / (60 + rank)
            matches.setdefault(segment_id, []).append(query_index)
        trace.append({
            "stage": "query",
            "lane": lane,
            "query_index": query_index,
            "query": query,
            "retrieved": len(records),
            "accepted_in_scope": accepted,
        })

    ranked: list[dict[str, Any]] = []
    for segment_id in sorted(fused, key=lambda sid: (-scores[sid], sid)):
        record = dict(fused[segment_id])
        record["evidence_role"] = lane
        record["multi_query_score"] = scores[segment_id]
        record["matched_query_indices"] = sorted(set(matches[segment_id]))
        existing_reason = str(record.get("retrieval_reason") or "")
        rrf_reason = (
            f"lane={lane} | multi_query_rrf={scores[segment_id]:.6f} | "
            f"matched_queries={','.join(str(value) for value in record['matched_query_indices'])}"
        )
        record["retrieval_reason"] = f"{existing_reason} | {rrf_reason}".strip(" |")
        ranked.append(record)
    diversified = _diversify(ranked, limit=k)
    trace.append({
        "stage": "lane_fusion",
        "lane": lane,
        "raw_candidates": raw_count,
        "scope_rejections": rejected_scope,
        "deduplicated": len(ranked),
        "selected_after_diversity": len(diversified),
        "selected_ids": [record.get("segment_id") for record in diversified],
    })
    return diversified, trace, len(ranked)


def record_in_scope(
    record: dict[str, Any],
    *,
    as_of: date,
    country_iso3: str,
    party_id: str | None,
    election_id: str | None,
    include_unverified: bool,
) -> bool:
    meta = record.get("meta") or {}
    if country_iso3 and str(meta.get("country_iso3") or "").upper() != country_iso3.upper():
        return False
    if party_id and str(meta.get("party_id") or "") != str(party_id):
        return False
    if election_id and settings.chat_strict_election_scope:
        if str(meta.get("election_id") or "") != str(election_id):
            return False
    doc_date = str(meta.get("doc_date") or "")
    if doc_date:
        try:
            if date.fromisoformat(doc_date[:10]) > as_of:
                return False
        except ValueError:
            return False
    if not include_unverified and meta.get("temporal_ok") in (0, "0", False):
        return False
    return True


def _diversify(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    parent_counts: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for record in records:
        parent_id = str((record.get("meta") or {}).get("parent_segment_id") or "")
        too_similar = any(
            _token_jaccard(str(record.get("text") or ""), str(item.get("text") or ""))
            >= settings.chat_retrieval_diversity_threshold
            for item in selected
        )
        same_parent = bool(parent_id and parent_counts.get(parent_id, 0) >= 1)
        if too_similar or same_parent:
            deferred.append(record)
            continue
        selected.append(record)
        if parent_id:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        if len(selected) >= limit:
            return selected
    for record in deferred:
        if len(selected) >= limit:
            break
        selected.append(record)
    return selected


def _record_quality(record: dict[str, Any]) -> float:
    return float(
        record.get("rerank_score")
        or record.get("hybrid_score")
        or record.get("multi_query_score")
        or 0.0
    )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[\w']+", left.casefold()))
    right_tokens = set(re.findall(r"[\w']+", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _append_unique(
    selected: list[dict[str, Any]],
    seen: set[str],
    record: dict[str, Any],
) -> None:
    segment_id = str(record.get("segment_id") or "")
    if segment_id and segment_id not in seen:
        seen.add(segment_id)
        selected.append(record)
