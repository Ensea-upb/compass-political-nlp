"""C02b - Political knowledge graph for actors and relations.

Why this component exists:
    Vector memory is strong for local evidence retrieval, but weak for global
    relational questions such as how a party evolved toward an institution,
    coalition or foreign actor over time. This component adds a lightweight
    GraphRAG-style layer: named entities and inferred relations are extracted
    from dated segments, persisted as a graph, and queried under the same
    temporal discipline as country memory.

State of the art reused:
    - GraphRAG pattern: entity/relation extraction -> graph -> neighborhood
      retrieval for global reasoning.
    - spaCy NER for multilingual named entities.
    - NetworkX for graph storage and GraphML persistence.

Custom part:
    The political entity/relation typing is project-specific. Relations are
    inferred from co-occurrence and keywords, so they must remain marked as
    inferred context, not verified facts.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import networkx as nx

from compass.config import settings
from compass.schemas import Segment

logger = logging.getLogger(__name__)

_POLITICAL_ENT_TYPES = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LAW"}

_ALLIANCE_KEYWORDS = {
    "allie", "coalition", "accord", "alliance", "partenaire",
    "ally", "agreement", "partner", "united",
}
_OPPOSITION_KEYWORDS = {
    "oppose", "contre", "rival", "adversaire", "opposant",
    "opposed", "against", "adversary", "enemy",
}
_MERGER_KEYWORDS = {
    "fusion", "rejoint", "integre", "merge", "joined", "integrated",
}


def _party_edge_view(
    data: dict,
    party_id: str,
    country_iso3: str | None,
    cutoff: str,
) -> dict | None:
    """Build a party- and time-scoped edge view from its provenance records."""
    country = str(country_iso3 or "").upper()
    try:
        proofs = json.loads(data.get("proofs", "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        proofs = []
    matching = [
        proof for proof in proofs
        if str(proof.get("party_id") or "") == party_id
        and str(proof.get("date") or "9999") <= cutoff
        and (
            not country
            or str(proof.get("country_iso3") or country).upper() == country
        )
    ]
    if matching:
        dates = sorted(str(proof.get("date") or "") for proof in matching)
        view = dict(data)
        view.update({
            "weight": len(matching),
            "party_id": party_id,
            "country_iso3": country or data.get("country_iso3", ""),
            "date_first_seen": dates[0],
            "date_last_seen": dates[-1],
            "doc_type": str(matching[-1].get("doc_type") or data.get("doc_type") or ""),
            "proofs": json.dumps(matching),
        })
        return view

    # Backward compatibility for graphs persisted before per-proof provenance.
    country_ok = not country or str(data.get("country_iso3") or country).upper() == country
    if (
        country_ok
        and data.get("party_id") == party_id
        and data.get("date_first_seen", "9999") <= cutoff
    ):
        return dict(data)
    return None


class PoliticalGraph:
    """Political knowledge graph with temporal edges.

    Nodes represent named entities. Edges represent inferred co-mentions or
    heuristic relations, with date and source attributes. The temporal filter is
    applied to edges, allowing historical graph queries at election time.
    """

    def __init__(self, country_iso3: str | None = None) -> None:
        self.country = str(country_iso3 or "").upper() or None
        self._graph: nx.DiGraph = nx.DiGraph()
        self._ingested_segment_ids: set[str] = set()
        self._spacy_nlp = self._load_spacy()

    @staticmethod
    def _load_spacy():
        """Loads the configured spaCy model with graceful fallback."""
        import spacy

        model = settings.graph_spacy_model
        try:
            return spacy.load(model)
        except OSError:
            logger.warning(
                "spaCy model '%s' is missing. Trying 'en_core_web_sm'. "
                "Install the multilingual model with: "
                "python -m spacy download xx_ent_wiki_sm",
                model,
            )
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                logger.error(
                    "No spaCy NER model available. Install one with: "
                    "python -m spacy download en_core_web_sm"
                )
                return None

    def ingest(self, segments: list[Segment]) -> int:
        """Extracts entities and inferred relations from dated segments.

        Returns:
            Number of new edges added to the graph.
        """
        if self._spacy_nlp is None:
            logger.warning("C02b: spaCy unavailable, skipping graph extraction.")
            return 0

        if self.country:
            mismatched = [
                seg.segment_id for seg in segments
                if seg.meta.country_iso3.upper() != self.country
            ]
            if mismatched:
                raise ValueError(
                    f"{len(mismatched)} segment(s) outside graph country {self.country}: "
                    f"{mismatched[:3]}"
                )

        parents = [seg for seg in segments if seg.parent_segment_id is None]
        source_segments = parents or segments
        new_edges = 0
        for seg in source_segments:
            if seg.segment_id in self._ingested_segment_ids:
                continue
            doc = self._spacy_nlp(seg.text[:1000])
            entities = [ent for ent in doc.ents if ent.label_ in _POLITICAL_ENT_TYPES]
            self._ingested_segment_ids.add(seg.segment_id)
            if not entities:
                continue

            seg_date = seg.meta.doc_date
            doc_type = seg.meta.doc_type
            party_id = seg.meta.party_id or ""

            for ent in entities:
                node_id = self._normalize(ent.text)
                if not self._graph.has_node(node_id):
                    self._graph.add_node(
                        node_id,
                        label=ent.text,
                        ent_type=ent.label_,
                        first_seen=seg_date.isoformat(),
                        last_seen=seg_date.isoformat(),
                        mention_count=1,
                    )
                else:
                    node = self._graph.nodes[node_id]
                    node["mention_count"] = node.get("mention_count", 0) + 1
                    if seg_date.isoformat() > node.get("last_seen", ""):
                        node["last_seen"] = seg_date.isoformat()

            relation = self._classify_relation(seg.text.lower())
            for idx, ent_a in enumerate(entities):
                for ent_b in entities[idx + 1:]:
                    src = self._normalize(ent_a.text)
                    tgt = self._normalize(ent_b.text)
                    if src == tgt:
                        continue
                    proof = {
                        "date": seg_date.isoformat(),
                        "doc_type": doc_type,
                        "party_id": party_id,
                        "country_iso3": seg.meta.country_iso3.upper(),
                        "segment_id": seg.segment_id,
                        "regime": "inferred_cooccurrence",
                    }
                    if self._graph.has_edge(src, tgt):
                        edge = self._graph.edges[src, tgt]
                        edge["weight"] = edge.get("weight", 1) + 1
                        if seg_date.isoformat() > edge.get("date_last_seen", ""):
                            edge["date_last_seen"] = seg_date.isoformat()
                        proofs = json.loads(edge.get("proofs", "[]"))
                        proofs.append(proof)
                        edge["proofs"] = json.dumps(proofs[-20:])
                        party_ids = set(json.loads(edge.get("party_ids", "[]")))
                        if party_id:
                            party_ids.add(party_id)
                        edge["party_ids"] = json.dumps(sorted(party_ids))
                    else:
                        self._graph.add_edge(
                            src,
                            tgt,
                            relation=relation,
                            weight=1,
                            regime="inferred_cooccurrence",
                            date_first_seen=seg_date.isoformat(),
                            date_last_seen=seg_date.isoformat(),
                            doc_type=doc_type,
                            party_id=party_id,
                            party_ids=json.dumps([party_id] if party_id else []),
                            country_iso3=seg.meta.country_iso3.upper(),
                            segment_id=seg.segment_id,
                            proofs=json.dumps([proof]),
                        )
                        new_edges += 1

        logger.info(
            "C02b ingest: %d segments -> +%d edges (graph: %d nodes, %d edges)",
            len(segments),
            new_edges,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )
        return new_edges

    def query_party(
        self,
        party_id: str,
        as_of: date,
        k_hops: int = 2,
        top_k: int = 10,
    ) -> list[dict]:
        """Returns temporal relation summaries for a party neighborhood."""
        cutoff = as_of.isoformat()
        filtered = nx.DiGraph()
        party_nodes: set[str] = set()
        for src, tgt, data in self._graph.edges(data=True):
            party_view = _party_edge_view(data, party_id, self.country, cutoff)
            if party_view is not None:
                filtered.add_edge(src, tgt, **party_view)
                party_nodes.add(src)
                party_nodes.add(tgt)

        if not party_nodes:
            logger.info("C02b: no graph nodes found for party %s", party_id)
            return []

        neighborhood: set[str] = set(party_nodes)
        frontier = set(party_nodes)
        for _ in range(k_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in filtered:
                    next_frontier.update(filtered.successors(node))
                    next_frontier.update(filtered.predecessors(node))
            next_frontier -= neighborhood
            neighborhood.update(next_frontier)
            frontier = next_frontier

        subgraph = filtered.subgraph(neighborhood)
        edges = sorted(
            subgraph.edges(data=True),
            key=lambda item: item[2].get("weight", 0),
            reverse=True,
        )

        results: list[dict] = []
        for src, tgt, data in edges[:top_k]:
            src_label = self._graph.nodes.get(src, {}).get("label", src)
            tgt_label = self._graph.nodes.get(tgt, {}).get("label", tgt)
            relation = data.get("relation", "co_mention")
            weight = data.get("weight", 1)
            first_seen = data.get("date_first_seen", "")
            doc_type = data.get("doc_type", "")
            results.append({
                "entity": src_label,
                "relation": relation,
                "neighbor": tgt_label,
                "weight": weight,
                "date_first_seen": first_seen,
                "date_last_seen": data.get("date_last_seen", ""),
                "doc_type": doc_type,
                "regime": "inferred_cooccurrence",
                "summary": (
                    f"[INFERRED] {src_label} - {relation} - {tgt_label} "
                    f"(co-mentioned {weight} times since {first_seen[:10] or '?'}, "
                    f"source: {doc_type or '?'})"
                ),
            })

        logger.info(
            "C02b query %s (as_of %s): %d relation summaries",
            party_id,
            as_of,
            len(results),
        )
        return results

    def query_entity(self, entity_name: str, as_of: date, top_k: int = 8) -> list[dict]:
        """Queries the graph by entity name."""
        node_id = self._normalize(entity_name)
        if node_id not in self._graph:
            return []
        cutoff = as_of.isoformat()
        results = []
        for src, tgt, data in self._graph.edges(nbunch=[node_id], data=True):
            country_ok = not self.country or data.get("country_iso3") in {None, "", self.country}
            if country_ok and data.get("date_first_seen", "9999") <= cutoff:
                tgt_label = self._graph.nodes.get(tgt, {}).get("label", tgt)
                relation = data.get("relation", "co_mention")
                results.append({
                    "entity": entity_name,
                    "relation": relation,
                    "neighbor": tgt_label,
                    "weight": data.get("weight", 1),
                    "summary": f"[INFERRED] {entity_name} - {relation} - {tgt_label}",
                })
        return sorted(results, key=lambda item: item["weight"], reverse=True)[:top_k]

    def save(self, path: Path | None = None) -> None:
        """Persists the graph as GraphML."""
        target = path or self.storage_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._graph.graph["ingested_segment_ids"] = json.dumps(
            sorted(self._ingested_segment_ids)
        )
        if self.country:
            self._graph.graph["country_iso3"] = self.country
        nx.write_graphml(self._graph, str(target))
        logger.info(
            "C02b: graph saved (%d nodes, %d edges) -> %s",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            target,
        )

    def load(self, path: Path | None = None) -> None:
        """Loads a persisted GraphML graph if it exists."""
        target = path or self.storage_path
        if not target.exists():
            logger.info("C02b: no persisted graph found at %s, starting empty.", target)
            return
        self._graph = nx.read_graphml(str(target))
        stored_country = str(self._graph.graph.get("country_iso3") or "").upper()
        if self.country and stored_country and stored_country != self.country:
            raise ValueError(
                f"Graph country mismatch: expected {self.country}, found {stored_country}"
            )
        try:
            self._ingested_segment_ids = set(json.loads(
                self._graph.graph.get("ingested_segment_ids", "[]")
            ))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._ingested_segment_ids = set()
        logger.info(
            "C02b: graph loaded (%d nodes, %d edges)",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    @property
    def storage_path(self) -> Path:
        """Country-isolated GraphML path derived from the configured base path."""
        base = settings.graph_path
        if not self.country:
            return base
        suffix = base.suffix or ".graphml"
        return base.with_name(f"{base.stem}_{self.country.lower()}{suffix}")

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalizes an entity name into a stable node id."""
        return text.strip().lower().replace(" ", "_")[:80]

    @staticmethod
    def _classify_relation(text_lower: str) -> str:
        """Heuristic v0 relation typing from context keywords."""
        if any(keyword in text_lower for keyword in _ALLIANCE_KEYWORDS):
            return "alliance"
        if any(keyword in text_lower for keyword in _OPPOSITION_KEYWORDS):
            return "opposition"
        if any(keyword in text_lower for keyword in _MERGER_KEYWORDS):
            return "merger"
        return "co_mention"
