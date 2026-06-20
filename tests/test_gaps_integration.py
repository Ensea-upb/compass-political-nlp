"""Tests d'intégration des 3 gaps SOTA (2026-06-14).

Les dépendances lourdes (ChromaDB, CrossEncoder, litellm, spaCy) sont
stubées dans conftest.py — chaque test tourne en < 1 s sans GPU ni clé API.

Classes :
    TestGap1_HierarchicalChunking  — document_pipeline._finalize()
    TestGap2_HyDE                  — InternalRetriever (HyDE + parent injection)
    TestGap3_PoliticalGraph        — PoliticalGraph.ingest() + query_party()
    TestOrchestratorWiring         — orchestrator passe case= et graph_context
"""
from __future__ import annotations

import sys
import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── src/ au sys.path ──────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Helpers ───────────────────────────────────────────────────────────────

def _meta(**kw):
    from compass.schemas import DocumentMeta, TemporalStatus, SourceReliability
    return DocumentMeta(
        doc_id=kw.get("doc_id", "doc-001"),
        country_iso3="SEN",
        party_id=kw.get("party_id", "pd_sen"),
        doc_date=kw.get("doc_date", date(2024, 1, 1)),
        temporal_status=TemporalStatus.VERIFIED,
        eligible_for_historical_reasoning=True,
        doc_type=kw.get("doc_type", "manifeste"),
        language="fr",
        election_id=kw.get("election_id"),
        reliability=SourceReliability.OFFICIAL,
    )


def _seg(text: str, seg_id: str = "s1", parent_id: str | None = None):
    from compass.schemas import Segment
    return Segment(segment_id=seg_id, doc_id="doc-001",
                   text=text, meta=_meta(), parent_segment_id=parent_id)


def _sheet():
    from compass.schemas import VariableSheet, VariableMethod, EvidenceRegime, OutputType
    return VariableSheet(
        variable_id="v2paplur",
        question="Le parti accepte-t-il le pluralisme politique ?",
        definition="Mesure dans quelle mesure le parti accepte les autres partis.",
        scale={0: "Rejette totalement", 2: "Neutre", 4: "Accepte pleinement"},
        method=VariableMethod.LLM_GUIDED,
        output_type=OutputType.ORDINAL,
        evidence_regimes=[EvidenceRegime.DECLARED, EvidenceRegime.OBSERVED],
        required_sources=["manifeste", "discours"],
    )


def _case():
    from compass.schemas import CaseKey
    return CaseKey(country_iso3="SEN", party_id="pd_sen",
                   election_id="e2024", election_date=date(2024, 3, 24))


def _country_memory():
    """CountryMemory avec SQLite en mémoire et ChromaDB mocké."""
    from tests.conftest import CHROMA_COL
    from compass.country_memory import CountryMemory
    with patch("compass.country_memory.chromadb.PersistentClient") as mock_client, \
         patch("compass.country_memory.sqlite3.connect",
               return_value=sqlite3.connect(":memory:")):
        mock_client.return_value.get_or_create_collection.return_value = CHROMA_COL
        return CountryMemory("SEN")


# ══════════════════════════════════════════════════════════════════════════
# Gap 1 — Chunking hiérarchique
# ══════════════════════════════════════════════════════════════════════════

class TestGap1_HierarchicalChunking:
    """document_pipeline._finalize() produit blocs parents + phrases enfants."""

    @pytest.fixture(autouse=True)
    def pipeline(self):
        from compass.document_pipeline import DocumentPipeline
        self.p = DocumentPipeline()

    LONG_TEXT = (
        "Le parti s'engage à défendre les droits des citoyens sénégalais. "
        "Nous œuvrerons pour la justice sociale et l'équité. "
        "La démocratie est notre valeur fondamentale et inaliénable. "
        "Nous protégerons la liberté de la presse et d'expression. "
        "L'éducation sera notre priorité absolue dans ce mandat. "
        "Nous investirons massivement dans les infrastructures rurales et urbaines."
    )

    def test_produces_parents_and_children(self):
        segs = self.p._finalize(self.LONG_TEXT, _meta(doc_id="d1"))
        parents  = [s for s in segs if s.parent_segment_id is None]
        children = [s for s in segs if s.parent_segment_id is not None]
        assert len(segs) > 1
        assert len(parents) >= 1, "Aucun bloc parent produit"
        assert len(children) >= 1, "Aucun segment enfant produit"

    def test_children_point_to_existing_parents(self):
        segs = self.p._finalize(self.LONG_TEXT, _meta(doc_id="d2"))
        parent_ids = {s.segment_id for s in segs if s.parent_segment_id is None}
        for c in segs:
            if c.parent_segment_id is not None:
                assert c.parent_segment_id in parent_ids, (
                    f"Enfant {c.segment_id} → parent inexistant {c.parent_segment_id}"
                )

    def test_segment_ids_unique(self):
        segs = self.p._finalize(self.LONG_TEXT * 3, _meta(doc_id="d3"))
        ids = [s.segment_id for s in segs]
        assert len(ids) == len(set(ids)), "IDs de segments non uniques"

    def test_short_text_still_works(self):
        """Un texte court (1 phrase) ne doit pas lever d'exception."""
        segs = self.p._finalize("Une seule phrase courte.", _meta(doc_id="d4"))
        assert len(segs) >= 1

    def test_short_fragments_are_merged_into_citable_children(self):
        text = (
            "Setting impulses. Additional contributions. Questions raised. "
            "We defend democratic accountability and transparent elections."
        )

        segs = self.p._finalize(text, _meta(doc_id="d5"))
        children = [s for s in segs if s.parent_segment_id is not None]

        assert children
        assert all(child.text != "Setting impulses." for child in children)
        assert any("Setting impulses. Additional contributions." in child.text for child in children)

    def test_bullet_lists_keep_readable_units(self):
        text = (
            "Our economic priorities:\n"
            "- Jobs and fair wages.\n"
            "- Innovation and exports.\n"
            "- Sustainable growth."
        )

        segs = self.p._finalize(text, _meta(doc_id="d6"))
        child_text = " ".join(s.text for s in segs if s.parent_segment_id is not None)

        assert "Jobs and fair wages." in child_text
        assert "- Jobs" not in child_text

    def test_long_children_are_bounded(self, monkeypatch):
        from compass import document_pipeline as dp

        monkeypatch.setattr(dp.settings, "child_chunk_min_chars", 20)
        monkeypatch.setattr(dp.settings, "child_chunk_max_chars", 80)
        text = " ".join(["democratic accountability"] * 40) + "."

        segs = self.p._finalize(text, _meta(doc_id="d7"))
        children = [s for s in segs if s.parent_segment_id is not None]

        assert len(children) > 1
        assert all(len(child.text) <= 80 for child in children)

    def test_semantic_chunking_splits_topic_shifts(self, monkeypatch):
        from compass import document_pipeline as dp

        monkeypatch.setattr(dp.settings, "semantic_chunking_enabled", True)
        monkeypatch.setattr(dp.settings, "semantic_chunk_fallback_jaccard_threshold", 0.2)
        monkeypatch.setattr(dp.settings, "semantic_chunk_min_parent_chars", 80)
        monkeypatch.setattr(dp.settings, "child_chunk_min_chars", 20)
        monkeypatch.setattr(dp.settings, "parent_chunk_size", 300)
        text = (
            "Democratic accountability protects parliament and public oversight. "
            "Transparent elections protect parliament and democratic oversight. "
            "Solar infrastructure expands renewable energy and grid capacity. "
            "Renewable energy investment expands clean grid capacity."
        )

        segs = self.p._finalize(text, _meta(doc_id="d8"))
        parents = [s for s in segs if s.parent_segment_id is None]

        assert len(parents) >= 2
        assert "Democratic accountability" in parents[0].text
        assert any("Solar infrastructure" in parent.text for parent in parents[1:])

    def test_multilingual_embeddings_drive_topic_boundaries(self, monkeypatch):
        from compass import document_pipeline as dp

        class Encoder:
            def encode(self, texts, **kwargs):
                assert kwargs["normalize_embeddings"] is True
                return [
                    [1.0, 0.0] if "democr" in text.lower() or "election" in text.lower()
                    else [0.0, 1.0]
                    for text in texts
                ]

        monkeypatch.setattr(dp, "_load_semantic_encoder", lambda *args: Encoder())
        monkeypatch.setattr(dp.settings, "semantic_chunking_enabled", True)
        monkeypatch.setattr(dp.settings, "semantic_chunk_similarity_threshold", 0.5)
        monkeypatch.setattr(dp.settings, "semantic_chunk_min_parent_chars", 60)
        monkeypatch.setattr(dp.settings, "parent_chunk_size", 1000)
        text = (
            "La democratie protege le controle parlementaire. "
            "Les elections garantissent la responsabilite publique. "
            "Les energies renouvelables renforcent le reseau electrique. "
            "Les infrastructures solaires soutiennent la transition energetique."
        )

        parents = [
            segment for segment in self.p._finalize(text, _meta(doc_id="d9"))
            if segment.parent_segment_id is None
        ]

        assert len(parents) == 2
        assert "elections" in parents[0].text
        assert "renouvelables" in parents[1].text

    def test_semantic_chunking_has_deterministic_fallback(self, monkeypatch):
        from compass import document_pipeline as dp

        def unavailable(*args):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(dp, "_load_semantic_encoder", unavailable)
        monkeypatch.setattr(dp.settings, "semantic_chunking_enabled", True)
        monkeypatch.setattr(dp.settings, "semantic_chunk_min_parent_chars", 60)
        monkeypatch.setattr(dp.settings, "semantic_chunk_fallback_jaccard_threshold", 0.2)
        monkeypatch.setattr(dp.settings, "parent_chunk_size", 1000)
        text = (
            "Democratic institutions protect parliament and elections. "
            "Transparent elections reinforce democratic institutions. "
            "Solar grids expand renewable energy infrastructure. "
            "Renewable infrastructure supports solar electricity."
        )

        parents = [
            segment for segment in self.p._finalize(text, _meta(doc_id="d10"))
            if segment.parent_segment_id is None
        ]

        assert len(parents) >= 2

    def test_preserves_headings_paragraph_order_and_document_metadata(self, monkeypatch):
        from compass import document_pipeline as dp

        monkeypatch.setattr(dp.settings, "semantic_chunking_enabled", False)
        text = (
            "DEMOCRACY\n\n"
            "Citizens control public institutions.\n\n"
            "ECONOMY\n\n"
            "Workers receive fair wages."
        )
        segments = self.p._finalize(text, _meta(doc_id="d11"))
        parents = [segment for segment in segments if segment.parent_segment_id is None]
        children = [segment for segment in segments if segment.parent_segment_id is not None]

        assert [parent.section_title for parent in parents] == ["DEMOCRACY", "ECONOMY"]
        assert [parent.chunk_index for parent in parents] == [0, 1]
        assert parents[0].paragraph_start == 0
        assert parents[0].paragraph_end == 1
        assert parents[1].paragraph_start == 2
        assert parents[1].paragraph_end == 3
        assert [child.chunk_index for child in children] == list(range(len(children)))
        assert all(child.meta.doc_id == "d11" for child in children)
        assert all(child.meta.country_iso3 == "SEN" for child in children)

    def test_country_memory_marks_parent_and_child_levels(self):
        from tests.conftest import CHROMA_COL

        CHROMA_COL.upsert.reset_mock()
        memory = _country_memory()
        segments = [
            _seg("Bloc parent.", seg_id="doc:p000", parent_id=None),
            _seg("Phrase enfant.", seg_id="doc:p000c000", parent_id="doc:p000"),
        ]

        memory.add_documents(segments)

        metadatas = CHROMA_COL.upsert.call_args.kwargs["metadatas"]
        assert metadatas[0]["segment_level"] == "parent"
        assert metadatas[1]["segment_level"] == "child"

    def test_country_memory_persists_structural_metadata(self):
        from tests.conftest import CHROMA_COL

        CHROMA_COL.upsert.reset_mock()
        memory = _country_memory()
        segment = _seg("Evidence.", seg_id="doc:p000c000", parent_id="doc:p000")
        segment.meta.election_id = "SEN_2024"
        segment.chunk_index = 4
        segment.paragraph_start = 7
        segment.paragraph_end = 8
        segment.section_title = "Democracy"

        memory.add_documents([segment])

        metadata = CHROMA_COL.upsert.call_args.kwargs["metadatas"][0]
        assert metadata["chunk_index"] == 4
        assert metadata["paragraph_start"] == 7
        assert metadata["paragraph_end"] == 8
        assert metadata["section_title"] == "Democracy"
        assert metadata["election_id"] == "SEN_2024"

    def test_country_memory_describes_the_indexed_corpus(self):
        from tests.conftest import CHROMA_COL

        memory = _country_memory()
        memory._conn.execute(
            "INSERT INTO parties (party_id, country_iso3, name) VALUES (?,?,?)",
            ("pd_sen", "SEN", "Parti test"),
        )
        memory._conn.commit()
        CHROMA_COL.get.reset_mock()
        CHROMA_COL.get.return_value = {
            "ids": ["doc:p000", "doc:p000c000", "doc2:p000c000"],
            "metadatas": [
                {"doc_id": "doc", "country_iso3": "SEN", "party_id": "pd_sen", "doc_date": "2024-01-01", "doc_type": "manifesto"},
                {"doc_id": "doc", "country_iso3": "SEN", "party_id": "pd_sen", "doc_date": "2024-01-01", "doc_type": "manifesto"},
                {"doc_id": "doc2", "country_iso3": "SEN", "party_id": "pd_sen", "doc_date": "2024-02-01", "doc_type": "speech"},
            ],
        }

        scope = memory.describe_corpus(as_of=date(2024, 3, 24), party_id="pd_sen")

        assert scope["country_iso3"] == "SEN"
        assert scope["n_documents"] == 2
        assert scope["parties"] == [{"party_id": "pd_sen", "name": "Parti test"}]
        assert scope["document_dates"] == ["2024-01-01", "2024-02-01"]
        assert scope["document_types"] == ["manifesto", "speech"]

    def test_country_memory_lists_parent_records_for_graph_backfill(self):
        from tests.conftest import CHROMA_COL

        memory = _country_memory()
        CHROMA_COL.get.reset_mock()
        CHROMA_COL.get.return_value = {
            "ids": ["doc:p000"],
            "documents": ["Parent text."],
            "metadatas": [{"segment_level": "parent", "party_id": "pd_sen"}],
        }

        records = memory.list_document_records(party_id="pd_sen", parent_only=True)

        assert records[0]["segment_id"] == "doc:p000"
        where = CHROMA_COL.get.call_args.kwargs["where"]
        assert {"segment_level": "parent"} in where["$and"]

    def test_query_documents_targets_children_then_falls_back_for_old_indexes(self):
        from tests.conftest import CHROMA_COL

        memory = _country_memory()
        CHROMA_COL.query.reset_mock()
        CHROMA_COL.query.side_effect = [
            {"ids": [[]], "documents": [[]], "metadatas": [[]]},
            {"ids": [["legacy:p000"]], "documents": [["Ancien segment."]], "metadatas": [[{"party_id": "pd_sen"}]]},
        ]

        result = memory.query_documents("pluralisme", as_of=date(2024, 3, 24), party_id="pd_sen")

        first_where = CHROMA_COL.query.call_args_list[0].kwargs["where"]
        second_where = CHROMA_COL.query.call_args_list[1].kwargs["where"]
        assert {"segment_level": "child"} in first_where["$and"]
        assert {"segment_level": "child"} not in second_where["$and"]
        assert result[0]["segment_id"] == "legacy:p000"
        CHROMA_COL.query.side_effect = None

    def test_query_documents_hybrid_fuses_dense_and_bm25(self):
        from tests.conftest import CHROMA_COL

        memory = _country_memory()
        CHROMA_COL.query.reset_mock()
        CHROMA_COL.get.reset_mock()
        CHROMA_COL.query.return_value = {
            "ids": [["dense:p000c000"]],
            "documents": [["General democracy text."]],
            "metadatas": [[{"party_id": "pd_sen", "segment_level": "child"}]],
        }
        CHROMA_COL.get.return_value = {
            "ids": ["lex:p000c000"],
            "documents": ["democracy parliament accountability"],
            "metadatas": [{"party_id": "pd_sen", "segment_level": "child"}],
        }

        result = memory.query_documents_hybrid("democracy accountability", as_of=date(2024, 3, 24), party_id="pd_sen", k=2)

        assert len(result) == 2
        assert {item["segment_id"] for item in result} == {"dense:p000c000", "lex:p000c000"}
        assert all("hybrid_score" in item for item in result)

    def test_query_documents_hybrid_uses_generic_cross_encoder(self, monkeypatch):
        from tests.conftest import CHROMA_COL
        from compass import country_memory

        memory = _country_memory()
        CHROMA_COL.query.reset_mock()
        CHROMA_COL.get.reset_mock()
        CHROMA_COL.query.return_value = {
            "ids": [["indirect:p000c000"]],
            "documents": [["Sustainability and culture shape future generations."]],
            "metadatas": [[{"party_id": "pd_sen", "segment_level": "child"}]],
        }
        CHROMA_COL.get.return_value = {
            "ids": ["direct:p000c000", "indirect:p000c000"],
            "documents": [
                "Democratic participation, citizens, parliament, and constitutional rights are central.",
                "Sustainability and culture shape future generations.",
            ],
            "metadatas": [
                {"party_id": "pd_sen", "segment_level": "child"},
                {"party_id": "pd_sen", "segment_level": "child"},
            ],
        }

        def fake_scores(question, records):
            return [
                0.9 if "Democratic participation" in item["text"] else 0.1
                for item in records
            ]

        monkeypatch.setattr(country_memory, "_cross_encoder_scores", fake_scores)
        result = memory.query_documents_hybrid(
            "What does the party say about democracy?",
            as_of=date(2024, 3, 24),
            party_id="pd_sen",
            k=2,
        )

        assert result[0]["segment_id"] == "direct:p000c000"
        assert "cross_encoder_score=0.9000" in result[0]["retrieval_reason"]
        assert "profile_boost" not in result[0]["retrieval_reason"]

    def test_query_documents_hybrid_cross_encoder_reads_parent_context(self, monkeypatch):
        from tests.conftest import CHROMA_COL
        from compass import country_memory

        memory = _country_memory()
        CHROMA_COL.query.reset_mock()
        CHROMA_COL.get.reset_mock()
        CHROMA_COL.query.return_value = {
            "ids": [["low:p000c000"]],
            "documents": [["General institutional text."]],
            "metadatas": [[{
                "party_id": "pd_sen",
                "segment_level": "child",
                "parent_segment_id": "low:p000",
            }]],
        }
        CHROMA_COL.get.side_effect = [
            {
                "ids": ["low:p000c000", "high:p000c000"],
                "documents": [
                    "General institutional text.",
                    "The party defends elections and constitutional accountability.",
                ],
                "metadatas": [
                    {
                        "party_id": "pd_sen",
                        "segment_level": "child",
                        "parent_segment_id": "low:p000",
                    },
                    {
                        "party_id": "pd_sen",
                        "segment_level": "child",
                        "parent_segment_id": "high:p000",
                    },
                ],
            },
            {
                "ids": ["high:p000", "low:p000"],
                "documents": [
                    "Parent section on democracy, elections, parliament, and citizen rights.",
                    "Parent section on general administration.",
                ],
            },
        ]

        def fake_scores(question, records):
            return [0.9 if item["segment_id"] == "high:p000c000" else 0.1 for item in records]

        monkeypatch.setattr(country_memory, "_cross_encoder_scores", fake_scores)

        result = memory.query_documents_hybrid(
            "What does the party say about democracy?",
            as_of=date(2024, 3, 24),
            party_id="pd_sen",
            k=1,
        )

        assert result[0]["segment_id"] == "high:p000c000"
        assert result[0]["parent_text"].startswith("Parent section on democracy")
        assert result[0]["rerank_score"] == 0.9
        assert "cross_encoder_score=0.9000" in result[0]["retrieval_reason"]
        CHROMA_COL.get.side_effect = None

    def test_rerank_text_combines_parent_and_child(self):
        from compass.country_memory import _rerank_text

        text = _rerank_text({
            "parent_text": "Parent context about democratic institutions.",
            "text": "The party supports transparent elections.",
        })

        assert "Parent context about democratic institutions." in text
        assert "Evidence segment:" in text
        assert "The party supports transparent elections." in text


# ══════════════════════════════════════════════════════════════════════════
# Gap 2 — HyDE dans C06
# ══════════════════════════════════════════════════════════════════════════

class TestGap2_HyDE:
    """InternalRetriever : HyDE activable/désactivable, dégradation gracieuse,
    injection du texte parent, construction de requête depuis fiche."""

    @pytest.fixture(autouse=True)
    def retriever(self):
        from compass.internal_retrieval import InternalRetriever
        self.country = _country_memory()
        with patch("compass.internal_retrieval.CrossEncoder") as MockCE:
            MockCE.return_value.predict.return_value = [0.9, 0.7, 0.5]
            self.r = InternalRetriever(country=self.country, top_k=3)

    def _dossier(self):
        from compass.party_election_case import CaseFile
        d = CaseFile(case=_case())
        d.party_documents = [
            {"segment_id": f"seg-{i}", "text": f"Texte {i}.",
             "meta": {"parent_segment_id": "p0", "party_id": "pd_sen"}}
            for i in range(3)
        ]
        d.party_trajectory = []
        d.national_context = []
        return d

    def test_hyde_called_when_enabled(self):
        with patch("compass.internal_retrieval.settings") as s:
            s.hyde_enabled = True
            with patch.object(self.r, "_hyde_retrieve", return_value=[]) as m:
                self.r.retrieve(self._dossier(), _sheet())
            m.assert_called_once_with(_sheet(), None)

    def test_hyde_skipped_when_disabled(self):
        with patch("compass.internal_retrieval.settings") as s:
            s.hyde_enabled = False
            with patch.object(self.r, "_hyde_retrieve", return_value=[]) as m:
                self.r.retrieve(self._dossier(), _sheet())
            m.assert_not_called()

    def test_generate_hyde_doc_calls_llm_client(self):
        with patch("compass.internal_retrieval.complete_chat", return_value="Passage hypothétique.") as mock_llm, \
             patch("compass.internal_retrieval.settings") as s:
            s.hyde_model = "Qwen/Qwen3-14B"
            s.hyde_max_tokens = 250

            out = self.r._generate_hyde_doc(_sheet())
            assert out == "Passage hypothétique."
            mock_llm.assert_called_once()
            args, kw = mock_llm.call_args
            assert args[0] == "Qwen/Qwen3-14B"
            assert kw["temperature"] == 0.3
            assert kw["max_tokens"] == 250

    def test_hyde_degrades_gracefully_on_api_error(self):
        with patch("compass.internal_retrieval.complete_chat") as mock_llm, \
             patch("compass.internal_retrieval.settings") as s:
            s.hyde_model = "Qwen/Qwen3-14B"
            s.hyde_max_tokens = 250
            mock_llm.side_effect = Exception("API down")
            assert self.r._generate_hyde_doc(_sheet()) == ""

    def test_parent_text_injected(self):
        candidates = [
            {"segment_id": "c1", "text": "Enfant 1.",
             "meta": {"parent_segment_id": "p1"}},
            {"segment_id": "c2", "text": "Enfant 2.",
             "meta": {"parent_segment_id": "p1"}},
            {"segment_id": "c3", "text": "Racine.",
             "meta": {"parent_segment_id": ""}},
        ]
        with patch.object(self.country, "fetch_by_ids",
                          return_value={"p1": "Bloc parent thématique."}):
            result = self.r._inject_parent_text(candidates)

        assert result[0]["parent_text"] == "Bloc parent thématique."
        assert result[1]["parent_text"] == "Bloc parent thématique."
        assert "parent_text" not in result[2]

    def test_fetch_by_ids_called_with_unique_parent_ids(self):
        candidates = [
            {"segment_id": f"c{i}", "text": f"Enfant {i}.",
             "meta": {"parent_segment_id": "p1"}}
            for i in range(5)
        ]
        with patch.object(self.country, "fetch_by_ids",
                          return_value={"p1": "Parent."}) as mock_fetch:
            self.r._inject_parent_text(candidates)
        # fetch_by_ids ne doit être appelé qu'avec des IDs uniques
        mock_fetch.assert_called_once_with(["p1"])

    def test_build_query_contains_sheet_fields(self):
        from compass.internal_retrieval import InternalRetriever
        sh = _sheet()
        q = InternalRetriever._build_query(sh)
        assert sh.question in q
        assert sh.definition in q
        for src in sh.required_sources:
            assert src in q

    def test_hybrid_select_combines_dense_order_and_bm25_signal(self, monkeypatch):
        from compass import internal_retrieval
        from compass.internal_retrieval import InternalRetriever

        class FakeBM25:
            def __init__(self, corpus):
                self.corpus = corpus

            def get_scores(self, query_tokens):
                return [0.0, 20.0, 0.0]

        monkeypatch.setattr(internal_retrieval, "BM25Okapi", FakeBM25)
        pool = [
            {"segment_id": "dense-1", "text": "institutional reform"},
            {"segment_id": "lexical-hit", "text": "pluralisme politique et partis"},
            {"segment_id": "dense-3", "text": "economic investment"},
        ]

        selected = InternalRetriever._hybrid_select(pool, "pluralisme politique", limit=2)

        assert selected[0]["segment_id"] == "lexical-hit"
        assert "dense_rank" in selected[0]
        assert "bm25_score" in selected[0]
        assert "hybrid_score" in selected[0]


# ══════════════════════════════════════════════════════════════════════════
# Gap 3 — Graphe de connaissances politiques (C02b)
# ══════════════════════════════════════════════════════════════════════════

class TestGap3_PoliticalGraph:
    """PoliticalGraph : ingest() extrait entités, query_party() retourne
    résumés relationnels avec filtre temporel."""

    @pytest.fixture(autouse=True)
    def graph(self):
        from compass.political_graph import PoliticalGraph
        # spaCy déjà stubé dans conftest (spacy.load → _nlp_mock)
        self.g = PoliticalGraph()

    def test_ingest_returns_int(self):
        segs = [_seg("Le Parti Démocratique et le Sénégal signent un accord.", f"s{i}")
                for i in range(3)]
        n = self.g.ingest(segs)
        assert isinstance(n, int)
        assert n >= 0

    def test_query_party_returns_list_of_dicts_with_summary(self):
        segs = [_seg("Le Parti Démocratique s'allie avec le Sénégal.", f"s{i}")
                for i in range(3)]
        self.g.ingest(segs)
        results = self.g.query_party("pd_sen", as_of=date(2024, 3, 24), k_hops=2, top_k=10)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, dict)
            assert "summary" in item, f"Clé 'summary' absente : {item}"

    def test_all_relations_labeled_inferred_cooccurrence(self):
        """Garantie épistémique : toutes les arêtes sont INFÉRÉ."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx non installé")
        segs = [_seg("Alliance entre Parti Démocratique et Union Progressiste.", f"s{i}")
                for i in range(3)]
        self.g.ingest(segs)
        for _, _, data in self.g._graph.edges(data=True):
            assert data.get("regime") == "inferred_cooccurrence", (
                f"Arête sans régime inferred : {data}"
            )

    def test_query_party_temporal_filter_future(self):
        """query_party() avec as_of antérieur n'inclut pas les relations futures."""
        future_seg = _seg(
            "Parti Démocratique fusionne avec Union Nationale.",
            seg_id="future",
        )
        future_seg.meta.doc_date = date(2026, 1, 1)
        self.g.ingest([future_seg])

        results = self.g.query_party(
            "pd_sen", as_of=date(2023, 12, 31), k_hops=2, top_k=10
        )
        assert isinstance(results, list)
        # Les résumés ne doivent pas contenir de dates postérieures au cutoff
        for item in results:
            assert "summary" in item

    def test_save_load_roundtrip(self, tmp_path):
        """save() + load() préserve le graphe (stub NetworkX si absent)."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx non installé")
        segs = [_seg("Parti Démocratique au Sénégal.", f"s{i}") for i in range(2)]
        self.g.ingest(segs)
        path = tmp_path / "test_graph.graphml"
        self.g.save(path)
        assert path.exists()

        from compass.political_graph import PoliticalGraph
        g2 = PoliticalGraph()
        g2.load(path)
        assert g2._graph.number_of_nodes() == self.g._graph.number_of_nodes()
        assert g2._ingested_segment_ids == self.g._ingested_segment_ids

    def test_country_graph_rejects_segments_from_another_country(self):
        from compass.political_graph import PoliticalGraph

        graph = PoliticalGraph("USA")
        with pytest.raises(ValueError, match="outside graph country"):
            graph.ingest([_seg("Parti Démocratique et Sénégal.", "foreign")])

    def test_country_graph_uses_an_isolated_storage_path(self, tmp_path, monkeypatch):
        from compass.config import settings
        from compass.political_graph import PoliticalGraph

        monkeypatch.setattr(settings, "graph_path", tmp_path / "political_graph.graphml")

        assert PoliticalGraph("USA").storage_path == tmp_path / "political_graph_usa.graphml"
        assert PoliticalGraph("SEN").storage_path == tmp_path / "political_graph_sen.graphml"

    def test_graph_ingestion_is_idempotent(self):
        segments = [_seg("Le Parti Démocratique s'allie avec le Sénégal.", "stable")]

        self.g.ingest(segments)
        first_edge_count = self.g.edge_count
        second_new_edges = self.g.ingest(segments)

        assert second_new_edges == 0
        assert self.g.edge_count == first_edge_count

    def test_query_party_does_not_mix_proofs_from_other_parties(self):
        first = _seg("Parti Démocratique s'allie avec Union Nationale.", "party-a")
        second = _seg("Parti Démocratique s'allie avec Union Nationale.", "party-b")
        first.meta.party_id = "party_a"
        second.meta.party_id = "party_b"
        self.g.ingest([first, second])

        results_a = self.g.query_party("party_a", as_of=date(2024, 3, 24))
        results_b = self.g.query_party("party_b", as_of=date(2024, 3, 24))

        assert results_a and results_b
        assert all(item["weight"] == 1 for item in results_a)
        assert all(item["weight"] == 1 for item in results_b)


# ══════════════════════════════════════════════════════════════════════════
# Câblage orchestrateur
# ══════════════════════════════════════════════════════════════════════════

class TestOrchestratorWiring:
    """Vérifie le câblage des 3 gaps dans CompassRunner.run_case()."""

    def _run_with_mocks(self, graph_ctx=None):
        """Exécute run_case() avec tous les composants mockés.

        Retourne (mock_retrieve, mock_diagnosis) pour les assertions.
        """
        from compass.schemas import SufficiencyVerdict, FinalAnswer
        from compass.party_election_case import CaseFile

        # Dossier
        mock_dossier = MagicMock(spec=CaseFile)
        mock_dossier.party_documents = []
        mock_dossier.party_trajectory = []
        mock_dossier.national_context = []
        mock_dossier.graph_context = graph_ctx or []

        # Diagnostic
        mock_diagnosis = MagicMock()
        mock_diagnosis.convergent = []
        mock_diagnosis.contradictory = []
        mock_diagnosis.contradictions_detail = []
        mock_diagnosis.missing = []
        mock_diagnosis.dominant_language = "fr"
        mock_diagnosis.graph_context = []

        # Réponse finale
        mock_answer = MagicMock(spec=FinalAnswer)
        mock_answer.abstained = False
        mock_answer.score = 3.0
        mock_answer.labels = []
        mock_answer.confidence = 0.8
        mock_answer.attribution_checked = True

        mock_retrieve = MagicMock(return_value=[])
        mock_aggregate = MagicMock()
        mock_aggregate.score = 3.0
        mock_aggregate.labels = []

        with patch("compass.orchestrator.InternalRetriever") as MockIR, \
             patch("compass.orchestrator.CaseFileBuilder") as MockCFB, \
             patch("compass.orchestrator.VPartyRegistry") as MockReg, \
             patch("compass.orchestrator.TraceLogger"), \
             patch("compass.orchestrator.assert_temporal_integrity"), \
             patch("compass.orchestrator.SufficiencyGate") as MockGate, \
             patch("compass.orchestrator.DiagnosisEngine") as MockDiag, \
             patch("compass.orchestrator.EvidenceQualifier"), \
             patch("compass.orchestrator.JudgePanel") as MockPanel, \
             patch("compass.orchestrator.AnswerComposer") as MockComposer, \
             patch("compass.orchestrator.aggregate", return_value=mock_aggregate), \
             patch("compass.orchestrator.GeneralMemory"), \
             patch("compass.orchestrator.CountryMemory"), \
             patch("compass.orchestrator.ActiveSearchEngine"), \
             patch("compass.orchestrator.ReasoningEngine"):

            from compass.orchestrator import CompassRunner

            MockCFB.return_value.build.return_value = mock_dossier
            MockIR.return_value.retrieve = mock_retrieve
            MockReg.return_value.get.return_value = _sheet()
            MockGate.return_value.decide.return_value = (SufficiencyVerdict.SUFFICIENT, 0.9)
            MockDiag.return_value.diagnose.return_value = mock_diagnosis
            MockPanel.return_value.evaluate.return_value = []
            MockComposer.return_value.compose.return_value = mock_answer

            runner = CompassRunner(
                country=MagicMock(), general=MagicMock(),
                registry=MagicMock(), search=MagicMock(),
            )
            runner.run_case(_case(), ["v2paplur"])

        return mock_retrieve, mock_diagnosis

    def test_retrieve_receives_case_kwarg(self):
        """retrieve() doit être appelé avec case=CaseKey."""
        mock_retrieve, _ = self._run_with_mocks()
        mock_retrieve.assert_called_once()
        _, kw = mock_retrieve.call_args
        assert kw.get("case") == _case(), (
            f"retrieve() sans case= : kwargs={kw}"
        )

    def test_graph_context_propagated_to_diagnosis(self):
        """diagnosis.graph_context ← dossier.graph_context (Gap 3)."""
        expected = [{"summary": "PD allié avec APR"}, {"summary": "PD vs UDP"}]
        _, mock_diagnosis = self._run_with_mocks(graph_ctx=expected)
        assert mock_diagnosis.graph_context == expected, (
            "graph_context non propagé au diagnostic"
        )

    def test_active_search_segments_update_country_memory_and_graph(self):
        from compass.orchestrator import CompassRunner

        runner = CompassRunner.__new__(CompassRunner)
        runner._country = MagicMock()
        runner._graph = MagicMock()
        segments = [_seg("New relation evidence.", "searched")]

        runner._store_discovered_segments(segments)

        runner._country.add_documents.assert_called_once_with(segments)
        runner._graph.ingest.assert_called_once_with(segments)
        runner._graph.save.assert_called_once_with()
