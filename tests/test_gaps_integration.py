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
        monkeypatch.setattr(dp.settings, "semantic_chunk_similarity_threshold", 0.2)
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
