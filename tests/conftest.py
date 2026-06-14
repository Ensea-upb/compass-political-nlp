"""conftest.py — stubs des dépendances lourdes pour les tests COMPASS.

Tous les modules externes non installés dans le sandbox sont stubbed ici,
AVANT que pytest n'importe les modules compass. L'ordre est garanti par
pytest qui charge conftest.py en premier.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str, **attrs) -> types.ModuleType:
    """Crée et enregistre un module stub dans sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ── ChromaDB ──────────────────────────────────────────────────────────────
_col = MagicMock()
_col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
_col.get.return_value = {"ids": [], "documents": []}
_col.upsert = MagicMock()
_chroma_client = MagicMock()
_chroma_client.get_or_create_collection.return_value = _col
_chroma = _stub("chromadb", PersistentClient=MagicMock(return_value=_chroma_client))
_stub("chromadb.utils")
_stub("chromadb.utils.embedding_functions",
      SentenceTransformerEmbeddingFunction=MagicMock())

# ── sentence-transformers ─────────────────────────────────────────────────
_cross_enc = MagicMock()
_cross_enc.predict = MagicMock(return_value=[0.9, 0.5, 0.2])
_stub("sentence_transformers", CrossEncoder=MagicMock(return_value=_cross_enc))

# ── rank-bm25 ─────────────────────────────────────────────────────────────
class _BM25:
    def __init__(self, corpus): pass
    def get_scores(self, q): return [1.0] * 30
_stub("rank_bm25", BM25Okapi=_BM25)

# ── litellm ───────────────────────────────────────────────────────────────
_litellm_resp = MagicMock()
_litellm_resp.choices[0].message.content = (
    '{"score": 3, "confidence": 0.8, "rationale": "test", '
    '"declared": [], "observed": [], "inferred": []}'
)
_litellm = _stub("litellm",
                 completion=MagicMock(return_value=_litellm_resp),
                 BadRequestError=Exception)

# ── transformers ──────────────────────────────────────────────────────────
_clf_result = {"labels": ["3: fort pluralisme"], "scores": [0.85]}
_stub("transformers", pipeline=MagicMock(return_value=MagicMock(return_value=_clf_result)))

# ── spaCy ─────────────────────────────────────────────────────────────────
_ent1 = MagicMock(); _ent1.text = "Parti Démocratique"; _ent1.label_ = "ORG"
_ent2 = MagicMock(); _ent2.text = "Sénégal"; _ent2.label_ = "GPE"
_doc = MagicMock(); _doc.ents = [_ent1, _ent2]
_nlp = MagicMock(return_value=_doc)
_stub("spacy", load=MagicMock(return_value=_nlp), blank=MagicMock(return_value=_nlp))

# ── networkx ─────────────────────────────────────────────────────────────
try:
    import networkx  # noqa: F401 — réel si installé
except ImportError:
    class _NodeView:
        def __init__(self, graph):
            self._graph = graph

        def __getitem__(self, node):
            return self._graph._nodes[node]

        def __contains__(self, node):
            return node in self._graph._nodes

        def get(self, node, default=None):
            return self._graph._nodes.get(node, default)

    class _EdgeView:
        def __init__(self, graph):
            self._graph = graph

        def __getitem__(self, key):
            return self._graph._edges[key]

        def __call__(self, nbunch=None, data=False):
            rows = []
            for (src, tgt), attrs in self._graph._edges.items():
                if nbunch is not None:
                    nodes = {nbunch} if isinstance(nbunch, str) else set(nbunch)
                    if src not in nodes and tgt not in nodes:
                        continue
                rows.append((src, tgt, attrs) if data else (src, tgt))
            return rows

    class _MiniDiGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = {}
            self.nodes = _NodeView(self)
            self.edges = _EdgeView(self)

        def add_node(self, node, **attrs):
            self._nodes.setdefault(node, {}).update(attrs)

        def add_edge(self, src, tgt, **attrs):
            self._nodes.setdefault(src, {})
            self._nodes.setdefault(tgt, {})
            self._edges[(src, tgt)] = dict(attrs)

        def has_node(self, node):
            return node in self._nodes

        def has_edge(self, src, tgt):
            return (src, tgt) in self._edges

        def successors(self, node):
            return [tgt for src, tgt in self._edges if src == node]

        def predecessors(self, node):
            return [src for src, tgt in self._edges if tgt == node]

        def subgraph(self, nodes):
            out = _MiniDiGraph()
            wanted = set(nodes)
            for node in wanted:
                if node in self._nodes:
                    out.add_node(node, **self._nodes[node])
            for (src, tgt), attrs in self._edges.items():
                if src in wanted and tgt in wanted:
                    out.add_edge(src, tgt, **attrs)
            return out

        def number_of_nodes(self):
            return len(self._nodes)

        def number_of_edges(self):
            return len(self._edges)

        def __contains__(self, node):
            return node in self._nodes

    _GRAPH_STORE = {}

    def _write_graphml(graph, path):
        _GRAPH_STORE[str(path)] = graph
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<graphml />")

    def _read_graphml(path):
        return _GRAPH_STORE.get(str(path), _MiniDiGraph())

    _stub("networkx", DiGraph=_MiniDiGraph,
          write_graphml=_write_graphml, read_graphml=_read_graphml)

# ── trafilatura ───────────────────────────────────────────────────────────
_stub("trafilatura")

# ── joblib ────────────────────────────────────────────────────────────────
_stub("joblib")

# ── sklearn ───────────────────────────────────────────────────────────────
_stub("sklearn")
_sk_cal=_stub("sklearn.calibration", CalibratedClassifierCV=MagicMock())
_stub("sklearn.linear_model", LogisticRegression=MagicMock())
_stub("sklearn.svm", SVC=MagicMock())

# ── ddgs (moteur de recherche C08)
_stub("ddgs", DDGS=MagicMock())

# ── htmldate ──────────────────────────────────────────────────────────────
_stub("htmldate")

# ── lingua ────────────────────────────────────────────────────────────────
try:
    import lingua  # noqa: F401
except ImportError:
    _l = _stub("lingua")
    _l.Language = MagicMock()
    _l.LanguageDetectorBuilder = MagicMock()

# ── fitz (PyMuPDF) ────────────────────────────────────────────────────────
_stub("fitz")

# ── pydantic-settings ────────────────────────────────────────────────────
try:
    import pydantic_settings  # noqa: F401
except ImportError:
    from pydantic import BaseModel

    class _BaseSettings(BaseModel):
        model_config = {}

    _stub("pydantic_settings",
          BaseSettings=_BaseSettings,
          SettingsConfigDict=dict)

# ── Exposer les mocks utiles aux tests ────────────────────────────────────
# Les tests peuvent faire : from tests.conftest import CHROMA_COL, LITELLM
CHROMA_COL = _col
LITELLM = _litellm
NLP_MOCK = _nlp
