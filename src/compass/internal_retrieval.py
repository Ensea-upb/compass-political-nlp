"""C06 - Retrieval interne : retrouver les preuves pertinentes pour UNE variable.

ETAT DE L'ART REUTILISE (choix verifies par recherche web le 2026-06-11,
cf. CHOIX_COMPOSANTS.md) :
    - Recherche lexicale : rank-bm25 (BM25Okapi) -- hybride dense+lexical,
      standard RAG (Gao et al. 2023).
    - Re-ranking : BGE-reranker-v2-m3 -- meilleur defaut 2026 multilingue.
    - HyDE (Gap 2, 2026-06-14) : Gao et al. 2022 -- passage hypothetique
      genere par litellm (deja en place via C10/C11) avant le retrieval dense.
      Rapproche la requete de l'espace semantique des documents cibles.
    - Enrichissement parent (Gap 1, 2026-06-14) : texte du bloc parent (~400
      chars) injecte en prefixe du pair de re-ranking. Le cross-encoder voit
      le contexte thematique, pas seulement la phrase isolee.
      Brique : CountryMemory.fetch_by_ids().

CUSTOM (justifie) : formulation des requetes depuis la fiche C05 et prompt
HyDE ancre sur la fiche -- logique metier de quelques dizaines de lignes.
"""

from __future__ import annotations

import logging

import litellm
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from compass.country_memory import CountryMemory
from compass.party_election_case import CaseFile
from compass.config import settings
from compass.schemas import CaseKey, VariableSheet

logger = logging.getLogger(__name__)


class InternalRetriever:
    """Selectionne, dans le dossier, les passages pertinents pour une variable.

    Pipeline :
        1. HyDE -- genere un passage hypothetique ancre sur la fiche (optionnel).
        2. BM25 hybride -- score sur requete originale, pool elargi par HyDE.
        3. Enrichissement parent -- injecte le texte du bloc parent en prefixe.
        4. Re-ranking cross-encoder sur (requete, parent_prefix + enfant).
    """

    def __init__(self, country: CountryMemory, top_k: int = 10) -> None:
        self._country = country
        self._top_k = top_k
        self._reranker = CrossEncoder(settings.reranker_model)

    # ------------------------------------------------------------------ public
    def retrieve(
        self,
        dossier: CaseFile,
        sheet: VariableSheet,
        case: CaseKey | None = None,
    ) -> list[dict]:
        """Retrieval hybride (BM25 + dense + HyDE) puis re-ranking avec contexte parent.

        Args:
            dossier: dossier parti x election (C04) -- perimetre de recherche.
            sheet: fiche de la variable (C05) -- dicte la requete et le prompt HyDE.
            case: cle du cas, transmise a HyDE pour le filtre temporel ChromaDB.
                  Si None, HyDE est limite au pool du dossier existant.

        Returns:
            Passages tries par pertinence decroissante, avec score de re-ranking
            et texte parent injecte dans 'parent_text'.
        """
        query = self._build_query(sheet)
        pool = (
            dossier.party_documents
            + dossier.party_trajectory
            + dossier.national_context
        )
        if not pool:
            return []

        # --- Etape 1 : HyDE -- passe de retrieval dense supplementaire ---
        if settings.hyde_enabled:
            hyde_extras = self._hyde_retrieve(sheet, case)
            existing_ids = {p["segment_id"] for p in pool}
            new_extras = [p for p in hyde_extras if p["segment_id"] not in existing_ids]
            pool = pool + new_extras
            logger.info(
                "HyDE %s : +%d passages (pool total : %d)",
                sheet.variable_id, len(new_extras), len(pool),
            )

        # --- Etape 2 : BM25 sur le pool elargi ---
        corpus_tokens = [p["text"].lower().split() for p in pool]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query.lower().split())
        candidates = [
            p for _, p in sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
        ][:30]

        # --- Etape 3 : enrichissement parent ---
        candidates = self._inject_parent_text(candidates)

        # --- Etape 4 : re-ranking avec contexte parent en prefixe ---
        pairs = [
            (query, (f"{c.get('parent_text') or ''}\n{c['text']}").strip())
            for c in candidates
        ]
        rerank_scores = self._reranker.predict(pairs)
        ranked = sorted(
            zip(rerank_scores, candidates), key=lambda x: float(x[0]), reverse=True
        )
        out = []
        for score, cand in ranked[: self._top_k]:
            cand = dict(cand)
            cand["relevance"] = float(score)
            out.append(cand)
        logger.info("Retrieval %s : %d passages retenus", sheet.variable_id, len(out))
        return out

    # ------------------------------------------------------------------ HyDE
    def _hyde_retrieve(
        self, sheet: VariableSheet, case: CaseKey | None
    ) -> list[dict]:
        """Genere un passage hypothetique et interroge ChromaDB directement.

        Le passage HyDE est ancre sur la fiche (question + definition + ancres
        d'echelle) -- jamais un texte libre. Reponse a Halterman & Keith 2025 :
        la variable pilote tout, y compris le retrieval.
        """
        hyde_doc = self._generate_hyde_doc(sheet)
        if not hyde_doc:
            return []
        if case is not None:
            return self._country.query_documents(
                question=hyde_doc,
                as_of=case.election_date,
                k=15,
                party_id=case.party_id,
            )
        try:
            res = self._country._col.query(query_texts=[hyde_doc], n_results=15)
            return [
                {"segment_id": i, "text": d, "meta": m}
                for i, d, m in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0]
                )
            ]
        except Exception as exc:
            logger.warning("HyDE retrieval sans filtre echoue : %s", exc)
            return []

    def _generate_hyde_doc(self, sheet: VariableSheet) -> str:
        """Genere un passage hypothetique via litellm, ancre sur la fiche C05.

        En cas d'echec (pas de cle API, timeout), retourne une chaine vide --
        le retrieval continue sans HyDE, sans exception fatale.
        """
        scale_anchors = "\n".join(f"  {k}: {v}" for k, v in sheet.scale.items())
        prompt = (
            f"Tu es un expert en analyse politique comparative.\n"
            f"Variable a coder : {sheet.variable_id}\n"
            f"Question : {sheet.question}\n"
            f"Definition : {sheet.definition}\n"
            f"Echelle :\n{scale_anchors}\n\n"
            f"Redige un passage court (3-5 phrases) comme s'il etait extrait "
            f"d'un manifeste electoral, discours ou rapport d'observation qui "
            f"constituerait une preuve claire d'un score eleve sur cette variable. "
            f"Ecris directement le passage, sans introduction ni commentaire."
        )
        try:
            resp = litellm.completion(
                model=settings.hyde_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=settings.hyde_max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(
                "HyDE desactive pour %s : %s", sheet.variable_id, exc
            )
            return ""

    # ------------------------------------------------------------------ parent
    def _inject_parent_text(self, candidates: list[dict]) -> list[dict]:
        """Recupere et injecte le texte du bloc parent pour chaque passage enfant.

        Les passages sans parent_segment_id (segments racines ou anciens segments
        non hierarchiques) sont laisses intacts -- compatibilite ascendante.
        """
        parent_ids = list({
            m.get("parent_segment_id", "")
            for c in candidates
            for m in [c.get("meta", {})]
            if m.get("parent_segment_id")
        })
        if not parent_ids:
            return candidates
        parent_texts = self._country.fetch_by_ids(parent_ids)
        enriched = []
        for cand in candidates:
            cand = dict(cand)
            pid = cand.get("meta", {}).get("parent_segment_id", "")
            if pid and pid in parent_texts:
                cand["parent_text"] = parent_texts[pid]
            enriched.append(cand)
        return enriched

    # ------------------------------------------------------------------ requete
    @staticmethod
    def _build_query(sheet: VariableSheet) -> str:
        """La requete vient de la fiche : question + definition + sources requises."""
        return " ".join(
            [sheet.question, sheet.definition, " ".join(sheet.required_sources)]
        )
