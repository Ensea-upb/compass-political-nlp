"""Configuration centralisée du système COMPASS Expert-Pays.

Colonne vertébrale partagée — pas un composant de l'architecture.
État de l'art réutilisé : pydantic-settings (validation + variables d'environnement),
conformément au standard du projet (Claude.md §10 : configuration externalisée,
secrets via variables d'environnement, jamais en dur).

Toutes les valeurs sont surchargeables par variable d'environnement préfixée
``COMPASS_`` ou par fichier ``.env`` à la racine de ``compass_system/``.

Choix de modèles sourcés par recherche web le 2026-06-11 — justification et
alternatives : CHOIX_COMPOSANTS.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CompassSettings(BaseSettings):
    """Paramètres globaux, injectés dans chaque composant à la construction.

    Attributes:
        data_dir: racine des données (corpus, bases structurées, index).
        registry_dir: dossier des fiches YAML du registre V-Party (C05).
        chroma_dir: dossier de persistance des index vectoriels (C02/C03).
        sqlite_path: base structurée de la mémoire pays (C03, couche structurée).
        trace_dir: journal d'exécution (C15 — traçabilité).
        embedding_model: BGE-M3 — standard production multilingue 100+ langues
            (MTEB/MMTEB 2026, cf. CHOIX_COMPOSANTS.md §1).
        reranker_model: BGE-reranker-v2-m3 — meilleur défaut qualité/latence/
            licence des comparatifs 2026 (CHOIX_COMPOSANTS.md §2).
        nli_model: mDeBERTa-v3 XNLI — confirmé par la littérature stance
            cross-lingue 2025, surtout langues sous-représentées (§3).
        political_classifier: Political DEBATE (Political Analysis 2025) —
            zero/few-shot spécialisé textes politiques, ANGLAIS uniquement ;
            bascule automatique vers nli_model hors anglais (§3).
        judge_models: identifiants litellm des juges (C11) — modèles de bases
            DIFFÉRENTES, condition d'indépendance (Weidmann et al. 2026) ;
            panel à confirmer sur AfroBench au pilote (§6).
        llm_temperature: 0.0 pour la reproductibilité (Ornstein et al. 2025).
        search_max_queries: budget de requêtes par cas — borne la boucle C07-C08.
        search_max_iterations: nombre maximal de cycles suffisance-recherche.
        sufficiency_threshold: seuil de la prédiction sélective (à calibrer, R-2).
    """

    model_config = SettingsConfigDict(
        env_prefix="COMPASS_", env_file=".env", env_file_encoding="utf-8"
    )

    data_dir: Path = Path("data")
    registry_dir: Path = Path("registry")
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/compass_structured.db")
    vault_path: Path = Path("data/evaluation_vault.db")  # P0-3 : étalon V-Party, ISOLÉ de la production
    trace_dir: Path = Path("data/traces")

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    political_classifier: str = "mlburnham/Political_DEBATE_large_v1.0"

    judge_models: list[str] = Field(
        default=["gpt-4o", "mistral/mistral-large-latest", "claude-sonnet-4-6"]
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    search_max_queries: int = 8
    search_max_iterations: int = 2
    sufficiency_threshold: float = 0.6  # provisoire — à calibrer sur courbe risque-couverture

    # --- HyDE (Hypothetical Document Embeddings, Gao et al. 2022) ---
    # Génère un passage hypothétique avant le retrieval dense pour mieux
    # capturer la sémantique des documents cibles (vs. la question abstraite).
    # Le modèle HyDE doit être rapide (max_tokens court) — gpt-4o-mini suffit.
    hyde_enabled: bool = True
    hyde_model: str = "gpt-4o-mini"
    hyde_max_tokens: int = 250

    # --- Chunking hiérarchique parent-child (Gap 1) ---
    # Taille cible des blocs parents en caractères (~3-5 phrases).
    parent_chunk_size: int = 400

    # --- Graphe de connaissances politiques (C02b, Gap 3) ---
    graph_path: Path = Path("data/political_graph.graphml")
    graph_spacy_model: str = "xx_ent_wiki_sm"

    def ensure_dirs(self) -> None:
        """Crée les dossiers de travail si absents (idempotent)."""
        for d in (self.data_dir, self.registry_dir, self.chroma_dir, self.trace_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = CompassSettings()

