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
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM backend configuration.

    Defaults target a local OpenAI-compatible server, for example vLLM on
    Onyxia. Every value can be overridden with ``COMPASS_*`` environment
    variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="COMPASS_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    judge_models: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "Qwen/Qwen2.5-3B-Instruct",
    ])
    hyde_model: str = "Qwen/Qwen2.5-3B-Instruct"
    vision_model: str | None = None

    llm_backend: str = "local"          # "local" | "api"
    llm_api_base: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    hf_device: str = "auto"  # "auto" | "cpu" | "cuda" | device index

    @field_validator("judge_models", mode="before")
    @classmethod
    def _parse_judge_models(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def hf_pipeline_kwargs(self) -> dict[str, int | str]:
        """Device kwargs for transformers pipelines.

        Set COMPASS_HF_DEVICE=cpu on small GPUs when vLLM already owns the GPU.
        """
        device = str(self.hf_device).strip().lower()
        if device == "auto" or not device:
            return {}
        if device == "cpu":
            return {"device": -1}
        if device.isdigit() or (device.startswith("-") and device[1:].isdigit()):
            return {"device": int(device)}
        return {"device": device}

    def hf_model_device(self) -> str | None:
        """Device value for sentence-transformers classes."""
        device = str(self.hf_device).strip().lower()
        if device == "auto" or not device:
            return None
        if device == "-1":
            return "cpu"
        return device

    def litellm_model(self, model_name: str) -> str:
        """Returns the LiteLLM model identifier for the configured backend."""
        if self.llm_backend == "local" and not model_name.startswith("openai/"):
            return f"openai/{model_name}"
        return model_name

    def litellm_kwargs(self, model_name: str) -> dict[str, str]:
        """Connection kwargs for LiteLLM completion calls."""
        if self.llm_backend == "local":
            return {
                "model": self.litellm_model(model_name),
                "api_base": self.llm_api_base,
                "api_key": self.llm_api_key,
            }
        return {"model": model_name}


class CompassSettings(LLMConfig):
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
        judge_models: identifiants Hugging Face des juges locaux (C11) — modèles
            de bases DIFFÉRENTES, servis via endpoint OpenAI-compatible local.
        llm_temperature: 0.0 pour la reproductibilité (Ornstein et al. 2025).
        search_max_queries: budget de requêtes par cas — borne la boucle C07-C08.
        search_max_iterations: nombre maximal de cycles suffisance-recherche.
        sufficiency_threshold: seuil de la prédiction sélective (à calibrer, R-2).
    """

    model_config = SettingsConfigDict(
        env_prefix="COMPASS_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    registry_dir: Path = Path("registry")
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/compass_structured.db")
    vault_path: Path = Path("data/evaluation_vault.db")  # P0-3 : étalon V-Party, ISOLÉ de la production
    trace_dir: Path = Path("data/traces")

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_pool_size: int = 24
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    political_classifier: str = "mlburnham/Political_DEBATE_large_v1.0"

    # --- Chat RAG : budgets de preuve et garde-fou sémantique optionnel ---
    chat_max_prompt_citations: int = 4
    chat_max_evidence_text_chars: int = 420
    chat_semantic_validation_enabled: bool = False
    chat_nli_entailment_threshold: float = 0.65

    search_max_queries: int = 8
    search_max_iterations: int = 2
    sufficiency_threshold: float = 0.6  # provisoire — à calibrer sur courbe risque-couverture

    # --- HyDE (Hypothetical Document Embeddings, Gao et al. 2022) ---
    # Génère un passage hypothétique avant le retrieval dense pour mieux
    # capturer la sémantique des documents cibles (vs. la question abstraite).
    hyde_enabled: bool = False
    hyde_max_tokens: int = 250

    # --- Chunking hiérarchique parent-child (Gap 1) ---
    # Taille cible des blocs parents en caractères (~3-5 phrases).
    parent_chunk_size: int = 700
    # Les enfants restent citables, mais les fragments trop courts sont fusionnés
    # avec leurs voisins pour éviter des preuves du type "Setting impulses.".
    child_chunk_min_chars: int = 60
    child_chunk_max_chars: int = 650
    semantic_chunking_enabled: bool = True
    semantic_chunk_similarity_threshold: float = 0.08

    # --- Graphe de connaissances politiques (C02b, Gap 3) ---
    graph_path: Path = Path("data/political_graph.graphml")
    graph_spacy_model: str = "xx_ent_wiki_sm"

    def ensure_dirs(self) -> None:
        """Crée les dossiers de travail si absents (idempotent)."""
        for d in (self.data_dir, self.registry_dir, self.chroma_dir, self.trace_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = CompassSettings()
