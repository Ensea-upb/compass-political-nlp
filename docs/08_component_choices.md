# CHOIX_COMPOSANTS — sélection des briques sur preuves web

**Date des recherches** : 2026-06-11 (6 requêtes web ; sources en bas de chaque section).
**Pourquoi ce fichier** : la première version du squelette choisissait les briques de mémoire (sélection raisonnable mais non vérifiée contre l'état de l'art courant). Ce document corrige cela : chaque choix est confronté aux benchmarks 2025–2026, avec **deux critères de premier rang propres à COMPASS** : (i) couverture multilingue réelle, y compris langues africaines à faibles ressources ; (ii) robustesse à la variation de style rhétorique (valence, registres oratoires, discours non occidentaux — le cœur du constat Bleck & van de Walle).

**Méthode de décision** : à performance comparable, on préfère la brique (a) multilingue par construction, (b) open source auto-hébergeable (reproductibilité, Claude.md §10), (c) au coût d'inférence raisonnable pour un pilote. Les modèles « plafond » plus lourds sont notés comme variantes d'ablation, pas comme défauts.

---

## 1. Embeddings (C02, C03, C06)

**Décision : `BAAI/bge-m3`** (remplace `paraphrase-multilingual-mpnet-base-v2`, 2021 — dépassé).

Justification : le MTEB multilingue 2026 (MMTEB : 131 tâches, 250+ langues, agrégation Borda qui récompense la consistance inter-langues — exactement notre besoin) place en tête des modèles lourds (Qwen3-Embedding-8B ~70.6, Llama-Embed-Nemotron-8B, KaLM-Gemma3-12B) ; mais à taille déployable, **BGE-M3 est décrit comme le standard de production open source multilingue (100+ langues)**, avec retrieval dense + sparse + multi-vecteur dans un seul modèle — utile pour des registres rhétoriques variés où le lexical pur (BM25) et le sémantique se complètent.

Alternatives notées : `multilingual-e5-large` (proche, plus léger) ; `Qwen3-Embedding-8B` comme **variante d'ablation « plafond »** au pilote (si le gain sur les langues africaines justifie le coût GPU) ; Jina v3 pour les documents longs (late chunking).

Point de vigilance honnête : aucun des leaderboards consultés ne ventile finement les langues africaines subsahariennes ; la performance réelle sur swahili/haoussa/yoruba devra être mesurée au pilote (échantillon de paires requête-passage annotées) — le leaderboard ne nous en dispense pas.

Sources : [MTEB rankings 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-april-2026/), [comparatif 2026](https://app.ailog.fr/en/blog/news/embedding-models-2026), [analyse MTEB](https://modal.com/blog/mteb-leaderboard-article).

## 2. Reranker (C06)

**Décision : `BAAI/bge-reranker-v2-m3`** (remplace `mmarco-mMiniLMv2`, 2022 — dépassé).

Justification : les comparatifs 2026 le donnent comme « le bon défaut production : meilleur combiné qualité / latence / licence ». `jina-reranker-v3` fait +5,4 % à taille égale (MIRACL 66.5 sur 18 langues) — à tester en ablation si le re-ranking s'avère être un goulot de qualité, mais licence et déploiement moins simples.

Sources : [guide rerankers 2026](https://localaimaster.com/blog/reranking-cross-encoders-guide), [jina-reranker-v3](https://jina.ai/models/jina-reranker-v3/), [benchmark rerankers](https://aimultiple.com/rerankers).

## 3. NLI multilingue — contradictions (C09), attribution (C13), zero-shot (C10)

**Décision : `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` — CONFIRMÉ** (le choix de mémoire résiste à la vérification).

Justification : la littérature 2025 sur la stance detection cross-lingue retient mDeBERTa-v3 contre XLM-R et mBERT, « particulièrement pour les langues sous-représentées », attention désentrelacée adaptée à la nuance idéologique. **Ajout issu de la recherche** : pour l'anglais, `Political_DEBATE` (Burnham et al., *Political Analysis* 2025) — DeBERTa entraîné spécifiquement au zero/few-shot sur textes politiques — est meilleur que le NLI générique ; il devient le classifieur C10 pour les documents anglais, avec bascule automatique vers mDeBERTa-XNLI hors anglais (routage par langue, champ `language` de C01).

Vigilance rhétorique : aucun de ces modèles n'a été évalué sur la valence africaine ; le test d'adhérence R-1 (C05) sert aussi à détecter ce désalignement stylistique variable par variable.

Sources : [Political DEBATE, PA 2025](https://www.cambridge.org/core/journals/political-analysis/article/political-debate-efficient-zeroshot-and-fewshot-classifiers-for-political-text/8D0B3E2AAF711F4812E42466DE503A13), [stance cross-lingue 2025](https://peerj.com/articles/cs-2955/), [survey stance/LLM](https://arxiv.org/abs/2505.08464).

## 4. OCR (C01)

**Décision : Tesseract par défaut, `Surya OCR` pour les scans dégradés** (option `surya-ocr` dans requirements).

Justification : les comparatifs 2026 donnent Surya (90+ langues, détection ligne à ligne, layout) au-dessus de Tesseract sur la plupart des benchmarks ; Tesseract reste sensible au flou et aux gradients d'éclairage — précisément le profil des manifestes africains scannés. PaddleOCR-VL-1.5 (109 langues, tableaux/formules) est l'alternative lourde ; Docling (IBM) pour la structure documentaire mais faible sur scans. Stratégie : Tesseract en première passe (léger, déjà installé), Surya en seconde passe sur les pages où la confiance Tesseract est basse.

Sources : [comparatif OCR Python 2026](https://www.codesota.com/ocr/best-for-python), [outils OCR open source 2026](https://unstract.com/blog/best-opensource-ocr-tools/), [PaddleOCR vs Tesseract](https://www.codesota.com/ocr/paddleocr-vs-tesseract).

## 5. Datation des pages web (C08 → C01)

**Décision : `htmldate`** — comble le maillon le plus faible identifié dans l'auto-critique du squelette (`_safe_date` assignait la date d'élection par défaut : trou dans le contrôle temporel).

Justification : htmldate (publié JOSS, même auteur que trafilatura) bat les alternatives (articleDateExtractor, date_guesser) en précision ET en vitesse, avec une couverture explicitement meilleure sur les petits sites non anglophones — le profil exact de la presse locale africaine. Règle implémentée : date htmldate si trouvée ; sinon marquage `_undated` + pénalisation au diagnostic. Jamais d'antériorité présumée.

Sources : [évaluation htmldate](https://htmldate.readthedocs.io/en/latest/evaluation.html), [comparatif datation](https://adrien.barbaresi.eu/blog/evaluation-date-extraction-python.html), [papier JOSS](https://www.theoj.org/joss-papers/joss.02439/10.21105.joss.02439.pdf).

## 6. Juges LLM (C11) — couverture linguistique du panel

**Décision : panel hétérogène inchangé (GPT / Mistral / Claude via litellm), MAIS sélection finale conditionnée à un benchmark langues africaines au pilote, avec `AfroBench` comme grille.**

Justification : la recherche confirme que la couverture africaine des LLM généralistes est inégale et mal documentée ; les ressources dédiées existent désormais pour la mesurer — AfroBench (15 tâches, 64 langues africaines), IrokoBench — et des modèles adaptés émergent (Aya Expanse 8B/32B, 23 langues ; InkubaLM 0.4B pré-entraîné sur isiXhosa/isiZulu/swahili/haoussa/yoruba ; Lugha-Llama). Conséquence opérationnelle : au pilote, exécuter les juges candidats sur le sous-ensemble AfroBench des langues du corpus AVANT de figer le panel ; Aya Expanse entre comme 4e juge candidat si le corpus contient des langues qu'il couvre. La variation rhétorique est traitée par la variante de prompt `behavior_first` (C10) et par le test d'adhérence par variable — pas par une confiance aveugle dans le multilinguisme déclaré des modèles.

Sources : [AfroBench](https://arxiv.org/html/2311.07978), [état des LLM langues africaines 2025](https://arxiv.org/html/2506.02280v3), [Lugha-Llama](https://arxiv.org/pdf/2504.06536), [évaluation LLM langues africaines](https://arxiv.org/pdf/2502.19582).

---

## Récapitulatif des changements de code (2026-06-11)

| Fichier | Changement |
|---|---|
| `config.py` | `embedding_model` → bge-m3 ; ajout `reranker_model` (bge-reranker-v2-m3) et `political_classifier` (Political DEBATE, anglais) |
| `c01_pipeline_documentaire.py` | htmldate dans `ingest_url` (datation réelle + marquage `_undated`) ; option Surya documentée |
| `c06_retrieval_interne.py` | reranker lu depuis la config (bge-reranker-v2-m3) |
| `c08_recherche_active.py` | `_safe_date` rétrogradée en valeur d'amorçage — la datation fait foi en C01 |
| `requirements-full.txt` | + htmldate ; + surya-ocr (optionnel) |

**Règle pour la suite** : tout nouveau composant ou remplacement passe par ce fichier — recherche web datée, alternatives, critère multilingue/rhétorique explicite. Les choix de modèles se périment vite ; la procédure, non.
