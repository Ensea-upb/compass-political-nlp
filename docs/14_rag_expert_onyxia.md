# Validation du RAG expert sur Onyxia

## Objectif

Ce protocole valide séparément le nouveau chunking, le retrieval expert et la génération réparatrice. Il utilise un nouveau répertoire de données afin de ne pas détruire l'index actuellement démontrable.

## 1. Mettre le code à jour

```bash
cd ~/work/compass-political-nlp
git pull
source .venv/bin/activate
pip install -r requirements-onyxia.txt
```

Le chunking sémantique télécharge automatiquement `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` lors de sa première utilisation. Si ce modèle est indisponible, le journal indique le repli lexical déterministe.

## 2. Créer un index isolé

```bash
export COMPASS_DATA_DIR=$PWD/data/manifesto_ingestion_rag_v2
export COMPASS_CHROMA_DIR=$COMPASS_DATA_DIR/chroma
export COMPASS_SQLITE_PATH=$COMPASS_DATA_DIR/compass_structured.db
export COMPASS_GRAPH_PATH=$COMPASS_DATA_DIR/political_graph.graphml
export COMPASS_TRACE_DIR=$COMPASS_DATA_DIR/traces
```

Ne pas réutiliser l'ancien dossier `manifesto_ingestion` pour ce premier essai. Cette séparation permet de revenir immédiatement à la démonstration précédente.

## 3. Réindexer le corpus pilote

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --manifest data/manifests/deu_2009.csv \
  --translation en
```

Vérifier dans le rapport que les documents ont été indexés. Les nouveaux segments doivent porter `election_id`, `section_title`, `paragraph_start`, `paragraph_end` et `chunk_index`. Le filtre strict du chat refuse désormais une preuve appartenant à une autre élection ou une preuve sans `election_id` lorsqu'une élection est demandée.

## 4. Configurer vLLM et COMPASS

Le contexte déclaré à COMPASS doit rester inférieur ou égal à `--max-model-len` de vLLM.
Le profil expert exige une fenêtre d'au moins 4096 tokens.

```bash
export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu

export COMPASS_CHAT_LLM_CONTEXT_WINDOW=4096
export COMPASS_CHAT_PROMPT_RESERVED_OUTPUT_TOKENS=500
export COMPASS_CHAT_QUERY_ANALYSIS_ENABLED=true
export COMPASS_CHAT_SEMANTIC_VALIDATION_ENABLED=true
export COMPASS_CHAT_REPAIR_MAX_ATTEMPTS=1
export COMPASS_CHAT_STRICT_ELECTION_SCOPE=true
```

Exemple de serveur :

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

## 5. Vérifier le serveur

```bash
curl http://localhost:8000/v1/models
```

Le nom retourné doit correspondre à `COMPASS_JUDGE_MODELS`.

## 6. Lancer le chat

```bash
python apps/chat_web.py \
  --country DEU \
  --party 41320 \
  --election-id DEU_2009 \
  --as-of 2009-09-27 \
  --port 41771
```

Après chaque réponse, ouvrir `Voir le prompt LLM` et vérifier :

- `QUESTION_ANALYSIS` et les sous-requêtes ;
- `RETRIEVAL_TRACE` et les rejets de périmètre ;
- `GENERAL_CONTEXT` composé de plusieurs parents ;
- `PRIMARY_EVIDENCE`, `NUANCE_EVIDENCE` et `COUNTER_EVIDENCE_CANDIDATES` ;
- les citations `[Sx]` dans chaque affirmation politique.

## 7. Produire un snapshot de référence

```bash
python scripts/evaluate_chat_rag.py \
  --country DEU \
  --party 41320 \
  --election-id DEU_2009 \
  --as-of 2009-09-27 \
  --output outputs/rag_expert_deu_2009_v2.json
```

Pour comparer une exécution ultérieure :

```bash
python scripts/evaluate_chat_rag.py \
  --country DEU \
  --party 41320 \
  --election-id DEU_2009 \
  --as-of 2009-09-27 \
  --baseline outputs/rag_expert_deu_2009_v2.json \
  --output outputs/rag_expert_deu_2009_after.json
```

Le rapport conserve les réponses, les segments, les rôles de preuve, le rappel lexical indicatif, la latence, le plan de requête et les traces de validation. Ce jeu n'est pas un étalon scientifique annoté : il sert à détecter les régressions et à préparer l'annotation humaine.

## 8. Critères de sortie

- aucune preuve hors pays, parti, élection ou date ;
- présence d'au moins une preuve principale sur les questions couvertes ;
- absence de doublons proches dans les premières sources ;
- chaque affirmation politique porte `[Sx]` ;
- une affirmation rejetée par NLI est corrigée ou déclenche le fallback ;
- aucune erreur HTTP 400 avec la fenêtre configurée ;
- rapport JSON écrit dans `outputs/`.
