# Interface COMPASS Chat

COMPASS Chat est une couche conversationnelle au-dessus de la mémoire documentaire. Elle ne remplace ni l'ingestion, ni le retrieval, ni les garde-fous du pipeline.

## Chaîne de traitement

```text
question utilisateur
→ routage
→ récupération du périmètre ou retrieval documentaire
→ dense + BM25
→ contexte parent
→ cross-encoder
→ construction du prompt
→ vLLM local
→ AnswerValidator
→ réponse ou fallback extractif
```

## Routes

Trois routes sont actuellement autorisées :

- `direct_lookup` : récupération directe d'un `segment_id` ;
- `corpus_scope` : description déterministe du pays, du parti, de la date limite et du stockage actifs ;
- `evidence_query` : question politique traitée par retrieval, génération et validation stricte.

La politique de validation dépend de la route :

```text
direct_lookup → none
corpus_scope  → none
evidence_query → strict_evidence
```

Une route inconnue est refusée.

## Routage déterministe ou LLM

L'interface propose un contrôle `Routage` modifiable avant chaque question.

### Déterministe

Le système cherche d'abord un identifiant de segment, puis des formulations relatives au corpus ; toute autre question devient `evidence_query`.

### LLM

Le modèle reçoit une courte demande de classification et doit retourner exactement une route autorisée. Toute explication supplémentaire, erreur ou indisponibilité de vLLM déclenche le repli vers le routeur déterministe.

Le mode de routage ne change pas la politique de validation : celle-ci dépend de la route finalement retenue.

## Retrieval

Pour une question politique, `CountryMemory.query_documents_hybrid()` :

1. récupère des candidats par embeddings Chroma ;
2. renforce les correspondances lexicales avec BM25 ;
3. rattache le contexte parent aux segments enfants ;
4. applique `BAAI/bge-reranker-v2-m3` sur la paire question / parent + enfant ;
5. transmet les meilleurs enfants comme preuves citables.

## Structure du prompt

Le prompt distingue trois blocs :

- `ANALYTICAL_CONTEXT` : grille de lecture politique, non citable ;
- `GENERAL_CONTEXT` : contexte documentaire général, non citable ;
- `CITED_EVIDENCE` : passages `[S1]`, `[S2]`, etc., seuls autorisés à soutenir une affirmation.

Chaque preuve contient son pays, son parti, sa date, son type de document, la raison de retrieval et un extrait.

## Budget pour petit vLLM

Le profil opérationnel limite le prompt à :

- 4 preuves ;
- 1 bloc de contexte général ;
- des extraits parent et enfant tronqués ;
- 1 message d'historique compact ;
- 350 tokens de sortie.

Ces limites réduisent les erreurs `400 Bad Request` avec un contexte vLLM de 4096 tokens.

## Validation anti-hallucination

Pour `evidence_query`, `AnswerValidator` refuse une réponse qui :

- cite `[A]` ou `[Cx]` comme preuve ;
- invente un identifiant `[Sx]` absent du prompt ;
- produit une réponse politique sans citation ;
- utilise une politique de validation inconnue.

Une réponse d'insuffisance peut ne pas contenir de citation. En cas de rejet, le chat retourne les passages les plus pertinents sous forme extractive.

## Inspection du prompt

Après une réponse LLM, `Voir le prompt LLM` ouvre un onglet d'inspection. Tous les clics réutilisent ce même onglet, ce qui évite d'accumuler des pages pendant une démonstration.

La page montre :

- les messages système et utilisateur ;
- les trois blocs de contexte et de preuve ;
- les métadonnées et scores de retrieval ;
- le JSON exact envoyé à vLLM.

## Installation

```bash
pip install -r requirements-chat.txt
```

Pour le profil complet Onyxia :

```bash
pip install -r requirements-onyxia.txt
```

## Configuration

```bash
export COMPASS_DATA_DIR=$PWD/data/manifesto_ingestion
export COMPASS_CHROMA_DIR=$PWD/data/manifesto_ingestion/chroma
export COMPASS_SQLITE_PATH=$PWD/data/manifesto_ingestion/compass_structured.db

export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_ENABLED=false
export COMPASS_HF_DEVICE=cpu
export COMPASS_RERANK_ENABLED=true
export COMPASS_RERANK_POOL_SIZE=24
```

## Lancement actuel

```bash
python apps/chat_web.py \
  --country DEU \
  --as-of 2009-09-27 \
  --party 41320 \
  --port 41771
```

`chat_web.py` est recommandé sur Onyxia. Le prototype `chat_gradio.py` reste disponible mais peut rencontrer des incompatibilités avec les versions FastAPI/Starlette du runtime vLLM.

## Questions de vérification

```text
À quel corpus es-tu connecté ?
Que dit ce parti sur la démocratie ?
Quelles sont ses priorités économiques ?
Donne les preuves de sa position sur l'intégration européenne.
Je veux le passage <segment_id>.
```

## Diagnostic

- réponse extractive avec `AnswerContractError` : la sortie LLM a enfreint le contrat de preuve ;
- réponse extractive avec erreur OpenAI : vérifier vLLM et `/v1/models` ;
- passages incohérents : vérifier le pays, le parti, la date et les chemins Chroma/SQLite ;
- sources `UNK` : réindexer avec les métadonnées attendues ;
- erreur 400 : vérifier le nom du modèle, `max-model-len` et le budget de prompt.

## Limite actuelle

Le chat est encore lié à une `CountryMemory` au lancement. Le chat mondial multi-pays, le résolveur de périmètre et les protections anti-contamination sont documentés dans la roadmap mais ne sont pas encore implémentés.
