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

Sept routes sont actuellement autorisées :

- `direct_lookup` : récupération directe d'un `segment_id` ;
- `corpus_scope` : description déterministe du pays, du parti, de la date limite et du stockage actifs ;
- `evidence_query` : question politique traitée par retrieval, génération et validation stricte.
- `FOLLOW_UP_SOURCES` : restitution des preuves structurées de la réponse précédente ;
- `OUT_OF_CORPUS` : demande exhaustive que le corpus documentaire actif ne peut établir ;
- `COMPARISON_NEEDS_MORE_CORPUS` : comparaison explicite entre au moins deux partis ;
- `ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA` : résultats, sièges, participation, vainqueur ou gouvernement.

Les trois dernières routes ne produisent pas un refus sec. Elles décrivent ce que le corpus actif permet d'établir, puis indiquent les données manquantes. Le pays, les partis, les dates, les types de documents et leur nombre sont lus dans la `CountryMemory` active : aucune réponse de périmètre n'est écrite en dur pour une démonstration particulière.

La politique de validation dépend de la route :

```text
direct_lookup → none
corpus_scope  → none
evidence_query → strict_evidence
FOLLOW_UP_SOURCES → none
OUT_OF_CORPUS → none
COMPARISON_NEEDS_MORE_CORPUS → none
ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA → none
```

Une route inconnue est refusée.

## Routage déterministe ou LLM

Le routage déterministe est utilisé par défaut. Le contrôle permettant de passer au routage LLM est masqué pendant une démonstration normale ; il apparaît uniquement avec l'option `--debug-routing`.

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

Le profil opérationnel distingue trois quantités affichées sous chaque réponse :

- `retrieval_count` : passages récupérés et classés ;
- `prompt_citation_count` : preuves réellement transmises au LLM ;
- sources affichées : exactement les preuves que le LLM a reçues, jamais les candidats supplémentaires.

Le prompt est limité à :

- 4 preuves ;
- 1 bloc de contexte général ;
- des extraits de preuve configurables, à 420 caractères par défaut ;
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

La règle closed-world interdit également de transformer une absence de preuve en preuve d'absence. Si les extraits ne mentionnent qu'un acteur, le modèle peut constater cette limite mais ne peut pas conclure que cet acteur était le seul.

Une vérification NLI phrase-preuve peut être activée avec `COMPASS_CHAT_SEMANTIC_VALIDATION_ENABLED=true`. Elle est désactivée par défaut, car un modèle NLI générique peut produire des faux négatifs sur des paraphrases politiques valides. En cas d'échec lorsqu'elle est activée, le système retourne le fallback extractif.

## Inspection du prompt

Après une réponse LLM, `Voir le prompt LLM` ouvre un onglet d'inspection. Tous les clics réutilisent ce même onglet, ce qui évite d'accumuler des pages pendant une démonstration.

La page montre :

- les messages système et utilisateur ;
- les trois blocs de contexte et de preuve ;
- les métadonnées et scores de retrieval ;
- le JSON exact envoyé à vLLM.

## Installation

L'interface stable `apps/chat_web.py` utilise le serveur HTTP de la bibliothèque standard. Elle dépend des composants COMPASS installés par `requirements-full.txt` ou `requirements-onyxia.txt`, mais pas de Gradio.

```bash
pip install -r requirements-full.txt
```

Le fichier `requirements-chat.txt` n'est requis que pour le prototype optionnel `apps/chat_gradio.py`.

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
export COMPASS_CHAT_MAX_PROMPT_CITATIONS=4
export COMPASS_CHAT_MAX_EVIDENCE_TEXT_CHARS=420
export COMPASS_CHAT_SEMANTIC_VALIDATION_ENABLED=false
export COMPASS_CHAT_NLI_ENTAILMENT_THRESHOLD=0.65
```

## Lancement actuel

```bash
python apps/chat_web.py \
  --country "$COUNTRY_ISO3" \
  --as-of "$CUTOFF_DATE" \
  --party "$PARTY_ID" \
  --port 41771
```

`chat_web.py` est recommandé sur Onyxia. Ajouter `--debug-routing` uniquement pour examiner ou comparer le routeur LLM. Le prototype `chat_gradio.py` reste disponible mais peut rencontrer des incompatibilités avec les versions FastAPI/Starlette du runtime vLLM.

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
