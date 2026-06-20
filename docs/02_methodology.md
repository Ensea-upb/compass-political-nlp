# Méthodologie

## Principe central

COMPASS distingue systématiquement la preuve, le contexte et l'interprétation. Le contexte aide à comprendre un passage, mais seules les unités identifiées comme preuves peuvent soutenir les affirmations finales.

## Étapes

### 1. Ingestion

Chaque document reçoit des métadonnées : pays, parti, élection, date, langue, type de source, fiabilité et origine. La date utilisée pour le raisonnement historique doit être vérifiée.

### 2. Chunking hiérarchique

Le pipeline produit :

- des parents, blocs thématiques conservant le contexte local ;
- des enfants, unités plus courtes destinées au retrieval et aux citations.

Les fragments trop courts sont fusionnés et les fragments trop longs sont divisés. Les titres de section, les limites de paragraphes et l'ordre de lecture sont conservés avec chaque segment, en plus du document, du parti, de la date et de la langue.

La rupture thématique repose sur des embeddings multilingues calculés en une passe sur les unités du document. Le pipeline compare l'unité suivante au contexte récent du parent et ouvre un nouveau parent lorsque leur similarité cosinus devient trop faible. Si le modèle d'embeddings est absent ou échoue, un repli lexical déterministe maintient l'ingestion sans masquer l'incident dans les journaux.

### 3. Analyse de la question

Avant toute recherche, le chat produit une représentation structurée de la demande : acteurs, thèmes, période, type de réponse, langue et sous-requêtes complémentaires. Le LLM local doit retourner un objet JSON strict et n'est jamais autorisé à répondre à la question pendant cette étape. Un JSON invalide, un serveur indisponible ou un acteur inventé déclenche une analyse déterministe sans dictionnaire thématique spécialisé.

La question originale reste toujours la première requête afin de préserver les termes exacts de l'utilisateur. Les reformulations ajoutent des angles conceptuels et orientés vers les preuves sans introduire de position politique supposée.

### 4. Retrieval

La recherche combine :

```text
recherche dense ChromaDB
+ BM25 lexical
+ signaux légers liés à la question
→ pool de candidats
→ contexte parent
→ reranking cross-encoder
```

Chaque sous-requête traverse dense + BM25. Le parent est rattaché avant le reranking afin que le cross-encoder évalue `requête × (parent + segment)`. Les classements sont fusionnés, dédupliqués et diversifiés. Trois recherches distinctes produisent les preuves principales, les nuances et les contre-preuves candidates. Un filtre final vérifie pays, parti, élection, date et statut temporel avant toute transmission au prompt.

### 5. Construction du prompt

Le prompt distingue :

- `ANALYTICAL_CONTEXT` : cadre conceptuel non citable ;
- `GENERAL_CONTEXT` : contexte documentaire non citable ;
- `CITED_EVIDENCE` : seules preuves autorisées, référencées par `[Sx]`.

`CITED_EVIDENCE` est lui-même divisé en `PRIMARY_EVIDENCE`, `NUANCE_EVIDENCE` et `COUNTER_EVIDENCE_CANDIDATES`. Le contexte général sélectionne plusieurs parents appartenant à des sections différentes. Son budget, celui des parents et celui des preuves sont calculés depuis la fenêtre de contexte déclarée pour vLLM.

### 6. Génération et validation

Le LLM produit une réponse directe et sourcée. Il doit distinguer une déclaration explicite d'une synthèse prudente. `AnswerValidator` vérifie chaque affirmation et ses identifiants `[Sx]`. La validation NLI confronte ensuite chaque phrase aux passages cités, individuellement puis conjointement lorsque plusieurs sources soutiennent une synthèse.

Si une affirmation échoue, le même LLM reçoit son brouillon, les erreurs et exactement le même paquet de preuves. Une seule réparation est tentée par défaut, sans nouveau retrieval. Le fallback extractif n'intervient qu'après l'échec de cette correction. Les affirmations acceptées et rejetées restent dans `validation_trace`.

### 7. Dégradation contrôlée

Si vLLM est indisponible, si le prompt est refusé ou si la réponse enfreint le contrat de citation, le chat retourne une réponse extractive construite à partir des passages récupérés.

## Limites

Le retrieval pertinent ne garantit pas à lui seul une interprétation correcte. Les résultats doivent être évalués sur un corpus annoté et relus humainement.
