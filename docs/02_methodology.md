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

### 3. Retrieval

La recherche combine :

```text
recherche dense ChromaDB
+ BM25 lexical
+ signaux légers liés à la question
→ pool de candidats
→ contexte parent
→ reranking cross-encoder
```

### 4. Construction du prompt

Le prompt distingue :

- `ANALYTICAL_CONTEXT` : cadre conceptuel non citable ;
- `GENERAL_CONTEXT` : contexte documentaire non citable ;
- `CITED_EVIDENCE` : seules preuves autorisées, référencées par `[Sx]`.

### 5. Génération et validation

Le LLM produit une réponse courte et sourcée. `AnswerValidator` applique une politique dépendante de la route : validation stricte pour une question politique, aucune exigence `[Sx]` pour une réponse déterministe de périmètre ou un lookup direct.

### 6. Dégradation contrôlée

Si vLLM est indisponible, si le prompt est refusé ou si la réponse enfreint le contrat de citation, le chat retourne une réponse extractive construite à partir des passages récupérés.

## Limites

Le retrieval pertinent ne garantit pas à lui seul une interprétation correcte. Les résultats doivent être évalués sur un corpus annoté et relus humainement.
