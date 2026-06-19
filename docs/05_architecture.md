# Architecture du système

## Démo légère

```text
examples/sample_manifesto.txt
→ src/compass/demo.py
→ examples/sample_party_profile.json
```

Cette démo déterministe sert uniquement à vérifier l'installation.

## Pipeline de recherche

```text
document brut
→ document_pipeline.py
→ general_memory.py / country_memory.py / political_graph.py
→ party_election_case.py
→ internal_retrieval.py
→ sufficiency_gate.py ↔ active_search.py
→ diagnostic_engine.py
→ reasoning_engine.py / judge_panel.py
→ aggregation.py
→ final_output.py
→ validation.py / guardrails.py
```

`config.py` centralise les paramètres et `schemas.py` définit les contrats échangés entre les composants.

## Mémoire documentaire

`CountryMemory` combine SQLite pour les données structurées et ChromaDB pour les passages. Une collection `compass_country_<iso3>` est créée par pays. Les filtres temporels utilisent `doc_date_ord`, valeur numérique compatible avec les comparaisons Chroma.

## Chunking et retrieval

Le document est découpé en parents contextuels et enfants citables. Le chat recherche les enfants, rattache leur parent, fusionne dense et BM25, puis applique le cross-encoder configuré.

## Chat

Le chat route d'abord la requête : lookup direct, question sur le périmètre du corpus ou question politique. Le routage peut être déterministe ou confié au LLM avec repli automatique. La politique de validation dépend ensuite de la route.

## État du corpus mondial

Les collections restent actuellement séparées par pays et le chat est lancé avec un pays obligatoire. La façade multi-pays et les protections anti-contamination sont un chantier différé, décrit dans la roadmap.
