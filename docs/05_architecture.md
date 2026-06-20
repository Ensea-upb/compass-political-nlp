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

## Graphe politique

L'ingestion alimente également `PoliticalGraph` à partir des chunks parents. Les entités nommées sont extraites par spaCy et les cooccurrences sont enregistrées comme relations **inférées**, avec date, parti, pays et segment source.

```text
segments parents
→ extraction d'entités spaCy
→ relations de cooccurrence typées
→ political_graph_<iso3>.graphml
→ query_party(party_id, as_of)
→ contexte relationnel inféré
```

Un fichier GraphML distinct est utilisé par pays. Les identifiants de segments déjà traités sont persistés afin qu'une réingestion ne double pas les relations. Dans le pipeline complet, ce contexte rejoint `Diagnosis.graph_context`. Dans le chat, il n'est chargé que pour une question relationnelle et ne peut jamais être cité comme preuve `[Sx]`.

## Chunking et retrieval

Le document est découpé en parents contextuels et enfants citables. Le chat recherche les enfants, rattache leur parent, fusionne dense et BM25, puis applique le cross-encoder configuré.

## Chat

Le chat est la façade des deux usages. Les questions documentaires utilisent le RAG hybride. Les commandes `/analyse <variable_id>` délèguent au même `CompassRunner` que le pipeline de recherche : registre, suffisance, recherche active, diagnostic, raisonnement, juges, agrégation, sortie et traçabilité. La validation externe C14 reste une commande séparée afin d'isoler physiquement l'étalon.

## État du corpus mondial

Les collections restent actuellement séparées par pays et le chat est lancé avec un pays obligatoire. La façade multi-pays et les protections anti-contamination sont un chantier différé, décrit dans la roadmap.
