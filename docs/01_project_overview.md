# Vue d'ensemble du projet

COMPASS Political NLP est un système de recherche destiné à analyser des manifestes électoraux et d'autres documents politiques tout en conservant une chaîne de preuve vérifiable.

## Problème traité

Les documents politiques sont multilingues, de qualité inégale et parfois disponibles uniquement sous forme de PDF scanné. Une analyse produite directement par un LLM peut mélanger des sources, introduire des connaissances extérieures ou formuler des conclusions non soutenues.

COMPASS sépare donc quatre opérations :

1. constituer et dater le corpus ;
2. découper les documents en unités citables avec leur contexte ;
3. retrouver et classer les preuves pertinentes ;
4. générer puis valider une réponse liée aux passages récupérés.

## Utilisateurs visés

- chercheurs en science politique et économie politique ;
- équipes travaillant sur les manifestes, les élections et les partis ;
- étudiants souhaitant expérimenter un pipeline RAG politique traçable ;
- ingénieurs NLP travaillant sur des corpus multilingues.

## Deux usages distincts

La démo synthétique permet de vérifier le dépôt sans infrastructure lourde. L'architecture de recherche utilise les composants réels : OCR, ChromaDB, embeddings, BM25, cross-encoder, NLI, vLLM et validation.

## Périmètre actuel

Le chat opérationnel est lancé sur une mémoire pays, avec un filtre temporel obligatoire et un filtre parti facultatif. Le passage à un corpus Manifesto mondial est documenté mais différé ; voir `docs/06_roadmap.md`.
