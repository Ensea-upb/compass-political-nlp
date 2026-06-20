# Feuille de route

## Fonctionnalités implémentées

- ingestion de textes, URL et PDF avec OCR de repli ;
- métadonnées temporelles et filtrage historique ;
- chunking parent/enfant avec embeddings multilingues, titres, paragraphes et repli déterministe ;
- mémoire structurée SQLite et mémoire documentaire ChromaDB ;
- retrieval dense + BM25, contexte parent et reranking cross-encoder ;
- ingestion Manifesto Project avec repli vers `texts_and_annotations` lorsque le PDF original est bloqué ;
- serveur vLLM local open-weight sur Onyxia ;
- chat avec citations `[Sx]`, contexte analytique, inspection du prompt et fallback extractif ;
- routage déterministe ou LLM sélectionnable depuis l'interface ;
- politique `AnswerValidator` dépendante de la route ;
- façade scientifique du chat vers `CompassRunner`, validation C14 et sonde C15 ;
- tests unitaires, tests d'intégration et smoke test de l'architecture réelle.

## Chantier différé : corpus Manifesto mondial et chat sans contamination

Ce chantier est documenté pour une reprise ultérieure. Il n'est pas encore implémenté.

1. Produire un manifeste d'ingestion couvrant tous les pays, partis et élections disponibles dans le Manifesto Project.
2. Conserver les collections `compass_country_<iso3>` et ajouter une façade légère capable de les découvrir et de les interroger ensemble.
3. Rendre les identifiants de documents déterministes afin qu'une réingestion mette à jour les segments existants au lieu de créer des doublons UUID.
4. Persister `election_id` dans Chroma et conserver une identité canonique entre texte original et traduction.
5. Permettre le lancement du chat sans `--country`, `--party` ou `--as-of` obligatoires.
6. Ajouter un résolveur de périmètre après le routage et avant le retrieval : pays, parti, élection, date, langue et comparaison.
7. Appliquer l'ordre de priorité suivant : périmètre écrit dans la question, filtres choisis dans l'interface, puis corpus global.
8. Demander une clarification lorsqu'un parti ou une élection est ambigu, au lieu de choisir arbitrairement.
9. Empêcher la contamination : preuve enfant, parent et contexte général doivent rester liés au même document, pays, parti, élection et variante linguistique.
10. Compartimenter les comparaisons et exiger des preuves pour chaque entité comparée.
11. Rendre `AnswerValidator` sensible au périmètre de chaque affirmation.
12. Valider le dispositif sur Onyxia avant de mettre à jour le guide opérationnel mondial.

## Prochaines validations de recherche

- réindexer le corpus pilote avec les paramètres actuels de chunking ;
- annoter un jeu de requêtes et passages pour mesurer rappel, précision et qualité du reranking ;
- calibrer la taille des parents, des enfants et du pool de reranking ;
- évaluer le graphe politique sur des relations annotées ;
- calibrer le seuil de rupture sémantique sur un échantillon multilingue annoté ;
- étendre le registre de variables ;
- mesurer l'incertitude, les contradictions et la fiabilité des sources ;
- documenter un protocole reproductible de comparaison entre pays et élections.
