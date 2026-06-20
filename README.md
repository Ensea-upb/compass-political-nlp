# COMPASS Political NLP

![tests](https://github.com/Ensea-upb/compass-political-nlp/actions/workflows/tests.yml/badge.svg)

COMPASS est un cadre de recherche pour ingérer des documents politiques, retrouver des passages pertinents et produire des analyses traçables. Le dépôt public contient le code réutilisable, des exemples synthétiques, les tests et la documentation d'exécution. Il ne contient ni documents privés, ni articles téléchargés, ni fichiers de supervision interne.

## Objectif

COMPASS vise à relier chaque conclusion politique aux documents qui la soutiennent. Le système distingue explicitement :

- le document brut et ses métadonnées ;
- les segments citables et leur contexte parent ;
- le contexte analytique utilisé pour interpréter une question ;
- les preuves autorisées à soutenir une affirmation ;
- la génération LLM et la validation de sa réponse.

Le pipeline de recherche suit cette chaîne :

```text
documents politiques
→ ingestion et datation
→ chunking parent/enfant
→ indexation vectorielle et lexicale
→ retrieval hybride
→ reranking par cross-encoder
→ génération contrôlée
→ validation et traçabilité
```

## Démarrage rapide

La démo légère fonctionne sans clé API et sans modèle volumineux :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
python examples/run_demo.py
```

Résultat attendu :

```text
Document loaded: sample_manifesto.txt
Detected political themes: economy, sovereignty, democracy
Generated party profile: examples/sample_party_profile.json
Validation status: passed
```

Pour vérifier les contrats de l'architecture réelle sans télécharger les modèles :

```powershell
pip install -r requirements-test.txt
python examples/run_real_architecture.py smoke
python -m pytest
```

## Exécution complète et Onyxia

Deux profils sont prévus :

```powershell
# environnement local complet
pip install -r requirements-full.txt

# environnement Onyxia avec vLLM
pip install -r requirements-onyxia.txt
```

Guides associés :

- [exécution pas à pas sur Onyxia](docs/09_onyxia_runbook.md) ;
- [modèles Hugging Face et serveur vLLM](docs/11_onyxia_hf_models.md) ;
- [ingestion du Manifesto Project](docs/12_manifesto_pdf_ingestion.md) ;
- [interface COMPASS Chat](docs/13_chat_interface.md) ;
- [validation du RAG expert sur Onyxia](docs/14_rag_expert_onyxia.md).

## Deux niveaux de démonstration

`compass.demo` constitue une démonstration déterministe et synthétique. Elle sert à vérifier rapidement l'installation.

Les modules de `src/compass/` constituent l'architecture de recherche : ingestion PDF/OCR, mémoire ChromaDB, retrieval hybride, raisonnement, juges, agrégation, validation et garde-fous. La démo légère ne remplace pas cette architecture.

## Organisation du dépôt

```text
compass-political-nlp/
├── apps/          interfaces de conversation
├── assets/        schémas et illustrations
├── docs/          documentation scientifique et opérationnelle
├── examples/      démonstrations et données synthétiques
├── registry/      fiches de variables politiques
├── scripts/       préparation des corpus et modèles
├── src/compass/   composants du système
└── tests/         tests unitaires et d'intégration
```

## État actuel

Le dépôt implémente notamment :

- le chunking hiérarchique parent/enfant avec embeddings multilingues, titres et provenance structurelle ;
- l'indexation par pays dans ChromaDB ;
- le retrieval dense + BM25 et le reranking cross-encoder ;
- l'analyse structurée des questions par LLM local, avec JSON strict, sous-requêtes et fallback déterministe ;
- le retrieval expert en trois lanes, avec diversité, filtrage de périmètre et trace inspectable ;
- la génération citée avec validation NLI phrase par phrase et une tentative de réparation ;
- l'ingestion Manifesto Project avec repli vers `texts_and_annotations` lorsque le PDF est bloqué ;
- un chat RAG avec citations `[Sx]`, page d'inspection du prompt, routage sélectionnable et validation dépendante de la route ;
- une façade scientifique du chat vers `CompassRunner` via `/analyse <variable_id>`, avec validation C14 séparée ;
- un profil Onyxia validé avec un modèle local open-weight de 3 milliards de paramètres.

Le corpus Manifesto mondial et le chat multi-pays sans contamination sont inscrits comme chantier différé dans la [roadmap](docs/06_roadmap.md). Ils ne sont pas encore implémentés.

## Politique de données

Les données téléchargées et les index sont placés sous `data/`, ignoré par Git. Les clés `MANIFESTO_API_KEY` et `HF_TOKEN` doivent rester dans les variables d'environnement. Aucun secret ne doit être ajouté au dépôt.

## Avertissement scientifique

COMPASS est un projet académique. Une réponse générée, un score ou un profil politique doit être contrôlé humainement et validé avant toute interprétation substantielle.

## Auteur

Inza Ouada Soro  
ENSAE Paris  
Projet de stage en NLP, économie politique et sciences sociales computationnelles.
