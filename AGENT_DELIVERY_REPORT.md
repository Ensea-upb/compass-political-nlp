# Rapport de livraison Codex — audit COMPASS Chat

Date : 19 juin 2026  
Référence auditée : `9fa70fa`  
Statut : **GO_WITH_WARNINGS**

## Résultat

Les corrections logiques et de démonstration demandées ont été intégrées sans refondre le pipeline C01-C15. Le chat reste une façade RAG au-dessus de `CountryMemory`.

## Corrections du chat

- ajout de la règle closed-world : une absence de preuve ne peut plus être présentée comme une preuve d'absence ;
- calcul du périmètre depuis les métadonnées réellement présentes dans la mémoire active ;
- réponse `corpus_scope` enrichie : pays, partis, noms disponibles, documents distincts, types, dates et borne `as_of` ;
- avertissement explicite indiquant que `as_of` n'est pas nécessairement la date du scrutin ;
- lookup direct conservant les métadonnées, avec avertissement visible pour les anciens index qui ne les exposent pas ;
- sources précédentes transportées sous forme structurée dans le payload, sans extraction fragile depuis le texte de la réponse ;
- payload enrichi avec `sources_markdown`, `route`, `retrieval_count` et `prompt_citation_count` ;
- distinction visible entre candidats récupérés, preuves envoyées au LLM et sources affichées ;
- seuil d'extrait porté à 420 caractères et rendu configurable ;
- validation NLI optionnelle ajoutée derrière un drapeau, désactivée par défaut ;
- sélecteur de routage LLM masqué par défaut et disponible avec `--debug-routing` ;
- bandeau du corpus actif et exemples de questions ajoutés à l'interface.

## Routes et politiques de validation

| Route | Politique | Fonction |
|---|---|---|
| `direct_lookup` | `none` | passage demandé par identifiant exact |
| `corpus_scope` | `none` | description du corpus réellement actif |
| `evidence_query` | `strict_evidence` | réponse LLM fondée sur les preuves `[Sx]` |
| `FOLLOW_UP_SOURCES` | `none` | preuves structurées de la réponse précédente |
| `OUT_OF_CORPUS` | `none` | demande exhaustive non couverte |
| `COMPARISON_NEEDS_MORE_CORPUS` | `none` | comparaison entre au moins deux partis identifiés |
| `ELECTION_CONTEXT_NEEDS_STRUCTURED_DATA` | `none` | résultats, sièges, participation, vainqueur ou gouvernement |

Les routes de périmètre sont consultatives : elles décrivent les données disponibles et les données manquantes, sans inventer de réponse et sans refus opaque.

## Noyau et crédibilité du dépôt

- docstring de l'orchestrateur alignée sur le comportement fatal réel des garde-fous ;
- mémoire générale limitée aux portées techniques `GEN`/`GLOBAL`, sans parti ;
- documentation du pipeline alignée sur la segmentation déterministe réellement utilisée ;
- import NumPy manquant corrigé dans `aggregation.py` ;
- tests directs ajoutés pour l'agrégation, la suffisance, le diagnostic, la temporalité et la mémoire générale ;
- Ruff ajouté aux dépendances de test et à GitHub Actions.

## Intégration du graphe politique

- l'ingestion PDF ou `texts_and_annotations` appelle désormais `PoliticalGraph.ingest()` puis `save()` ;
- le mode `run_real_architecture.py full` charge, alimente et transmet le graphe à `CompassRunner` ;
- `chat_web.py` et `chat_gradio.py` chargent automatiquement le graphe du pays actif ;
- les questions relationnelles ajoutent un bloc `RELATIONAL_CONTEXT` au prompt ;
- les relations `[Rx]` sont explicitement inférées, non citables et rejetées par `AnswerValidator` si le LLM tente de les utiliser comme preuve ;
- les fichiers GraphML sont isolés par pays ;
- les segments déjà ingérés sont mémorisés, ce qui rend la mise à jour idempotente ;
- chaque arête conserve le pays, le parti, la date et le segment source.
- `scripts/build_political_graph.py` permet de construire le graphe depuis un index Chroma existant, sans réingérer ni dupliquer les manifestes.

## Chat comme façade scientifique

- les questions libres conservent le chemin RAG rapide et cité ;
- `/variables` expose uniquement les fiches ayant passé la gate d'adhérence ;
- `/analyse <variable_id>` délègue au vrai `CompassRunner` et exécute C04-C15 ;
- la recherche active alimente aussi le graphe lorsqu'elle découvre de nouveaux segments ;
- la réponse scientifique expose score ou abstention, confiance, preuves, contre-preuves, attribution NLI, incertitude et trace C15 ;
- `/valider [variable_id]` exécute C14 sur les réponses de la session, depuis le coffre séparé ;
- `/contamination <variable_id>` expose explicitement la sonde C15 sans influencer le raisonnement de production ;
- le service est initialisé paresseusement : les modèles scientifiques lourds ne sont chargés qu'à la première analyse.

## Vérifications exécutées

```text
python -m pytest -q
132 passed

ruff check src apps tests examples scripts
All checks passed!

python examples/run_demo.py
Validation status: passed

python examples/run_real_architecture.py smoke
Smoke status: passed
```

Une recherche explicite dans `src/` et `apps/` a également vérifié l'absence des identifiants de l'ancien corpus de démonstration dans les réponses du chat. Les informations de périmètre proviennent de `CountryMemory.describe_corpus()`.

## Risques restants

1. La validation NLI est désactivée par défaut. Son activation augmente le contrôle sémantique mais peut produire des faux négatifs et augmenter la latence.
2. Le détecteur déterministe de comparaison s'appuie sur les partis connus de la mémoire et sur la forme de la question. Le mode LLM reste disponible pour diagnostic, pas comme dépendance obligatoire.
3. La démonstration États-Unis nécessite toujours l'ingestion réelle d'un corpus USA sur Onyxia. Elle n'est pas vérifiable depuis cet environnement sans accès à l'API Manifesto et aux données persistées.
4. `chat_gradio.py` reste un prototype optionnel. L'interface recommandée est `chat_web.py`.
5. L'extraction du graphe exige un modèle NER spaCy installé. Sans modèle, l'ingestion documentaire continue mais le graphe reste vide et un avertissement est journalisé.

## Verdict

**GO_WITH_WARNINGS** : le code et les tests sont prêts pour une démonstration fondée sur un corpus déjà indexé. Le seul prérequis externe bloquant pour la démonstration USA reste l'ingestion effective de ce corpus sur Onyxia.
