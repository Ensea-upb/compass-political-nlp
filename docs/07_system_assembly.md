# ASSEMBLAGE — Guide de montage du système COMPASS Expert-Pays

**Philosophie** : intégration aéronautique. Chaque composant est construit et **réceptionné isolément** (test de réception = banc d'essai), puis intégré **par sous-ensembles**, puis le système complet passe des **essais au sol** (pilote) avant tout « vol » (production). On ne monte jamais une pièce non réceptionnée, et on ne soude rien : chaque interface passe par les schémas de `schemas.py`.

**Règle d'approvisionnement** : chaque composant réutilise une brique d'état de l'art identifiée (voir l'en-tête de chaque `.py`). Le custom est limité à l'orchestration et aux protocoles métier sans équivalent existant — chaque exception est justifiée dans le docstring du module.

---

## Vue d'ensemble des pièces

| Pièce | Fichier | Brique état de l'art principale | Custom résiduel |
|---|---|---|---|
| Colonne vertébrale | `config.py`, `schemas.py` | pydantic / pydantic-settings | contrats d'interface |
| C01 Pipeline documentaire | `c01_pipeline_documentaire.py` | PyMuPDF, pytesseract, trafilatura, lingua, spaCy | contrat de métadonnées |
| C02 Mémoire générale | `c02_memoire_generale.py` | ChromaDB + sentence-transformers | interdiction des faits pays |
| C03 Mémoire pays | `c03_memoire_pays.py` | SQLite + pandas + ChromaDB (filtre `$lte`) | schéma SQL, règle d'historisation |
| C04 Dossier parti-élection | `c04_dossier_parti_election.py` | orchestration de C02/C03 | logique métier (assumé) |
| C05 Registre V-Party | `c05_registre_vparty.py` | pydantic + YAML ; contenu = codebook officiel | gate d'adhérence R-1 |
| C06 Retrieval interne | `c06_retrieval_interne.py` | rank-bm25 + cross-encoder multilingue | requête dictée par la fiche |
| C07 Test de suffisance | `c07_test_suffisance.py` | scikit-learn (LogReg calibrée) | traits métier, seuil calibré R-2 |
| C08 Recherche active | `c08_recherche_active.py` | ddgs/Tavily + trafilatura (via C01) | budget, fiabilité des domaines |
| C09 Diagnostic | `c09_moteur_diagnostic.py` | NLI multilingue (mDeBERTa XNLI) | structure du rapport |
| C10 Raisonnement adaptable | `c10_moteur_raisonnement.py` | litellm + endpoint local vLLM + transformers + pandas | routeur par régime de preuve |
| C11 Juges multiples | `c11_juges_multiples.py` | modèles Hugging Face open-weight via vLLM | orchestration du panel |
| C12 Agrégation | `c12_agregation.py` | numpy/scipy + krippendorff | décomposition de l'incertitude |
| C13 Sortie finale | `c13_sortie_finale.py` | pydantic + NLI (réutilisé) — cadre AIS | mise en forme |
| C14 Validation | `c14_validation.py` | scikit-learn, scipy ; étalon V-Party importé | ECE (formule standard), strates |
| C15 Garde-fous | `c15_garde_fous.py` | structlog, hashlib, litellm | 3 protocoles propres au projet |

Dépendances entre pièces (qui s'emboîte sur quoi) :

```text
config + schemas  (toutes les pièces s'y branchent)
C01 ──> C02, C03, C08
C02, C03 ──> C04 ──> C06
C05 ──> C06, C07, C10
C06 + C08 ──> C09 ──> C10 ──> C11 ──> C12 ──> C13 ──> C14
C15 traverse tout (temporalité, traces, contamination)
```

---

## Étape 0 — Atelier (installation)

```bash
cd compass_system
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-full.txt
python -m spacy download xx_sent_ud_sm              # optionnel : sentencizer amélioré
# Binaire OCR : installer tesseract-ocr + paquets langue fra/eng (apt, brew ou installeur Windows)
# Mode local Onyxia : export COMPASS_LLM_BACKEND=local ; export COMPASS_LLM_API_BASE=http://localhost:8000/v1
```

Critère de sortie : `python -c "import config, schemas"` sans erreur.

---

## Étape 1 — Réception de la colonne vertébrale

Test de réception (`pytest tests/test_schemas.py`, à écrire en premier) :

1. `CompassSettings()` se construit et `ensure_dirs()` crée les dossiers.
2. `VariableSheet` refuse une fiche sans échelle ; `FinalAnswer` sérialise en JSON.
3. `DocumentMeta.compute_hash()` est stable (même texte → même hash).

**Tant que cette étape n'est pas verte, ne rien monter d'autre** : tous les composants se branchent sur ces types.

---

## Étape 2 — Réception de C01 (pipeline documentaire)

Banc d'essai : 3 documents réels du dépôt (`Les Articles/` contient des PDF) + 1 URL de presse.

1. PDF texte → segments non vides, langue détectée, hash rempli.
2. PDF scanné (ou page image) → l'OCR se déclenche (vérifier le log « OCR appliqué »).
3. URL → trafilatura extrait l'article, pas le menu du site.
4. **Test négatif** : un appel sans date dans `DocumentMeta` doit échouer à la validation pydantic.

Critère de sortie : 100 % des segments portent `doc_date`, `language`, `sha256`.

---

## Étape 3 — Réception des mémoires (C02, C03)

1. C02 : indexer 50 segments du codebook V-Party (`vparty_codebook_v2.pdf` via C01) ; `query("pluralisme")` retourne des passages du codebook. **Test négatif** : `add()` d'un segment avec `party_id` → `ValueError`.
2. C03 structuré : importer les CSV officiels (REUSE_DIRECT) —
   - V-Party (`https://www.v-dem.net/data/v-party-dataset/`) → tables `vparty_scores`, `parties` ;
   - résultats électoraux (commission électorale ou ParlGov) → `elections`, `results` ;
   - Party Facts pour `pf_id`.
3. C03 documentaire : indexer de la presse datée ; **test décisif de l'historisation** : `query_documents(question, as_of=2015-01-01)` ne retourne AUCUN document de 2016. C'est le test le plus important de tout l'assemblage — s'il échoue, rien d'autre ne vaut.

---

## Étape 4 — Réception de C05 (registre) — AVANT C04/C06

Le registre se monte tôt parce qu'il pilote tout l'aval.

1. Écrire 3 fiches YAML dans `registry/` en transcrivant le codebook V-Party officiel, une par méthode :
   - `v2pavote.yaml` (`method: structured_query`),
   - `v2paplur.yaml` (`method: llm_guided`),
   - une variable de classification (`method: nlp_classifier`).
2. `VPartyRegistry().get("v2paplur")` est maintenant autorisé en production : la fiche a passé R-1 et la trace est dans `docs/adherence/v2paplur_R1_report.md`. Les autres fiches restent bloquées tant qu'elles n'ont pas leur propre rapport R-1.
3. `build_adherence_tests("v2paplur")` produit permutations, paraphrases, sondes d'inclusion/exclusion. Pour `v2paplur`, les sondes R-1 consolidées passent (7/7) et le YAML est figé à `adherence_passed: true` avec rapport versionné.

---

## Étape 5 — Premier sous-ensemble : l'instruction du cas (C04 + C06)

Montage : C04 se branche sur C02+C03 ; C06 se branche sur C04+C05.

1. `CaseFileBuilder.build(case)` pour un cas réel (ex. un parti ivoirien, présidentielle 2020) → le dossier contient documents du parti, contexte, faits structurés, **tous antérieurs à l'élection**.
2. `InternalRetriever.retrieve(dossier, fiche_v2paplur)` → ~10 passages classés ; vérifier à l'œil que les premiers parlent bien de pluralisme/opposition/alternance.

Critère de sortie : revue humaine de 2 dossiers — pertinence jugée acceptable avant d'aller plus loin (inutile de juger sur des dossiers vides).

---

## Étape 6 — Boucle d'enquête (C07 + C08), bornée

1. C07 en mode amorçage (non entraîné) : `decide()` → vérifier la mécanique des trois verdicts et la **borne** : au-delà de `search_max_iterations`, le verdict devient `ABSTAIN`, jamais une boucle infinie.
2. C08 : `investigate()` sur un manque artificiel → segments ingérés **via C01** (donc datés/hashés), budget `search_max_queries` respecté (test : le log « Budget de requêtes épuisé » apparaît quand on force 20 manques).
3. L'entraînement réel de C07 (`fit`) attend les étiquettes du pilote (étape 9) — le composant est monté, sa calibration viendra des essais.

---

## Étape 7 — Cellule de jugement (C09 + C10 + C11 + C12)

C'est le cœur ; monter dans cet ordre :

1. **C09 seul** : construire des `EvidenceItem` depuis l'étape 5 (`to_evidence`), injecter une paire artificiellement contradictoire (programme pro-pluralisme vs discours anti-opposition) → la contradiction doit apparaître dans `contradictions_detail`. Si le NLI ne la voit pas, ajuster le seuil avant de continuer.
2. **C10 par méthode** :
   - `structured` sur `v2pavote` → le score = la valeur SQL, confiance 1.0 ;
   - `deterministic` avec une formule jouet → vérifier l'échec propre (NaN + confiance 0) quand une dépendance manque ;
   - `llm_guided` sur `v2paplur` → JSON valide, rationale citant les preuves. **Vérifier ici que le rapport R-1 existe pour la variable servie** ; `v2paplur` est qualifiée, les autres variables doivent encore passer leur propre gate.
3. **C11** : panel sur le même diagnostic → ≥ 3 réponses de modèles différents. Sur 10 cas de contrôle, calculer `error_correlation` : si les corrélations hors diagonale dépassent ~0.9, le panel est redondant — diversifier (modèles, variantes) avant de continuer.
4. **C12** : agréger ; vérifier que `(3,3,3)` et `(1,3,5)` donnent le même score mais des désaccords très différents ; `panel_alpha` sur les 10 cas de contrôle.

---

## Étape 8 — Sortie et garde-fous (C13 + C15)

1. C13 : composer la réponse finale d'un cas → JSON complet (preuves, contre-preuves, déclaré/observé/inféré, sources) ; vérifier `attribution_checked` sur un cas où l'on a injecté une preuve hors sujet (il doit passer à `False`).
2. C15 :
   - `assert_temporal_integrity` avec une preuve postérieure injectée → exception `TemporalViolation` (test obligatoire) ;
   - `TraceLogger` : un fichier JSONL par cas, relisible ;
   - `contamination_probe` sur les 3 juges × 3 variables × 2 cas → conserver les résultats pour l'étape 9.

---

## Étape 9 — Essais au sol : le pilote

Assemblage complet sur périmètre restreint (cf. note Riboni §7) :

```text
Périmètre : 1–2 pays · ~10 variables (mix des 4 méthodes) · 3 juges
Étalon    : scores V-Party ≤ 2019 mis de côté AVANT tout réglage
```

1. **Corpus** : ingérer (C01→C03) les documents des cas pilotes.
2. **Run complet** : pour chaque cas × variable, dérouler C04→C13 sous traçage C15.
3. **Étiquetage suffisance** : un humain juge « preuves suffisantes ? » sur ~100 instances → `SufficiencyGate.fit()`, seuil choisi sur la courbe risque-couverture (R-2).
4. **Validation C14** :
   - rapport global + **stratifié par langue** (R-6) ;
   - couverture des intervalles V-Party (AM-11), pas seulement MAE ;
   - exclusion/pondération des cas marqués contaminés par la sonde C15 (A-2) ;
   - **ablations** : refaire tourner en éteignant un étage à la fois (sans C08 ; un seul juge ; sans C09) — chaque étage doit prouver son gain, sinon il saute (principe Claude.md §5.3).
5. **Recalibration** : corriger registre / retrieval / seuils d'après l'analyse d'erreurs ; re-valider. Une boucle complète minimum avant toute conclusion.

Critère de « bon de vol » : les 4 critères du bloc 14 documentés dans un rapport, ablations comprises — c'est le livrable annoncé à M. Riboni.

---

## Étape 10 — Vol : extension contrôlée

Seulement après le bon de vol : élargir variable par variable (chaque nouvelle fiche passe la gate R-1), pays par pays (chaque nouveau pays passe le test d'historisation de l'étape 3), puis cas post-2019 (là où le système crée de la valeur nouvelle — codage humain d'un sous-échantillon comme étalon frais).

---

## Pièges connus (à relire avant chaque étape)

1. **Ne jamais court-circuiter la gate R-1** « pour tester vite » : c'est précisément le raccourci que Halterman & Keith (2025) condamnent.
2. **Le désaccord des juges est une borne inférieure** de l'incertitude (dossier partagé) — ne pas le vendre comme l'incertitude totale ; c'est `combined_confidence` (C12) qui combine preuve et jugement.
3. **Une page web sans date n'est pas une preuve d'antériorité** — C08 la marque UNKNOWN ; ne pas « réparer » ce comportement.
4. **Les scores V-Party importés (C03) ne servent JAMAIS d'entrée au raisonnement** d'un cas évalué — uniquement d'étalon (C14) et de dépendances des formules déterministes sur variables élémentaires déjà validées. Mélanger les deux invalide toute la validation.
5. **Chiffrer le coût** au fil du pilote (tokens × juges × cas) — le passage à l'échelle se décide sur ce chiffre, pas sur l'enthousiasme.
