# Ingestion du Manifesto Project

COMPASS suit le workflow officiel du Manifesto Project et privilégie une réponse exploitable plutôt qu'un échec complet lorsque le PDF original est inaccessible.

## Déroulement

```text
dataset core
→ clé parti_date
→ endpoint metadata
→ manifesto_id et URL originale éventuelle
→ téléchargement PDF
→ sinon texts_and_annotations
→ DocumentPipeline
→ CountryMemory
→ PoliticalGraph par pays
→ rapport JSON
```

## Authentification

Créer une clé sur le Manifesto Project puis l'exposer :

```bash
export MANIFESTO_API_KEY="votre_cle"
```

La clé n'est jamais écrite dans le dépôt.

## Tester une entrée

Commencer par une résolution sans téléchargement :

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --keys 41320_200909 \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --party-id 41320 \
  --election-id DEU_2009 \
  --doc-date 2009-09-01 \
  --dry-run \
  --print-metadata
```

Puis lancer l'ingestion réelle :

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --keys 41320_200909 \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --party-id 41320 \
  --election-id DEU_2009 \
  --doc-date 2009-09-01
```

## Ingestion par CSV

Colonnes minimales :

```text
key,country_iso3,doc_date
```

Colonnes recommandées :

```text
key,metadata_version,country_iso3,party_id,election_id,doc_date,doc_type,language,reliability,pdf_url
```

Commande :

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --manifest examples/manifesto_manifest_deu_2009.csv \
  --translation en
```

Le CSV peut déjà contenir plusieurs pays. Le script crée une `CountryMemory` par valeur `country_iso3` et place toutes les collections dans le même dossier Chroma.

Il crée aussi un graphe isolé pour chaque pays. Après chaque document, les chunks parents sont analysés, puis le graphe est sauvegardé. L'option `--no-graph` permet exceptionnellement de désactiver cette étape.

L'extraction d'entités nécessite un modèle spaCy :

```bash
python -m spacy download xx_ent_wiki_sm
```

## Générer un CSV depuis le core

```bash
python scripts/build_manifesto_manifest.py \
  --core-version MPDS2024a \
  --core-kind dta \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --country-code 41 \
  --election-date 200909 \
  --language de \
  --output data/manifests/deu_2009.csv \
  --inspect
```

Le builder actuel produit un pays à la fois. La production automatique d'un manifeste mondial est différée dans `docs/06_roadmap.md`.

## PDF bloqué

Un lien `/down/originals/...pdf` peut retourner `401` ou `403` malgré une clé API valide. Sauf option `--no-text-fallback`, COMPASS appelle alors `texts_and_annotations`, écrit le texte sous `data/manifesto_texts/` et l'indexe par le même pipeline de chunking.

Exemple de résultat normal :

```text
41320_200909: PDF blocked; trying texts_and_annotations fallback
41320_200909: indexed API text fallback (2624 segments)
```

## Sorties

- PDF : `data/manifesto_pdfs/<PAYS>/` ;
- textes API : `data/manifesto_texts/<PAYS>/` ;
- Chroma : `data/manifesto_ingestion/chroma/` ;
- SQLite : `data/manifesto_ingestion/compass_structured.db` ;
- graphes : `data/manifesto_ingestion/political_graph_<pays>.graphml` ;
- rapport : `outputs/manifesto_pdf_ingestion_report.json`.

Le rapport contient `graph_new_edges` et `graph_total_edges` pour chaque document traité.

## Limites actuelles

- le builder mondial n'est pas encore disponible ;
- les `doc_id` sont encore générés par UUID, donc la stratégie de réingestion mondiale doit être corrigée avant un chargement complet ;
- `election_id` existe dans `DocumentMeta` mais n'est pas encore persisté dans les métadonnées Chroma ;
- la disponibilité d'un PDF dépend des droits et métadonnées du Manifesto Project.

Ces limites sont inscrites dans la roadmap pour éviter de présenter comme opérationnel un chantier encore différé.
