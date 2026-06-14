````md
# COMPASS — Political NLP for Manifesto Analysis

**COMPASS** is a research-oriented NLP framework for extracting, structuring and analyzing political positions from party manifestos and heterogeneous political documents.

The project combines **computational social science**, **political text-as-data**, **multilingual NLP**, **document engineering** and **political economy** to study party positioning, coalition formation, foreign policy orientations and political spaces, with a particular focus on African and non-Western contexts.

> Current status: active research prototype — under development.

---

## Why COMPASS?

Most existing approaches to party positioning rely on clean textual manifestos, manual coding schemes or datasets mostly designed around Western democratic contexts.

COMPASS starts from a broader problem:

Political documents are often heterogeneous, multilingual and multimodal. They may include clean PDFs, scanned documents, tables, images, low-quality OCR outputs and politically relevant visual material. Ignoring these formats creates a selection bias against countries and parties with weaker digital documentation infrastructures.

COMPASS aims to build a **country-agnostic** and **auditable** pipeline that can move from raw political documents to structured political measures.

---

## Core Objective

The long-term goal is to construct a quantitative and interpretable representation of political spaces by extracting party positions from manifestos and related political texts.

The framework is designed to support research on:

- party ideology and policy positioning;
- coalition formation and alliance strategies;
- foreign policy orientation;
- political polarization and fragmentation;
- external actors and geopolitical alignment;
- political determinants of international trade.

---

## Research Pipeline

```mermaid
flowchart TD
    A[Heterogeneous political documents] --> B[Document ingestion]
    B --> C[OCR, PDF parsing, multimodal extraction]
    C --> D[Text cleaning and segmentation]
    D --> E[Taxonomy-based annotation]
    E --> F[Political position estimation]
    F --> G[Party × election × country database]
    G --> H[Coalition, polarization and geopolitical analysis]
    H --> I[Extensions to international trade]
````

---

## Methodological Strategy

COMPASS follows a progressive and empirically justified NLP strategy.

The project does **not** start directly with complex models. It moves from simple, interpretable and auditable methods toward modern NLP only when the previous method’s limitations are clearly identified.

```text
Dictionary / lexical baselines
        ↓
TF-IDF and statistical text features
        ↓
Wordscores / Wordfish-style document scaling
        ↓
Multilingual sentence embeddings
        ↓
Transformer-based classification
        ↓
LLM-assisted annotation and validation
```

This progression is designed to preserve interpretability, robustness and scientific validity.

---

## Main Contributions

### 1. Robust document ingestion

COMPASS is designed to process heterogeneous political documents:

* clean textual PDFs;
* scanned PDFs;
* OCR outputs;
* tables;
* images;
* documents with mixed layout quality.

The objective is to reduce the bias introduced by methods that only work on clean text.

### 2. Political text annotation

The framework maps political statements to a structured analytical taxonomy inspired by existing manifesto coding traditions, while extending them to better capture non-Western political contexts.

### 3. Position estimation

COMPASS supports multiple strategies for estimating party positions:

* baseline lexical scores;
* supervised or semi-supervised classification;
* Wordscores-style supervised scaling;
* Wordfish-style unsupervised scaling;
* multilingual embeddings;
* Transformer and LLM-based classification.

### 4. Cross-country political analysis

The final output is intended to support comparative analysis of political spaces across parties, elections, countries and time.

---

## Research Workstreams

COMPASS is organized around five scientific workstreams.

### WS01 — Coalition Formation

How do ideological distance, office-seeking incentives, vote-seeking strategies and institutional constraints explain political alliances and coalition formation?

### WS02 — Foreign Policy Orientation

How can party texts reveal positions on sovereignty, regional cooperation, former colonial powers, trade openness, foreign investment, security and diplomacy?

### WS03 — Political Space and Polarization

How can we estimate ideological distance, centrality, fragmentation and polarization across parties, elections and countries?

### WS04 — External Actors and African Political Spaces

How do foreign states, international organizations, investors and firms interact with domestic political competition?

### WS05 — Political Space and International Trade

How do political orientation, polarization and geopolitical shifts relate to import, export and bilateral trade flows?

---

## Repository Structure

```text
compass-political-nlp/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── docs/
│   ├── project_overview.md
│   ├── methodology.md
│   ├── taxonomy_compass.md
│   ├── literature_review.md
│   └── roadmap.md
│
├── src/
│   └── compass/
│       ├── ingestion/
│       ├── preprocessing/
│       ├── taxonomy/
│       ├── annotation/
│       ├── scaling/
│       ├── validation/
│       └── analysis/
│
├── notebooks/
│   ├── 01_document_ingestion_demo.ipynb
│   ├── 02_taxonomy_annotation_demo.ipynb
│   └── 03_position_scaling_demo.ipynb
│
├── examples/
│   ├── sample_manifesto.txt
│   ├── sample_annotations.csv
│   └── demo_pipeline.py
│
├── tests/
│   └── test_basic_pipeline.py
│
└── assets/
    ├── architecture.png
    └── demo_screenshot.png
```

---

## Planned Features

* [ ] Document ingestion pipeline for political manifestos
* [ ] OCR and PDF parsing utilities
* [ ] Text cleaning and segmentation module
* [ ] COMPASS taxonomy annotation module
* [ ] Baseline lexical and TF-IDF models
* [ ] Wordscores-style supervised scaling
* [ ] Wordfish-style unsupervised scaling
* [ ] Multilingual embedding-based retrieval
* [ ] Transformer-based political text classification
* [ ] LLM-assisted annotation workflow
* [ ] Human validation and audit protocol
* [ ] Party × election × country position database
* [ ] Coalition and polarization analysis modules

---

## Example Use Case

```python
from compass.ingestion import load_document
from compass.preprocessing import clean_text
from compass.annotation import annotate_statements
from compass.scaling import estimate_positions

document = load_document("examples/sample_manifesto.txt")
cleaned_text = clean_text(document)

annotations = annotate_statements(
    cleaned_text,
    taxonomy="compass"
)

positions = estimate_positions(
    annotations,
    method="baseline"
)

print(positions)
```

Expected output:

```text
party_id     election_year     economic_position     foreign_policy_position     polarization_score
P001         2025              -0.32                 0.61                        0.44
```

---

## Validation Philosophy

COMPASS treats every NLP output as a measurement that must be validated.

The project follows four validation principles:

1. **Face validity** — Do the outputs make substantive political sense?
2. **Human validation** — Do expert or trained human coders agree with the model?
3. **Convergent validity** — Do the measures correlate with external political datasets?
4. **Error analysis** — Where and why does the model fail?

No model output should be interpreted as a scientific result without validation.

---

## Tech Stack

The planned technical stack includes:

* Python
* pandas
* numpy
* scikit-learn
* spaCy / NLTK
* Hugging Face Transformers
* sentence-transformers
* OCR and PDF parsing tools
* pytest
* Jupyter
* Streamlit for demos and inspection

---

## Research Foundations

COMPASS builds on several strands of literature:

* Comparative Manifesto Project
* Wordscores
* Wordfish
* Text-as-data methods in political science
* Multilingual sentence embeddings
* BERT and Transformer-based classification
* BERT-NLI and zero-shot text classification
* LLM-based political text annotation
* Coalition formation theory
* Political economy of international trade

---

## Current Development Stage

The project is currently focused on:

* consolidating the conceptual architecture;
* designing the COMPASS taxonomy;
* reviewing the political text-as-data literature;
* defining the document-to-position pipeline;
* preparing the first reproducible NLP modules.

This repository is therefore a research prototype, not a finished software package.

---

## Data Policy

This repository does not include copyrighted manifestos, restricted datasets, confidential research notes or private supervision material.

Public examples are synthetic or based on openly available sources.

---

## Disclaimer

COMPASS is an academic research project. The methods implemented here are intended for scientific analysis and require careful validation before any substantive interpretation.

Political scores produced by the pipeline should not be treated as definitive measures without human review, robustness checks and external validation.

---

## Author

**Inza Ouada Soro**
ENSAE Paris
Research internship project in NLP, political economy and computational social science.

```
```
