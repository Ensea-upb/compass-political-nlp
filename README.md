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
