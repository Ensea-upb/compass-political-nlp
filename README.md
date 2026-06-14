# COMPASS Political NLP

Research-oriented NLP framework for extracting, structuring, and validating political positions from party manifestos and heterogeneous political documents.

COMPASS combines computational social science, multilingual NLP, document engineering, and political economy. The public repository is a clean showcase version: reusable code, concise documentation, synthetic examples, tests, and no private PDFs, articles, slides, supervision notes, or internal work files.

## Why COMPASS?

Political documents are often multilingual, heterogeneous, and unevenly digitized. Many approaches work best on clean manifesto text, which can bias analysis against parties or countries with weaker documentation infrastructures.

COMPASS is designed as an auditable pipeline that moves from raw political evidence to traceable party profiles and, eventually, comparative political-space analysis.

## Public Demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python examples/run_demo.py
```

Expected output:

```text
Document loaded: sample_manifesto.txt
Detected political themes: economy, sovereignty, democracy
Generated party profile: examples/sample_party_profile.json
Validation status: passed
```

## Research Pipeline

```text
Raw political documents
-> ingestion
-> preprocessing
-> taxonomy annotation
-> retrieval / reasoning
-> validation
-> final political analysis
```

## Repository Structure

```text
compass-political-nlp/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── docs/
├── src/
│   └── compass/
├── examples/
├── notebooks/
├── tests/
└── assets/
```

## Current Scope

The public version currently includes:

- a deterministic synthetic demo pipeline;
- compact data schemas;
- theme retrieval over a sample manifesto;
- party-profile generation;
- validation checks;
- documentation for methodology, taxonomy, architecture, and roadmap.

The internal research version can replace the demo retrieval layer with OCR, PDF parsing, embeddings, reranking, NLI, LLM-assisted coding, and human validation while preserving the same modular story.

## Data Policy

This repository does not include copyrighted manifestos, restricted datasets, confidential research notes, private supervision material, downloaded articles, or internal Claude work files.

Public examples are synthetic.

## Disclaimer

COMPASS is an academic research project. Political scores or profiles produced by the pipeline require human review, robustness checks, and external validation before substantive interpretation.

## Author

Inza Ouada Soro  
ENSAE Paris  
Research internship project in NLP, political economy, and computational social science.
