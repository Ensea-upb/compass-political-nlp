# Architecture

```text
examples/sample_manifesto.txt
        |
        v
src/compass/ingestion/document_pipeline.py
        |
        v
src/compass/retrieval/internal_retrieval.py
        |
        v
src/compass/reasoning/reasoning_engine.py
        |
        v
src/compass/validation/validation.py
        |
        v
examples/sample_party_profile.json
```

The package keeps each responsibility isolated so heavier research components can be introduced without changing the end-to-end contract.
