# Architecture

```text
examples/sample_manifesto.txt
        |
        v
src/compass/demo.py
        |
        v
examples/sample_party_profile.json
```

The public quickstart uses the deterministic demo module.

The real research architecture is exposed through the renamed modules:

```text
src/compass/document_pipeline.py
src/compass/general_memory.py
src/compass/country_memory.py
src/compass/vparty_registry.py
src/compass/internal_retrieval.py
src/compass/diagnostic_engine.py
src/compass/reasoning_engine.py
src/compass/judge_panel.py
src/compass/aggregation.py
src/compass/final_output.py
src/compass/validation.py
src/compass/guardrails.py
src/compass/orchestrator.py
```

This separation keeps the public demo lightweight while preserving the real research pipeline.
