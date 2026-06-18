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
src/compass/political_graph.py
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

Recent integration work adds three research-oriented improvements:

- parent-child + semantic chunking in `document_pipeline.py` and `country_memory.py`: parent blocks preserve paragraph-level context, semantic topic shifts can start new parents, while child segments are citation units built from sentence/list fragments; very short fragments are merged and oversized fragments are split before indexing;
- hybrid retrieval in `country_memory.py`, `chat/engine.py`, and `internal_retrieval.py`: Chroma's dense ordering is fused with BM25 lexical ranking, then parent context is injected for each child segment;
- optional HyDE in `internal_retrieval.py`, using a variable-grounded hypothetical passage to improve semantic search;
- `political_graph.py`, a C02b knowledge-graph component that summarizes inferred relations between political actors under temporal constraints.
