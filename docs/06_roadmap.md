# Roadmap

## Implemented in the public pipeline

- Parent-child + semantic chunking is implemented and tested: retrieval targets bounded child citation units, while paragraph-level parent blocks are retained for contextual reranking.
- Hybrid retrieval is implemented and tested as the default path: dense Chroma order is fused with BM25, parent context is injected, then a cross-encoder reranks the candidate pool before final citation selection.
- COMPASS Chat exposes an inspectable prompt page after each LLM answer, so demos can show exactly what was sent to the local vLLM server.
- Optional HyDE retrieval is implemented with graceful fallback when the local/API LLM is unavailable.
- Manifesto Project ingestion is connected to the official API workflow, with text fallback when original PDFs are blocked.
- Onyxia local vLLM and COMPASS Chat are documented and validated on the small open-weight profile.

## Next research validation steps

- Reindex the pilot Manifesto corpus with semantic chunking and validate source readability in the chat.
- Validate parent/child block sizes and reranking pool size on pilot Manifesto documents; tune `COMPASS_PARENT_CHUNK_SIZE`, `COMPASS_CHILD_CHUNK_MIN_CHARS`, `COMPASS_CHILD_CHUNK_MAX_CHARS`, `COMPASS_SEMANTIC_CHUNK_SIMILARITY_THRESHOLD`, and `COMPASS_RERANK_POOL_SIZE`.
- Validate the political knowledge graph on annotated actor-relation examples.
- Add multilingual preprocessing and language-aware segmentation.
- Expand the taxonomy from the compact demo to the full research coding scheme.
- Add uncertainty, contradiction, and source-reliability reporting to final pilot reports.
- Provide more synthetic examples for cross-country and election-year scenarios.
- Package the framework for reproducible research workflows.
