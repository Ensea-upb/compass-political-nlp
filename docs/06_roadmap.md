# Roadmap

## Implemented in the public pipeline

- Parent-child chunking is implemented and tested: retrieval targets child passages, while parent blocks are retained for contextual reranking.
- Hybrid retrieval is implemented and tested: dense Chroma order is fused with BM25 before cross-encoder reranking.
- Optional HyDE retrieval is implemented with graceful fallback when the local/API LLM is unavailable.
- Manifesto Project ingestion is connected to the official API workflow, with text fallback when original PDFs are blocked.
- Onyxia local vLLM and COMPASS Chat are documented and validated on the small open-weight profile.

## Next research validation steps

- Validate parent block size on pilot Manifesto documents and tune `COMPASS_PARENT_CHUNK_SIZE`.
- Evaluate HyDE retrieval against the hybrid dense+BM25 baseline through ablations.
- Validate the political knowledge graph on annotated actor-relation examples.
- Add multilingual preprocessing and language-aware segmentation.
- Expand the taxonomy from the compact demo to the full research coding scheme.
- Add uncertainty, contradiction, and source-reliability reporting to final pilot reports.
- Provide more synthetic examples for cross-country and election-year scenarios.
- Package the framework for reproducible research workflows.
