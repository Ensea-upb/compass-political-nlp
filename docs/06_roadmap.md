# Roadmap

## Implemented in the public pipeline

- Parent-child + semantic chunking is implemented and tested: retrieval targets bounded child citation units, while paragraph-level parent blocks are retained for contextual reranking.
- Hybrid retrieval is implemented and tested as the default path: dense Chroma order is fused with BM25, parent context is injected, then a cross-encoder reranks the candidate pool before final citation selection.
- COMPASS Chat exposes an inspectable prompt page after each LLM answer, so demos can show exactly what was sent to the local vLLM server.
- Optional HyDE retrieval is implemented with graceful fallback when the local/API LLM is unavailable.
- Manifesto Project ingestion is connected to the official API workflow, with text fallback when original PDFs are blocked.
- Onyxia local vLLM and COMPASS Chat are documented and validated on the small open-weight profile.

## Deferred workstream: global Manifesto corpus and contamination-safe chat

This workstream is recorded for later implementation. No global-corpus code is implemented yet.

- Build one ingestion manifest covering all available Manifesto Project countries, parties, and elections instead of generating one country-specific CSV at a time.
- Keep the existing `compass_country_<iso3>` Chroma collections and add a lightweight corpus-level facade that discovers and queries them; avoid a costly migration to a new monolithic collection.
- Make Manifesto document identifiers deterministic so rerunning ingestion updates existing segments instead of creating UUID-based duplicates.
- Persist `election_id` in Chroma metadata and preserve a canonical manifesto identity across original text and translated variants.
- Allow COMPASS Chat to launch without mandatory `--country`, `--party`, or `--as-of` arguments; no country filter should mean all indexed country collections are available.
- Add a scope resolver after question routing and before retrieval. It must resolve explicit country, party, election, date, language, and comparison scopes.
- Apply scope priority in this order: entities explicitly stated in the current question, filters explicitly selected in the UI, then global corpus scope. Conversation history must not silently change documentary filters.
- Fail closed on ambiguous party or election references by asking for clarification instead of selecting a country or party arbitrarily.
- Prevent corpus contamination: cited child evidence, parent context, and general context must remain attached to the same collection, document, party, election, and language variant.
- Treat global exploration and explicit comparison separately. Comparative retrieval must create one evidence compartment per requested country/party and require evidence for every compared side.
- Make `AnswerValidator` scope-aware so each political claim is supported by citations from the correct scope; reject cross-country, cross-party, cross-election, and original/translation leakage.
- Add integration tests for global questions, single-country questions, ambiguous scopes, cross-country comparisons, direct segment lookup, and contamination rejection.
- Update the Onyxia runbook only after a complete global ingestion and chat validation run has passed.

## Next research validation steps

- Reindex the pilot Manifesto corpus with semantic chunking and validate source readability in the chat.
- Validate parent/child block sizes and reranking pool size on pilot Manifesto documents; tune `COMPASS_PARENT_CHUNK_SIZE`, `COMPASS_CHILD_CHUNK_MIN_CHARS`, `COMPASS_CHILD_CHUNK_MAX_CHARS`, `COMPASS_SEMANTIC_CHUNK_SIMILARITY_THRESHOLD`, and `COMPASS_RERANK_POOL_SIZE`.
- Validate the political knowledge graph on annotated actor-relation examples.
- Add multilingual preprocessing and language-aware segmentation.
- Expand the taxonomy from the compact demo to the full research coding scheme.
- Add uncertainty, contradiction, and source-reliability reporting to final pilot reports.
- Provide more synthetic examples for cross-country and election-year scenarios.
- Package the framework for reproducible research workflows.
