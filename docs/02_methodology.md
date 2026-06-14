# Methodology

The methodology separates evidence from interpretation.

Raw documents are first ingested with metadata. Passages are then matched to a political taxonomy. Reasoning operates on the retrieved evidence, not on an unbounded prompt. Validation checks whether the final profile includes a source document, detected themes, and evidence for each theme.

The public demo uses deterministic keyword retrieval so it can run anywhere. The research version can replace this layer with embeddings, reranking, NLI, or LLM-assisted coding while preserving the same interface.
