# COMPASS Chat Interface

COMPASS Chat is an optional conversational layer over the existing COMPASS memory stack. It does not replace ingestion, retrieval, reasoning, validation, or schemas. It calls `CountryMemory.query_documents()`, builds citations, then asks the configured local/API LLM to answer from retrieved evidence.

## Architecture

```text
User question
-> ChatEngine
-> CountryMemory.query_documents_hybrid()
-> dense retrieval + BM25 fusion
-> question-aware evidence boost
-> cross-encoder reranking over parent context + child segment
-> cited evidence segments
-> analytical political-science context
-> parent/general context retrieval
-> local vLLM through compass.llm_client
-> answer with [S1], [S2] citations
-> inspectable human-readable prompt link
```

If vLLM is not running, the engine returns an extractive answer from the retrieved passages instead of failing.

Before retrieval, the chat routes each request:

- `direct_lookup`: an explicit segment id is fetched directly;
- `corpus_scope`: questions such as `tu es connecte a quel corpus ?` receive a deterministic session-scope answer without retrieval or LLM generation;
- `evidence_query`: political questions run through hybrid retrieval, generation, and `AnswerValidator`.

This prevents metadata questions from retrieving unrelated manifesto fragments or being rejected for missing `[Sx]` citations.

The chat now separates three kinds of prompt material:

- cited evidence: short child segments used as proof and exposed as `[S1]`, `[S2]`;
- analytical context: a compact political-science reading frame derived from the question, used to identify relevant dimensions such as institutions, instruments, beneficiaries, rights, integration, or policy mechanisms;
- general context: parent-level manifesto blocks used only to frame the answer.

This matters for demos and audits: the model can understand both the political concept and the broader manifesto section, but every substantive claim must still be supported by a cited evidence segment.

The child segments are not raw one-word or one-line fragments. During ingestion, COMPASS merges very short fragments with neighboring text, splits oversized fragments, and can start new parent blocks when semantic cohesion drops. This is why source excerpts should be more readable after reindexing: instead of citations such as `Setting impulses.`, the chat should retrieve fuller citation units.

The retrieval layer first builds a broad candidate pool with dense Chroma retrieval, BM25 lexical matching, and light question-aware evidence scoring. It then reranks the pool with the configured cross-encoder (`COMPASS_RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`). The cross-encoder receives the user question paired with `parent context + child segment`: the parent block gives enough local manifesto context to judge relevance, while the child segment remains the citable proof shown as `[S1]`, `[S2]`, etc.

The prompt page exposes `retrieval_reason` for each cited passage, including `dense_rank`, `bm25_rank`, profile notes, and `cross_encoder_score`, so the selection can be audited.

After each LLM answer, the web interface displays a `Voir le prompt LLM` link. It opens or reuses a single prompt-inspection tab, so clicking the link for another answer updates the same page instead of creating many tabs. The page contains a human-readable rendering of the exact message list sent to the OpenAI-compatible vLLM endpoint, plus a collapsible raw JSON view for auditability. Source details are no longer appended at the end of every chat answer; the answer itself should cite evidence inline with `[S1]`, `[S2]`, etc.

For small local vLLM models, the chat also applies a prompt budget:

- at most 4 cited evidence passages are sent to the LLM;
- at most 1 general context block is sent;
- parent context, evidence text, and conversation history are truncated;
- chat answers request at most 350 output tokens.

This avoids common vLLM `400 Bad Request` failures caused by prompts that exceed the model context window, especially with `Qwen/Qwen2.5-3B-Instruct` served at `max_model_len=4096`.

## Install UI dependency

After the full or Onyxia requirements:

```bash
pip install -r requirements-chat.txt
```

## Launch on Onyxia

Recommended stable interface, without Gradio:

```bash
export COMPASS_DATA_DIR=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion
export COMPASS_CHROMA_DIR=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion/chroma
export COMPASS_SQLITE_PATH=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion/compass_structured.db

export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu
export COMPASS_RERANK_ENABLED=true
export COMPASS_RERANK_POOL_SIZE=24

python apps/chat_web.py \
  --country DEU \
  --as-of 2009-09-27 \
  --party 41320 \
  --port 41771
```

Expose port `41771` in the Onyxia service configuration. If another service already uses it, choose any free port and expose the same value.

The older Gradio prototype remains available as `apps/chat_gradio.py`, but `chat_web.py` is preferred on Onyxia when Gradio returns frontend JSON parsing errors such as:

```text
Unexpected token 'U', "Unsupported"... is not valid JSON
Could not parse server response
```

The dependency-light web app also avoids the version tension between recent Gradio packages and the FastAPI/Starlette pins used by the validated vLLM runtime.

## Launch with Gradio

After ingesting Manifesto documents:

```bash
export COMPASS_DATA_DIR=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion
export COMPASS_CHROMA_DIR=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion/chroma
export COMPASS_SQLITE_PATH=/home/onyxia/work/compass-political-nlp/data/manifesto_ingestion/compass_structured.db

python apps/chat_gradio.py \
  --country DEU \
  --as-of 2009-09-27 \
  --party 41320 \
  --port 7860
```

Expose port `7860` in the Onyxia service configuration if you want to open the UI from the browser. Prefer `chat_web.py` for the operational Onyxia workflow.

## Expected Behavior

When the local vLLM server is running, COMPASS Chat answers with a short synthesis and cites retrieved passages inline as `[S1]`, `[S2]`, etc. The prompt asks the model to attach an inline citation to every substantive claim and to avoid interpretations that are not directly supported by the retrieved passages. The chat interface does not append a separate source bibliography after each answer; use `Voir le prompt LLM` to inspect the evidence behind each `[Sx]`.

The prompt receives three explicitly separated blocks:

- `ANALYTICAL_CONTEXT`: conceptual and theoretical reading frame. It is not factual evidence, cannot be cited, and cannot introduce facts about the party.
- `GENERAL_CONTEXT`: broader parent chunks from the same country, date filter, and party scope. This is background only. The model is explicitly forbidden to cite `[C1]`, `[C2]`, or to use this block as proof.
- `CITED_EVIDENCE`: the only passages allowed to support political claims. Every substantive claim must cite `[S1]`, `[S2]`, etc.

The prompt is calibrated to reduce hallucination risk: it forbids outside knowledge, asks the model to reject unsupported user premises cautiously, and requires an explicit insufficiency statement when the cited evidence does not answer the question.

The chat also validates the LLM output before displaying it. If the model cites `ANALYTICAL_CONTEXT` or `GENERAL_CONTEXT`, invents a source id that was not shown in `CITED_EVIDENCE`, or gives a substantive answer without any `[Sx]` citation, COMPASS rejects the generated answer and falls back to an extractive evidence-only response. This keeps the demo usable even with small local models that sometimes ignore part of the prompt.

The prompt inspection page is designed for demos. Its `CITED_EVIDENCE` block includes metadata, segment id, and excerpt-style evidence:

```text
[S1] DEU | party=41320 | date=2009-09-01 | manifesto_api_text
segment: `...:p545c001`
excerpt: "..."
```

When vLLM is stopped, misconfigured, or returns an error, the chat should not crash. It returns an extractive fallback built from the most relevant retrieved passages and includes a technical note such as `fallback declenche`.

You can request an exact passage by segment id:

```text
je veux ce passage: 1312ffc6-e62d-4b91-a043-d384a8697f39:p018c001
```

Exact lookup uses `CountryMemory.fetch_records_by_ids()`, so it should display the same metadata as normal retrieval instead of `UNK party? date? document`.

If you ask a follow-up such as:

```text
What are the exact sources for your answer?
```

`chat_web.py` points you to the `Voir le prompt LLM` page for the previous answer. That page contains the exact `CITED_EVIDENCE` block used to ground the inline `[Sx]` citations.

If the answer cites evidence but then cannot print the cited passages, check that `COMPASS_CHROMA_DIR` and `COMPASS_SQLITE_PATH` point to the same ingestion run.

## Example Questions

```text
What does this party say about democracy?
What economic themes appear in the 2009 manifesto?
Which passages mention immigration or national identity?
Give me evidence for the party's position on European integration.
What are the exact sources for your answer?
```

## Programmatic Use

```python
from datetime import date
from compass.chat import ChatEngine, ChatRequest
from compass.country_memory import CountryMemory

engine = ChatEngine(CountryMemory("DEU"))
response = engine.ask(ChatRequest(
    question="What does the party say about democracy?",
    as_of=date(2009, 9, 27),
    party_id="41320",
))
print(response.answer)
```
