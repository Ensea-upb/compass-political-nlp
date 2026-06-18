# COMPASS Chat Interface

COMPASS Chat is an optional conversational layer over the existing COMPASS memory stack. It does not replace ingestion, retrieval, reasoning, validation, or schemas. It calls `CountryMemory.query_documents()`, builds citations, then asks the configured local/API LLM to answer from retrieved evidence.

## Architecture

```text
User question
-> ChatEngine
-> CountryMemory.query_documents_hybrid()
-> dense retrieval + BM25 fusion
-> cited evidence segments
-> parent/general context retrieval
-> local vLLM through compass.llm_client
-> answer with [S1], [S2] citations
-> inspectable human-readable prompt link
```

If vLLM is not running, the engine returns an extractive answer from the retrieved passages instead of failing.

The chat now separates two kinds of context:

- cited evidence: short child segments used as proof and exposed as `[S1]`, `[S2]`;
- general context: parent-level manifesto blocks used only to frame the answer.

This matters for demos and audits: the model can understand the broader manifesto section, but every substantive claim must still be supported by a cited evidence segment.

The child segments are not raw one-word or one-line fragments. During ingestion, COMPASS merges very short fragments with neighboring text, splits oversized fragments, and can start new parent blocks when semantic cohesion drops. This is why source excerpts should be more readable after reindexing: instead of citations such as `Setting impulses.`, the chat should retrieve fuller citation units.

After each LLM answer, the web interface displays a `Voir le prompt LLM` link. It opens a local inspection page containing a human-readable rendering of the exact message list sent to the OpenAI-compatible vLLM endpoint, plus a collapsible raw JSON view for auditability. Source details are no longer appended at the end of every chat answer; the answer itself should cite evidence inline with `[S1]`, `[S2]`, etc.

For small local vLLM models, the chat also applies a prompt budget:

- at most 6 cited evidence passages are sent to the LLM;
- at most 2 general context blocks are sent;
- parent context, evidence text, and conversation history are truncated;
- chat answers request at most 650 output tokens.

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

The prompt receives two explicitly separated blocks:

- `GENERAL_CONTEXT`: broader parent chunks from the same country, date filter, and party scope. This is background only. The model is explicitly forbidden to cite `[C1]`, `[C2]`, or to use this block as proof.
- `CITED_EVIDENCE`: the only passages allowed to support political claims. Every substantive claim must cite `[S1]`, `[S2]`, etc.

The prompt is calibrated to reduce hallucination risk: it forbids outside knowledge, asks the model to reject unsupported user premises cautiously, and requires an explicit insufficiency statement when the cited evidence does not answer the question.

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
