# COMPASS Chat Interface

COMPASS Chat is an optional conversational layer over the existing COMPASS memory stack. It does not replace ingestion, retrieval, reasoning, validation, or schemas. It calls `CountryMemory.query_documents()`, builds citations, then asks the configured local/API LLM to answer from retrieved evidence.

## Architecture

```text
User question
-> ChatEngine
-> CountryMemory.query_documents()
-> cited evidence segments
-> local vLLM through compass.llm_client
-> answer with [S1], [S2] citations
```

If vLLM is not running, the engine returns an extractive answer from the retrieved passages instead of failing.

## Install UI dependency

After the full or Onyxia requirements:

```bash
pip install -r requirements-chat.txt
```

## Launch on Onyxia

Recommended stable interface, without Gradio:

```bash
python apps/chat_web.py \
  --country DEU \
  --as-of 2009-09-27 \
  --party 41320 \
  --port 41771
```

The older Gradio prototype remains available as `apps/chat_gradio.py`, but `chat_web.py` is preferred on Onyxia when Gradio returns frontend JSON parsing errors.

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

Expose port `7860` in the Onyxia service configuration if you want to open the UI from the browser.

## Example questions

```text
What does this party say about democracy?
What economic themes appear in the 2009 manifesto?
Which passages mention immigration or national identity?
Give me evidence for the party's position on European integration.
```

## Programmatic use

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