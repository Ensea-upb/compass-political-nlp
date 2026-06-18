"""Dependency-light web chat for COMPASS memories.

This app avoids Gradio compatibility issues on managed notebook services. It
uses Python's standard library HTTP server and the same ``ChatEngine`` as the
Gradio prototype.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import format_citations

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COMPASS Chat</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #101113; color: #f4f4f5; }
    main { max-width: 1040px; margin: 0 auto; padding: 24px; min-height: 100vh; display: flex; flex-direction: column; gap: 16px; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; border-bottom: 1px solid #2f3338; padding-bottom: 12px; }
    h1 { font-size: 24px; margin: 0; letter-spacing: 0; }
    .scope { color: #b7bbc2; font-size: 14px; }
    #chat { flex: 1; border: 1px solid #30343a; background: #17191d; overflow: auto; padding: 16px; min-height: 520px; }
    .msg { max-width: 88%; margin: 0 0 14px; padding: 12px 14px; border-radius: 8px; white-space: pre-wrap; line-height: 1.45; }
    .user { margin-left: auto; background: #234a8b; }
    .assistant { margin-right: auto; background: #23272e; }
    .error { margin-right: auto; background: #4a2026; border: 1px solid #9f3a47; }
    form { display: flex; gap: 10px; }
    textarea { flex: 1; resize: vertical; min-height: 56px; max-height: 180px; padding: 12px; border-radius: 6px; border: 1px solid #3a3f47; background: #15171a; color: #fff; font: inherit; }
    button { width: 120px; border: 0; border-radius: 6px; background: #3d7cff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: wait; }
    code { color: #d7e0ff; }
    a { color: #8db3ff; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>COMPASS Chat</h1>
      <div class="scope">__SCOPE__</div>
    </header>
    <section id="chat" aria-live="polite">
      <div class="msg assistant">Bonjour. Pose une question sur le corpus indexe, par exemple: What does the party say about democracy?</div>
    </section>
    <form id="form">
      <textarea id="question" placeholder="Ask a question about the indexed corpus..."></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </main>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const question = document.getElementById('question');
    const send = document.getElementById('send');
    const history = [];

    function addMessage(role, text, cls, promptUrl) {
      const div = document.createElement('div');
      div.className = 'msg ' + (cls || role);
      div.textContent = text;
      if (promptUrl) {
        const link = document.createElement('a');
        link.href = promptUrl;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'Voir le prompt LLM';
        div.appendChild(document.createElement('br'));
        div.appendChild(document.createElement('br'));
        div.appendChild(link);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      addMessage('user', text);
      history.push({role: 'user', content: text});
      question.value = '';
      send.disabled = true;
      try {
        const response = await fetch('./ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: text, history})
        });
        const raw = await response.text();
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch (parseError) {
          throw new Error('Non-JSON response from server: ' + raw.slice(0, 240));
        }
        if (!response.ok || payload.error) {
          throw new Error(payload.error || ('HTTP ' + response.status));
        }
        addMessage('assistant', payload.answer, 'assistant', payload.prompt_url);
        history.push({role: 'assistant', content: payload.answer});
      } catch (err) {
        addMessage('assistant', 'Erreur COMPASS Chat: ' + err.message, 'error');
      } finally {
        send.disabled = false;
        question.focus();
      }
    });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch dependency-light COMPASS web chat.")
    parser.add_argument("--country", required=True, help="Country ISO3, for example DEU")
    parser.add_argument("--as-of", required=True, help="Temporal cutoff date, YYYY-MM-DD")
    parser.add_argument("--party", help="Optional party id filter")
    parser.add_argument("--k", type=int, default=8, help="Number of evidence segments to retrieve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    from compass.country_memory import CountryMemory

    cutoff = date.fromisoformat(args.as_of)
    engine = ChatEngine(CountryMemory(args.country))
    scope = f"Corpus: {args.country.upper()} | as_of={cutoff.isoformat()} | party={args.party or 'all'}"
    prompt_store: dict[str, list[dict[str, str]]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/prompt/"):
                self._send_prompt_page(path.removeprefix("/prompt/"))
                return
            if path not in {"/", "/index.html"}:
                self._send_json({"error": "not found"}, status=404)
                return
            html = HTML.replace("__SCOPE__", scope)
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/ask":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                question = str(payload.get("question") or "").strip()
                history = payload.get("history") if isinstance(payload.get("history"), list) else []
                payload_out = answer_question_payload(
                    engine=engine,
                    question=question,
                    history=history,
                    cutoff=cutoff,
                    party_id=args.party,
                    k=args.k,
                    prompt_store=prompt_store,
                )
                self._send_json(payload_out)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_prompt_page(self, prompt_id: str) -> None:
            messages = prompt_store.get(prompt_id)
            if messages is None:
                self._send_json({"error": "prompt not found"}, status=404)
                return
            body = html.escape(json.dumps(messages, ensure_ascii=False, indent=2))
            page = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>COMPASS LLM Prompt</title>"
                "<style>body{font-family:ui-monospace,Consolas,monospace;background:#101113;color:#f4f4f5;"
                "padding:24px;} pre{white-space:pre-wrap;line-height:1.45;background:#17191d;border:1px solid #30343a;"
                "padding:16px;border-radius:8px;}</style></head><body>"
                "<h1>COMPASS LLM Prompt</h1><pre>"
                + body
                + "</pre></body></html>"
            )
            data = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"COMPASS Chat running on http://{args.host}:{args.port}")
    print(scope)
    server.serve_forever()


def answer_question(
    *,
    engine: ChatEngine,
    question: str,
    history: list[dict[str, str]],
    cutoff: date,
    party_id: str | None,
    k: int,
) -> str:
    return answer_question_payload(
        engine=engine,
        question=question,
        history=history,
        cutoff=cutoff,
        party_id=party_id,
        k=k,
    )["answer"]


def answer_question_payload(
    *,
    engine: ChatEngine,
    question: str,
    history: list[dict[str, str]],
    cutoff: date,
    party_id: str | None,
    k: int,
    prompt_store: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, str]:
    if is_greeting(question):
        return {"answer": "Bonjour. Je suis COMPASS Chat. Pose une question sur le corpus indexe."}
    if is_source_followup(question):
        sources = latest_sources_from_history(history)
        if sources:
            return {"answer": "Voici les sources utilisees dans ma reponse precedente :\n\n" + sources}
    response = engine.ask(
        ChatRequest(
            question=question,
            as_of=cutoff,
            party_id=party_id,
            k=k,
            history=history,
        )
    )
    answer = response.answer + "\n\nSources\n" + format_citations(response.citations)
    payload = {"answer": answer}
    if prompt_store is not None and response.prompt_messages:
        prompt_id = uuid.uuid4().hex
        prompt_store[prompt_id] = response.prompt_messages
        payload["prompt_url"] = f"./prompt/{prompt_id}"
    return payload


def is_greeting(message: str) -> bool:
    text = (message or "").strip().lower()
    greetings = ("salut", "bonjour", "hello", "hi", "hey", "bonsoir", "ca va")
    return len(text) <= 40 and any(text == item or text.startswith(item + " ") or text.startswith(item + ",") for item in greetings)


def is_source_followup(message: str) -> bool:
    text = (message or "").strip().lower()
    markers = ("sources", "exact sources", "passages cites", "preuves", "evidence")
    return len(text) <= 90 and any(marker in text for marker in markers)


def latest_sources_from_history(history: list[dict[str, str]]) -> str:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        content = item.get("content") or ""
        marker = "\n\nSources\n"
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return ""


if __name__ == "__main__":
    main()
