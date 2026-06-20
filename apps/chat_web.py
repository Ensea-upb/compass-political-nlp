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
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import (
    citation_to_payload,
    describe_active_corpus,
    format_citations,
)


class ChatJobQueue:
    """Run long RAG requests outside the ingress HTTP request lifetime."""

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def submit(self, operation: Callable[[], dict[str, object]]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"status": "pending"}
        self._executor.submit(self._run, job_id, operation)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, operation: Callable[[], dict[str, object]]) -> None:
        with self._lock:
            self._jobs[job_id] = {"status": "running"}
        try:
            result = operation()
        except Exception as exc:
            job = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            job = {"status": "completed", "result": result}
        with self._lock:
            self._jobs[job_id] = job

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
    .routing { display: flex; align-items: center; gap: 8px; color: #b7bbc2; font-size: 14px; }
    .routing label { cursor: pointer; }
    .routing input { position: absolute; opacity: 0; pointer-events: none; }
    .routing span { display: inline-flex; min-height: 32px; align-items: center; padding: 0 10px; border: 1px solid #3a3f47; background: #17191d; color: #d5d8de; }
    .routing label:first-of-type span { border-radius: 6px 0 0 6px; }
    .routing label:last-of-type span { border-left: 0; border-radius: 0 6px 6px 0; }
    .routing input:checked + span { background: #2d5596; border-color: #568bf0; color: #ffffff; }
    .hidden { display: none; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; }
    .examples button { width: auto; min-height: 34px; padding: 0 10px; background: #252a31; border: 1px solid #3a3f47; font-weight: 500; }
    .response-meta { margin-top: 10px; color: #aeb4bd; font-size: 12px; }
    .sources { margin-top: 10px; border-top: 1px solid #3a3f47; padding-top: 8px; }
    .sources summary { cursor: pointer; color: #bcd0ff; }
    .sources pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, Consolas, monospace; color: #d7dae0; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>COMPASS Chat</h1>
      <div class="scope">__SCOPE__</div>
    </header>
    <div class="routing __ROUTING_CLASS__" role="group" aria-label="Mode de routage">
      <strong>Routage</strong>
      <label>
        <input type="radio" name="routing_mode" value="deterministic" checked>
        <span>Deterministe</span>
      </label>
      <label>
        <input type="radio" name="routing_mode" value="llm">
        <span>LLM</span>
      </label>
    </div>
    <div class="examples" aria-label="Exemples de questions">
      <button type="button" data-question="Que dit le parti sur la démocratie ?">Démocratie</button>
      <button type="button" data-question="Quelles priorités économiques apparaissent dans le manifeste ?">Économie</button>
      <button type="button" data-question="Tu es connecté à quel corpus ?">Corpus actif</button>
      <button type="button" data-question="/variables">Variables scientifiques</button>
      <button type="button" data-question="/valider">Valider la session</button>
    </div>
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
    let lastSources = [];

    function addMessage(role, text, cls, promptUrl, sourcesMarkdown, metaText) {
      const div = document.createElement('div');
      div.className = 'msg ' + (cls || role);
      div.textContent = text;
      if (promptUrl) {
        const link = document.createElement('a');
        link.href = promptUrl;
        link.target = 'compass_prompt_viewer';
        link.textContent = 'Voir le prompt LLM';
        link.addEventListener('click', (event) => {
          event.preventDefault();
          const promptWindow = window.open(promptUrl, 'compass_prompt_viewer');
          if (promptWindow) {
            promptWindow.focus();
          } else {
            window.location.href = promptUrl;
          }
        });
        div.appendChild(document.createElement('br'));
        div.appendChild(document.createElement('br'));
        div.appendChild(link);
      }
      if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'response-meta';
        meta.textContent = metaText;
        div.appendChild(meta);
      }
      if (sourcesMarkdown) {
        const details = document.createElement('details');
        details.className = 'sources';
        const summary = document.createElement('summary');
        summary.textContent = 'Preuves utilisées';
        const pre = document.createElement('pre');
        pre.textContent = sourcesMarkdown;
        details.appendChild(summary);
        details.appendChild(pre);
        div.appendChild(details);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      addMessage('user', text);
      question.value = '';
      send.disabled = true;
      send.textContent = 'Analyse...';
      try {
        const response = await fetch('./ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question: text,
            history,
            last_sources: lastSources,
            routing_mode: document.querySelector('input[name="routing_mode"]:checked').value
          })
        });
        let payload = await parseJsonResponse(response);
        if (response.status === 202 && payload.job_id) {
          payload = await waitForJob(payload.job_id);
        }
        if (!response.ok || payload.error) {
          throw new Error(payload.error || ('HTTP ' + response.status));
        }
        const meta = payload.route
          ? `route=${payload.route} | analyse=${(payload.query_analysis || {}).method || 'n/a'} | récupérés=${payload.retrieval_count || 0} | preuves LLM=${payload.prompt_citation_count || 0} | relations graphe=${payload.graph_context_count || 0} | sources affichées=${(payload.sources || []).length}`
          : '';
        addMessage('assistant', payload.answer, 'assistant', payload.prompt_url, payload.sources_markdown, meta);
        if (payload.sources && payload.sources.length) {
          lastSources = payload.sources;
        }
        history.push({role: 'user', content: text});
        history.push({role: 'assistant', content: payload.answer});
      } catch (err) {
        addMessage('assistant', 'Erreur COMPASS Chat: ' + err.message, 'error');
      } finally {
        send.disabled = false;
        send.textContent = 'Envoyer';
        question.focus();
      }
    });

    async function parseJsonResponse(response) {
      const raw = await response.text();
      try {
        return JSON.parse(raw);
      } catch (parseError) {
        throw new Error('Non-JSON response from server: ' + raw.slice(0, 240));
      }
    }

    async function waitForJob(jobId) {
      const deadline = Date.now() + 15 * 60 * 1000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const response = await fetch('./result/' + encodeURIComponent(jobId), {
          headers: {'Accept': 'application/json'}
        });
        const payload = await parseJsonResponse(response);
        if (response.status === 202) continue;
        if (!response.ok || payload.error) {
          throw new Error(payload.error || ('HTTP ' + response.status));
        }
        return payload.result;
      }
      throw new Error('Le pipeline COMPASS a depasse 15 minutes. Consultez les logs du serveur.');
    }
    document.querySelectorAll('[data-question]').forEach((button) => {
      button.addEventListener('click', () => {
        question.value = button.dataset.question;
        question.focus();
      });
    });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch dependency-light COMPASS web chat.")
    parser.add_argument("--country", required=True, help="Three-letter country ISO3 code")
    parser.add_argument("--as-of", required=True, help="Temporal cutoff date, YYYY-MM-DD")
    parser.add_argument("--party", help="Optional party id filter")
    parser.add_argument("--election-id", help="Election id for scientific /analyse commands")
    parser.add_argument("--k", type=int, default=8, help="Number of evidence segments to retrieve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--debug-routing",
        action="store_true",
        help="Show the deterministic/LLM routing selector in the interface",
    )
    args = parser.parse_args()

    from compass.country_memory import CountryMemory
    from compass.chat.scientific_service import ScientificChatService
    from compass.political_graph import PoliticalGraph

    cutoff = date.fromisoformat(args.as_of)
    graph = PoliticalGraph(args.country)
    graph.load()
    memory = CountryMemory(args.country)
    scientific_service = ScientificChatService(memory, graph)
    engine = ChatEngine(memory, graph=graph, scientific_service=scientific_service)
    scope_data = describe_active_corpus(
        engine.memory,
        ChatRequest(question="scope", as_of=cutoff, party_id=args.party),
    )
    scope = format_scope_banner(scope_data, cutoff, election_id=args.election_id)
    prompt_store: dict[str, dict[str, Any]] = {}
    job_queue = ChatJobQueue(max_workers=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/result/"):
                self._send_job_result(path.removeprefix("/result/"))
                return
            if path.startswith("/prompt/"):
                self._send_prompt_page(path.removeprefix("/prompt/"))
                return
            if path not in {"/", "/index.html"}:
                self._send_json({"error": "not found"}, status=404)
                return
            html_page = HTML.replace("__SCOPE__", html.escape(scope)).replace(
                "__ROUTING_CLASS__",
                "" if args.debug_routing else "hidden",
            )
            data = html_page.encode("utf-8")
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
                last_sources = payload.get("last_sources") if isinstance(payload.get("last_sources"), list) else []
                routing_mode = str(payload.get("routing_mode") or "deterministic")
                job_id = job_queue.submit(
                    lambda: answer_question_payload(
                        engine=engine,
                        question=question,
                        history=history,
                        cutoff=cutoff,
                        party_id=args.party,
                        election_id=args.election_id,
                        k=args.k,
                        prompt_store=prompt_store,
                        routing_mode=routing_mode,
                        previous_citations=last_sources,
                    )
                )
                self._send_json({"job_id": job_id, "status": "pending"}, status=202)
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
            inspection = prompt_store.get(prompt_id)
            if inspection is None:
                self._send_json({"error": "prompt not found"}, status=404)
                return
            page = render_prompt_page(inspection)
            data = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_job_result(self, job_id: str) -> None:
            job = job_queue.get(job_id)
            if job is None:
                self._send_json({"error": "job not found"}, status=404)
                return
            if job["status"] in {"pending", "running"}:
                self._send_json({"status": job["status"]}, status=202)
                return
            if job["status"] == "failed":
                self._send_json({"error": job.get("error", "job failed")}, status=500)
                return
            self._send_json(job)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"COMPASS Chat running on http://{args.host}:{args.port}")
    print(scope)
    try:
        server.serve_forever()
    finally:
        job_queue.close()


def answer_question(
    *,
    engine: ChatEngine,
    question: str,
    history: list[dict[str, str]],
    cutoff: date,
    party_id: str | None,
    election_id: str | None = None,
    k: int,
    routing_mode: str = "deterministic",
    previous_citations: list[dict] | None = None,
) -> str:
    return answer_question_payload(
        engine=engine,
        question=question,
        history=history,
        cutoff=cutoff,
        party_id=party_id,
        election_id=election_id,
        k=k,
        routing_mode=routing_mode,
        previous_citations=previous_citations,
    )["answer"]


def answer_question_payload(
    *,
    engine: ChatEngine,
    question: str,
    history: list[dict[str, str]],
    cutoff: date,
    party_id: str | None,
    election_id: str | None = None,
    k: int,
    prompt_store: dict[str, dict[str, Any]] | None = None,
    routing_mode: str = "deterministic",
    previous_citations: list[dict] | None = None,
) -> dict[str, object]:
    if is_greeting(question):
        return {
            "answer": (
                "Bonjour. Je suis COMPASS Chat. Pose une question sur le corpus indexé, "
                "ou utilise /variables puis /analyse <variable_id> pour le pipeline scientifique."
            ),
            "route": "greeting",
            "sources": [],
            "sources_markdown": "",
            "retrieval_count": 0,
            "prompt_citation_count": 0,
            "graph_context_count": 0,
        }
    response = engine.ask(
        ChatRequest(
            question=question,
            as_of=cutoff,
            party_id=party_id,
            election_id=election_id,
            k=k,
            history=history,
            routing_mode=routing_mode,
            previous_citations=previous_citations or [],
        )
    )
    source_items = [citation_to_payload(citation) for citation in response.citations]
    payload: dict[str, object] = {
        "answer": response.answer,
        "sources": source_items,
        "sources_markdown": format_citations(response.citations) if response.citations else "",
        "route": response.route,
        "retrieval_count": response.retrieval_count,
        "prompt_citation_count": response.prompt_citation_count,
        "graph_context_count": len(getattr(response, "graph_context", [])),
        "query_analysis": getattr(response, "query_analysis", {}),
        "retrieval_trace": getattr(response, "retrieval_trace", []),
        "validation_trace": getattr(response, "validation_trace", []),
    }
    if prompt_store is not None and response.prompt_messages:
        prompt_id = uuid.uuid4().hex
        prompt_store[prompt_id] = {
            "messages": response.prompt_messages,
            "query_analysis": getattr(response, "query_analysis", {}),
            "retrieval_trace": getattr(response, "retrieval_trace", []),
            "validation_trace": getattr(response, "validation_trace", []),
        }
        payload["prompt_url"] = f"./prompt/{prompt_id}"
    return payload


def is_greeting(message: str) -> bool:
    text = (message or "").strip().lower()
    greetings = ("salut", "bonjour", "hello", "hi", "hey", "bonsoir", "ca va")
    return len(text) <= 40 and any(text == item or text.startswith(item + " ") or text.startswith(item + ",") for item in greetings)


def format_scope_banner(
    scope: dict[str, object],
    cutoff: date,
    election_id: str | None = None,
) -> str:
    parties = []
    for item in scope.get("parties") or []:
        if isinstance(item, dict):
            label = str(item.get("party_id") or item.get("name") or "").strip()
            if label:
                parties.append(label)
    party_text = ", ".join(parties) if parties else "non renseigné"
    doc_types = ", ".join(str(value) for value in scope.get("document_types") or []) or "non renseigné"
    return (
        f"Corpus actif : {scope.get('country_iso3') or 'non renseigné'} | "
        f"partis={party_text} | documents={scope.get('n_documents', 0)} | "
        f"types={doc_types} | as_of={cutoff.isoformat()} | "
        f"election_id={election_id or 'non renseigné'} | "
        "mode=RAG + pipeline scientifique"
    )


def render_prompt_page(
    inspection: list[dict[str, str]] | dict[str, Any],
) -> str:
    if isinstance(inspection, list):
        messages = inspection
        audit: dict[str, Any] = {}
    else:
        messages = list(inspection.get("messages") or [])
        audit = inspection
    cards = "\n".join(_render_prompt_message(message) for message in messages)
    audit_cards = _render_audit_cards(audit)
    raw = html.escape(json.dumps(messages, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COMPASS LLM Prompt</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #101113; color: #f4f4f5; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    header {{ border-bottom: 1px solid #30343a; margin-bottom: 18px; padding-bottom: 14px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    .hint {{ color: #b7bbc2; margin: 0; line-height: 1.45; }}
    .message {{ border: 1px solid #30343a; background: #17191d; border-radius: 8px; margin: 16px 0; overflow: hidden; }}
    .role {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 14px; background: #23272e; color: #dbe7ff; font-weight: 700; }}
    .content {{ padding: 14px; white-space: pre-wrap; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: 14px; }}
    .system .role {{ background: #2b3340; }}
    .user .role {{ background: #233754; }}
    .assistant .role {{ background: #2b3b32; }}
    .audit .role {{ background: #493d24; color: #fff0c2; }}
    mark {{ background: #384d7a; color: #ffffff; padding: 0 3px; border-radius: 3px; }}
    details {{ margin-top: 22px; border: 1px solid #30343a; border-radius: 8px; background: #17191d; }}
    summary {{ cursor: pointer; padding: 12px 14px; font-weight: 700; }}
    pre {{ white-space: pre-wrap; line-height: 1.45; margin: 0; padding: 14px; border-top: 1px solid #30343a; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Prompt envoye au LLM</h1>
      <p class="hint">Les cartes SYSTEM, USER et ASSISTANT reproduisent les messages réellement transmis à vLLM. La section AUDIT est affichée pour la traçabilité, mais elle n'est pas envoyée au modèle. Seules les sources <code>[Sx]</code> peuvent justifier les affirmations.</p>
    </header>
    {cards}
    {audit_cards}
    <details>
      <summary>Voir le JSON exact envoye</summary>
      <pre>{raw}</pre>
    </details>
  </main>
</body>
</html>"""


def _render_audit_cards(audit: dict[str, Any]) -> str:
    sections = []
    values = (
        ("QUESTION_ANALYSIS", audit.get("query_analysis")),
        ("RETRIEVAL_TRACE", audit.get("retrieval_trace")),
        ("VALIDATION_TRACE", audit.get("validation_trace")),
    )
    for title, value in values:
        if not value:
            continue
        content = html.escape(json.dumps(value, ensure_ascii=False, indent=2))
        sections.append(
            "<section class='message audit'>"
            f"<div class='role'><span>{title}</span><span>audit non envoyé</span></div>"
            f"<div class='content'>{content}</div>"
            "</section>"
        )
    if not sections:
        return ""
    return "<h2>Audit hors prompt LLM</h2>" + "\n".join(sections)


def _render_prompt_message(message: dict[str, str]) -> str:
    role = html.escape(message.get("role") or "unknown")
    content = _highlight_prompt_content(html.escape(message.get("content") or ""))
    return (
        f"<section class='message {role}'>"
        f"<div class='role'><span>{role.upper()}</span><span>message</span></div>"
        f"<div class='content'>{content}</div>"
        "</section>"
    )


def _highlight_prompt_content(content: str) -> str:
    replacements = {
        "ANALYTICAL_CONTEXT": "<mark>ANALYTICAL_CONTEXT</mark>",
        "GENERAL_CONTEXT": "<mark>GENERAL_CONTEXT</mark>",
        "RELATIONAL_CONTEXT": "<mark>RELATIONAL_CONTEXT</mark>",
        "RETRIEVAL_TRACE": "<mark>RETRIEVAL_TRACE</mark>",
        "PRIMARY_EVIDENCE": "<mark>PRIMARY_EVIDENCE</mark>",
        "NUANCE_EVIDENCE": "<mark>NUANCE_EVIDENCE</mark>",
        "COUNTER_EVIDENCE_CANDIDATES": "<mark>COUNTER_EVIDENCE_CANDIDATES</mark>",
        "CITED_EVIDENCE": "<mark>CITED_EVIDENCE</mark>",
        "Answer contract": "<mark>Answer contract</mark>",
    }
    for needle, replacement in replacements.items():
        content = content.replace(needle, replacement)
    return content


if __name__ == "__main__":
    main()
