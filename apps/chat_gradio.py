"""Gradio chat application for COMPASS memories.

Run on Onyxia after ingesting documents:

    python apps/chat_gradio.py --country <ISO3> --as-of <YYYY-MM-DD> --party <PARTY_ID>
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.chat import ChatEngine, ChatRequest
from compass.chat.engine import format_citations


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch COMPASS Chat over an indexed country memory.")
    parser.add_argument("--country", required=True, help="Three-letter country ISO3 code")
    parser.add_argument("--as-of", required=True, help="Temporal cutoff date, YYYY-MM-DD")
    parser.add_argument("--party", help="Optional party id filter")
    parser.add_argument("--election-id", help="Election id for scientific /analyse commands")
    parser.add_argument("--k", type=int, default=8, help="Number of evidence segments to retrieve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the chat UI dependency first: pip install -r requirements-chat.txt") from exc

    from compass.country_memory import CountryMemory
    from compass.chat.scientific_service import ScientificChatService
    from compass.political_graph import PoliticalGraph

    graph = PoliticalGraph(args.country)
    graph.load()
    memory = CountryMemory(args.country)
    engine = ChatEngine(
        memory,
        graph=graph,
        scientific_service=ScientificChatService(memory, graph),
    )
    cutoff = date.fromisoformat(args.as_of)

    def respond(message: str, history):
        history = history or []
        answer = _answer_message(
            message=message,
            history=history,
            engine=engine,
            cutoff=cutoff,
            party_id=args.party,
            election_id=args.election_id,
            k=args.k,
        )
        history.append((message, answer))
        return history, ""

    with gr.Blocks(title="COMPASS Chat") as demo:
        gr.Markdown("# COMPASS Chat")
        gr.Markdown(f"Corpus: **{args.country.upper()}** | as_of={cutoff.isoformat()} | party={args.party or 'all'}")
        chatbot = gr.Chatbot(label="COMPASS Chat")
        message = gr.Textbox(label="Question", placeholder="What does the party say about democracy?")
        send = gr.Button("Send")
        message.submit(respond, inputs=[message, chatbot], outputs=[chatbot, message])
        send.click(respond, inputs=[message, chatbot], outputs=[chatbot, message])

    demo.launch(server_name=args.host, server_port=args.port)





def _answer_message(
    message: str,
    history,
    engine: ChatEngine,
    cutoff: date,
    party_id: str | None,
    k: int,
    election_id: str | None = None,
) -> str:
    try:
        if _is_greeting(message):
            return (
                "Bonjour. Je suis COMPASS Chat. Pose une question sur le corpus indexé, "
                "par exemple : What does the party say about democracy?"
            )
        request = ChatRequest(
            question=message,
            as_of=cutoff,
            party_id=party_id,
            election_id=election_id,
            k=k,
            history=_normalize_history(history),
        )
        response = engine.ask(request)
        return response.answer + "\n\n### Sources\n" + format_citations(response.citations)
    except Exception as exc:
        return (
            "Erreur COMPASS Chat : "
            f"{type(exc).__name__}: {exc}\n\n"
            "Vérifie que le corpus a été ingéré, que COMPASS_CHROMA_DIR pointe vers le bon dossier, "
            "et que le pays/parti/date existent dans l'index."
        )

def _is_greeting(message: str) -> bool:
    text = (message or "").strip().lower()
    greetings = ("salut", "bonjour", "hello", "hi", "hey", "bonsoir", "ça va", "ca va")
    return len(text) <= 40 and any(text == item or text.startswith(item + " ") or text.startswith(item + ",") for item in greetings)

def _normalize_history(history) -> list[dict[str, str]]:
    """Accept both old Gradio tuple history and newer message dictionaries."""
    if not history:
        return []
    if isinstance(history[0], dict):
        return [item for item in history if item.get("role") in {"user", "assistant"}]
    messages: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        user_msg, assistant_msg = item
        if user_msg:
            messages.append({"role": "user", "content": str(user_msg)})
        if assistant_msg:
            messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages

if __name__ == "__main__":
    main()
