"""Gradio chat application for COMPASS memories.

Run on Onyxia after ingesting documents:

    python apps/chat_gradio.py --country DEU --as-of 2009-09-27 --party 41320
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
    parser.add_argument("--country", required=True, help="Country ISO3, for example DEU")
    parser.add_argument("--as-of", required=True, help="Temporal cutoff date, YYYY-MM-DD")
    parser.add_argument("--party", help="Optional party id filter")
    parser.add_argument("--k", type=int, default=8, help="Number of evidence segments to retrieve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the chat UI dependency first: pip install -r requirements-chat.txt") from exc

    from compass.country_memory import CountryMemory

    engine = ChatEngine(CountryMemory(args.country))
    cutoff = date.fromisoformat(args.as_of)

    def respond(message: str, history: list[dict[str, str]]):
        request = ChatRequest(
            question=message,
            as_of=cutoff,
            party_id=args.party,
            k=args.k,
            history=history,
        )
        response = engine.ask(request)
        answer = response.answer + "\n\n### Sources\n" + format_citations(response.citations)
        return answer

    demo = gr.ChatInterface(
        fn=respond,
        title="COMPASS Chat",
        description=f"Corpus: {args.country.upper()} | as_of={cutoff.isoformat()} | party={args.party or 'all'}",
        type="messages",
    )
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()