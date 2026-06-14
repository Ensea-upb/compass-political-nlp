"""Public document ingestion pipeline for local text examples."""

from __future__ import annotations

from pathlib import Path

from compass.schemas import Document


def load_text_document(path: str | Path, language: str = "en") -> Document:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Document is empty: {file_path}")
    return Document(name=file_path.name, text=text, language=language)
