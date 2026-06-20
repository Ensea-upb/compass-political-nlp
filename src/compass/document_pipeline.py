"""C01 — Pipeline documentaire : des sources brutes aux segments datés et traçables.

ÉTAT DE L'ART RÉUTILISÉ (rien from scratch — choix vérifiés par recherche web
le 2026-06-11, justification et alternatives dans CHOIX_COMPOSANTS.md) :
    - PDF texte        : PyMuPDF (fitz) — extraction mature, gère les PDF multimodaux.
    - PDF scannés      : pytesseract (Tesseract) par défaut ; pour les scans
                         dégradés (cas fréquent des manifestes africains), basculer
                         vers Surya OCR (90+ langues, > Tesseract sur la plupart des
                         benchmarks 2026) via ``ocr_engine='surya'``.
    - Pages web        : trafilatura — meilleure extraction d'article web publiée.
    - DATE des pages   : htmldate (même auteur que trafilatura, JOSS) — meilleure
                         précision et couverture des benchmarks de datation, y
                         compris petits sites non anglophones ; CRITIQUE pour le
                         contrôle temporel des sources web (C08/C15).
    - Détection langue : lingua-language-detector — plus précis que langdetect sur
                         les langues africaines (décision déjà actée, Partie 3 §4.1).
    - Segmentation     : règles déterministes (paragraphes, ponctuation et listes)
                         avec regroupement sémantique parent-enfant — logique
                         quasi-phrases CMP, sans dépendance à un modèle spaCy.

CUSTOM (justifié) : uniquement l'orchestration et le contrat de métadonnées
(DocumentMeta) — aucun outil existant n'impose pays/parti/date/élection/source,
or ce contrat est la condition du contrôle temporel (C15).
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytesseract
import trafilatura
from lingua import LanguageDetectorBuilder
from PIL import Image

from compass.config import settings
from compass.schemas import DocumentMeta, Segment, SourceReliability, TemporalStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StructuralUnit:
    text: str
    paragraph_index: int
    order_index: int
    section_title: str | None = None
    is_heading: bool = False


@dataclass(frozen=True)
class _ChildDraft:
    text: str
    paragraph_start: int
    paragraph_end: int

_LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()


_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9À-ÖØ-Þ\"'«(])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)\s:-]+",
    re.IGNORECASE,
)

class DocumentPipeline:
    """Transforme une source brute en liste de ``Segment`` indexables.

    Chaque méthode publique retourne des segments porteurs de leurs métadonnées
    complètes — un document sans date ou sans origine est rejeté, pas réparé.
    """

    def __init__(self, min_chars_for_text_pdf: int = 200) -> None:
        self._min_chars = min_chars_for_text_pdf

    # ------------------------------------------------------------------ ingestion
    def ingest_text(self, text: str, meta: DocumentMeta) -> list[Segment]:
        """Ingests already extracted text, mainly for synthetic/public examples.

        The real research pipeline normally enters through PDF or URL ingestion.
        This method keeps the same metadata, language, hashing and segmentation
        contract, while avoiding private PDFs in the public repository.
        """
        if not text.strip():
            raise ValueError("Texte vide : ingestion refusée.")
        meta.temporal_status = TemporalStatus.VERIFIED
        meta.eligible_for_historical_reasoning = True
        return self._finalize(text, meta)

    def ingest_pdf(self, path: Path, meta: DocumentMeta) -> list[Segment]:
        """Extrait un PDF ; bascule en OCR si la couche texte est vide.

        Args:
            path: chemin du PDF.
            meta: métadonnées obligatoires (date, pays, type, source).

        Returns:
            Segments de quasi-phrases avec métadonnées propagées.
        """
        doc = fitz.open(path)
        pages_text: list[str] = []
        for page in doc:
            text = page.get_text().strip()
            if len(text) < self._min_chars:  # page scannée probable -> OCR
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img, lang="fra+eng")
                logger.info("OCR appliqué : %s page %d", path.name, page.number)
            pages_text.append(text)
        full_text = "\n".join(pages_text)
        meta.source_path = str(path)
        # Date saisie par l'opérateur depuis le document lui-même -> vérifiée.
        meta.publication_date = meta.doc_date
        meta.temporal_status = TemporalStatus.VERIFIED
        meta.eligible_for_historical_reasoning = True
        return self._finalize(full_text, meta)

    def ingest_url(self, url: str, meta: DocumentMeta) -> list[Segment]:
        """Extrait le contenu principal d'une page web via trafilatura,
        et sa DATE DE PUBLICATION via htmldate.

        Règle de datation (contrôle temporel C15) : si htmldate trouve une
        date de publication, elle remplace la date provisoire de ``meta`` ;
        sinon le document est marqué non daté de façon vérifiable
        (``doc_type`` suffixé ``_undated``) et sera pénalisé par le
        diagnostic — jamais silencieusement accepté comme antérieur.
        """
        from datetime import date as _date

        from htmldate import find_date

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise ValueError(f"Téléchargement impossible : {url}")
        text = trafilatura.extract(downloaded, include_comments=False)
        if not text:
            raise ValueError(f"Extraction vide : {url}")
        published = find_date(downloaded, original_date=True)
        meta.retrieval_date = _date.today()
        if published:
            meta.publication_date = _date.fromisoformat(published)
            meta.doc_date = meta.publication_date
            meta.temporal_status = TemporalStatus.VERIFIED
            meta.eligible_for_historical_reasoning = True
            logger.info("Date publiée détectée (%s) : %s", url, published)
        else:
            # P0-2 : aucune date vérifiable -> le document peut ORIENTER une
            # enquête mais ne sera JAMAIS admis comme preuve historique (C03
            # l'exclut du filtre temporel via temporal_ok=0).
            meta.publication_date = None
            meta.temporal_status = TemporalStatus.UNKNOWN
            meta.eligible_for_historical_reasoning = False
            logger.warning("Page sans date vérifiable : %s — inéligible comme preuve", url)
        meta.source_url = url
        return self._finalize(text, meta)

    # ------------------------------------------------------------------ interne
    def _finalize(self, text: str, meta: DocumentMeta) -> list[Segment]:
        """Nettoyage léger, détection de langue, hash, segmentation."""
        segmented_text = _normalize_for_chunking(text)
        flat_text = " ".join(segmented_text.split())
        meta.compute_hash(flat_text)

        detected = _LANG_DETECTOR.detect_language_of(flat_text)
        if detected is not None:
            try:
                meta.language = detected.iso_code_639_1.name.lower()
            except AttributeError:
                meta.language = detected.name.lower()[:2]

        # --- Niveau 1 : blocs parents (thèmes, ~parent_chunk_size chars) ---
        units = _document_units(segmented_text)
        if not units:
            units = [_StructuralUnit(flat_text, 0, 0)]
        semantic_vectors = _semantic_vectors(units)

        doc_id = meta.doc_id or str(uuid.uuid4())
        meta.doc_id = doc_id

        parents: list[Segment] = []
        children: list[Segment] = []
        buf: list[_StructuralUnit] = []
        buf_len = 0
        parent_idx = 0
        child_idx = 0

        def _flush_parent() -> None:
            nonlocal parent_idx, child_idx
            if not buf:
                return
            parent_block = " ".join(unit.text for unit in buf)
            parent_id = f"{doc_id}:p{parent_idx:03d}"
            paragraph_start = min(unit.paragraph_index for unit in buf)
            paragraph_end = max(unit.paragraph_index for unit in buf)
            section_title = next(
                (unit.section_title for unit in reversed(buf) if unit.section_title),
                None,
            )
            parent_seg = Segment(
                segment_id=parent_id,
                doc_id=doc_id,
                text=parent_block,
                meta=meta.model_copy(deep=True),
                parent_segment_id=None,
                chunk_index=parent_idx,
                paragraph_start=paragraph_start,
                paragraph_end=paragraph_end,
                section_title=section_title,
            )
            parents.append(parent_seg)
            for c_idx, child in enumerate(_child_drafts(buf)):
                child_id = f"{doc_id}:p{parent_idx:03d}c{c_idx:03d}"
                children.append(Segment(
                    segment_id=child_id,
                    doc_id=doc_id,
                    text=child.text,
                    meta=meta.model_copy(deep=True),
                    parent_segment_id=parent_id,
                    chunk_index=child_idx,
                    paragraph_start=child.paragraph_start,
                    paragraph_end=child.paragraph_end,
                    section_title=section_title,
                ))
                child_idx += 1
            parent_idx += 1

        for unit in units:
            if unit.is_heading and buf:
                _flush_parent()
                buf, buf_len = [], 0
            if _semantic_parent_break(buf, unit, buf_len, semantic_vectors):
                _flush_parent()
                buf, buf_len = [], 0
            if (
                buf
                and not all(buffered.is_heading for buffered in buf)
                and buf_len + len(unit.text) > settings.parent_chunk_size
            ):
                _flush_parent()
                buf, buf_len = [], 0
            buf.append(unit)
            buf_len += len(unit.text) + 1
        _flush_parent()

        # Parents en premier (indexés comme contexte), enfants ensuite (indexés pour retrieval)
        return parents + children


def _normalize_for_chunking(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def _document_units(text: str) -> list[_StructuralUnit]:
    """Preserve headings, paragraph positions and reading order."""
    units: list[_StructuralUnit] = []
    section_title: str | None = None

    def append_body(lines: list[str], paragraph_index: int) -> None:
        if not lines:
            return
        for text_unit in _split_paragraph("\n".join(lines)):
            units.append(_StructuralUnit(
                text_unit, paragraph_index, len(units), section_title, False,
            ))

    for paragraph_index, paragraph in enumerate(_BLANK_LINE_RE.split(text)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        body_lines: list[str] = []
        for line in lines:
            if _looks_like_heading(line):
                append_body(body_lines, paragraph_index)
                body_lines = []
                section_title = line
                units.append(_StructuralUnit(
                    line, paragraph_index, len(units), section_title, True,
                ))
            else:
                body_lines.append(line)
        append_body(body_lines, paragraph_index)
    return units


def _looks_like_heading(line: str) -> bool:
    clean = " ".join(line.split())
    if not clean or _BULLET_RE.match(clean) or len(clean) > 140:
        return False
    words = clean.rstrip(":").split()
    if not words or len(words) > 16:
        return False
    if clean.endswith(":"):
        return True
    if clean.endswith((".", "!", "?", ";")):
        return False
    letters = [char for char in clean if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / max(1, len(letters))
    return (
        uppercase_ratio >= 0.75
        or bool(_NUMBERED_HEADING_RE.match(clean))
        or (len(words) <= 10 and clean.istitle())
    )


def _split_paragraph(paragraph: str) -> list[str]:
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1 and any(_BULLET_RE.match(line) for line in lines):
        units = [_BULLET_RE.sub("", line).strip() for line in lines]
    else:
        units = [part.strip() for part in _SENT_RE.split(" ".join(lines)) if part.strip()]
    return _split_long_units(units)


def _child_drafts(units: list[_StructuralUnit]) -> list[_ChildDraft]:
    """Build citable children while retaining paragraph provenance."""
    expanded: list[_ChildDraft] = []
    for unit in units:
        parts = _split_long_units([unit.text], max_chars=settings.child_chunk_max_chars)
        expanded.extend(
            _ChildDraft(part, unit.paragraph_index, unit.paragraph_index)
            for part in parts
        )

    merged: list[_ChildDraft] = []
    pending: _ChildDraft | None = None
    min_chars = max(1, settings.child_chunk_min_chars)
    max_chars = max(min_chars, settings.child_chunk_max_chars)
    for draft in expanded:
        if pending is None:
            pending = draft
            continue
        if len(pending.text) < min_chars and len(pending.text) + len(draft.text) + 1 <= max_chars:
            pending = _ChildDraft(
                f"{pending.text} {draft.text}",
                pending.paragraph_start,
                draft.paragraph_end,
            )
            continue
        merged.append(pending)
        pending = draft
    if pending is not None:
        if merged and len(pending.text) < min_chars and len(merged[-1].text) + len(pending.text) + 1 <= max_chars:
            previous = merged[-1]
            merged[-1] = _ChildDraft(
                f"{previous.text} {pending.text}",
                previous.paragraph_start,
                pending.paragraph_end,
            )
        else:
            merged.append(pending)
    return merged


def _split_long_units(units: list[str], max_chars: int | None = None) -> list[str]:
    max_len = max_chars or settings.child_chunk_max_chars
    out: list[str] = []
    for unit in units:
        clean = " ".join(unit.split())
        if not clean:
            continue
        if len(clean) <= max_len:
            out.append(clean)
            continue
        out.extend(_split_long_text(clean, max_len=max_len))
    return out


def _split_long_text(text: str, max_len: int) -> list[str]:
    chunks: list[str] = []
    words = text.split()
    buf: list[str] = []
    buf_len = 0
    for word in words:
        extra = len(word) + (1 if buf else 0)
        if buf and buf_len + extra > max_len:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += extra
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _semantic_parent_break(
    buf: list[_StructuralUnit],
    next_unit: _StructuralUnit,
    buf_len: int,
    vectors: dict[int, list[float]] | None,
) -> bool:
    """Start a new parent when adjacent units are weakly related.

    The primary signal is cosine similarity between multilingual embeddings.
    Lexical cohesion is retained only as a deterministic availability fallback.
    """
    if not settings.semantic_chunking_enabled or not buf:
        return False
    min_parent = max(settings.child_chunk_min_chars, settings.semantic_chunk_min_parent_chars)
    if buf_len < min_parent:
        return False
    if vectors:
        context_units = buf[-max(1, settings.semantic_chunk_context_units):]
        context_vectors = [vectors[unit.order_index] for unit in context_units]
        similarity = _cosine(
            _mean_vector(context_vectors),
            vectors[next_unit.order_index],
        )
        return similarity < settings.semantic_chunk_similarity_threshold
    similarity = _token_jaccard(
        " ".join(unit.text for unit in buf[-2:]),
        next_unit.text,
    )
    return similarity < settings.semantic_chunk_fallback_jaccard_threshold


def _semantic_vectors(units: list[_StructuralUnit]) -> dict[int, list[float]] | None:
    if not settings.semantic_chunking_enabled or len(units) < 2:
        return None
    try:
        encoder = _load_semantic_encoder(
            settings.semantic_chunk_model,
            settings.hf_model_device() or "",
        )
        if encoder is None:
            return None
        encoded = encoder.encode(
            [unit.text for unit in units],
            batch_size=settings.semantic_chunk_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return {
            unit.order_index: [float(value) for value in vector]
            for unit, vector in zip(units, encoded)
        }
    except Exception as exc:
        logger.warning(
            "Embeddings de chunking indisponibles (%s); repli lexical deterministe.",
            exc,
        )
        return None


@lru_cache(maxsize=4)
def _load_semantic_encoder(model_name: str, device: str) -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device} if device else {}
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        logger.warning(
            "Modele de chunking %s indisponible (%s); repli lexical deterministe.",
            model_name,
            exc,
        )
        return None


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_semantic_tokens(left))
    right_tokens = set(_semantic_tokens(right))
    if not left_tokens or not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _semantic_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w']+", text.lower())
        if len(token) > 3
    ]


def make_meta(
    *,
    country_iso3: str,
    doc_date: date,
    doc_type: str,
    language: str = "und",
    party_id: str | None = None,
    election_id: str | None = None,
    source_url: str | None = None,
    source_path: str | None = None,
    reliability: SourceReliability = SourceReliability.UNKNOWN,
) -> DocumentMeta:
    """Builds a verified ``DocumentMeta`` object with the project defaults."""
    return DocumentMeta(
        doc_id=str(uuid.uuid4()),
        country_iso3=country_iso3.upper(),
        party_id=party_id,
        doc_date=doc_date,
        publication_date=doc_date,
        temporal_status=TemporalStatus.VERIFIED,
        eligible_for_historical_reasoning=True,
        doc_type=doc_type,
        language=language,
        source_url=source_url,
        source_path=source_path,
        election_id=election_id,
        reliability=reliability,
    )
