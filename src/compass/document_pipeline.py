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
    - Segmentation     : spaCy (sentencizer multilingue) — logique quasi-phrases CMP.

CUSTOM (justifié) : uniquement l'orchestration et le contrat de métadonnées
(DocumentMeta) — aucun outil existant n'impose pays/parti/date/élection/source,
or ce contrat est la condition du contrôle temporel (C15).
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import trafilatura
from lingua import LanguageDetectorBuilder
from PIL import Image

from compass.config import settings
from compass.schemas import DocumentMeta, Segment, SourceReliability, TemporalStatus

logger = logging.getLogger(__name__)

_LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()


_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9À-ÖØ-Þ\"'«(])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

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
        paragraphs = _paragraph_units(segmented_text)
        if not paragraphs:
            paragraphs = [[flat_text]]

        doc_id = meta.doc_id or str(uuid.uuid4())
        meta.doc_id = doc_id

        parents: list[Segment] = []
        children: list[Segment] = []
        buf: list[str] = []
        buf_len = 0
        parent_idx = 0

        def _flush_parent() -> None:
            nonlocal parent_idx
            if not buf:
                return
            parent_block = " ".join(buf)
            parent_id = f"{doc_id}:p{parent_idx:03d}"
            parent_seg = Segment(
                segment_id=parent_id,
                doc_id=doc_id,
                text=parent_block,
                meta=meta.model_copy(deep=True),
                parent_segment_id=None,
            )
            parents.append(parent_seg)
            for c_idx, sent in enumerate(_child_units(buf)):
                child_id = f"{doc_id}:p{parent_idx:03d}c{c_idx:03d}"
                children.append(Segment(
                    segment_id=child_id,
                    doc_id=doc_id,
                    text=sent,
                    meta=meta.model_copy(deep=True),
                    parent_segment_id=parent_id,
                ))
            parent_idx += 1

        for paragraph in paragraphs:
            para_len = sum(len(unit) + 1 for unit in paragraph)
            if buf and buf_len + para_len > settings.parent_chunk_size:
                _flush_parent()
                buf, buf_len = [], 0
            for unit in paragraph:
                if buf and buf_len + len(unit) > settings.parent_chunk_size:
                    _flush_parent()
                    buf, buf_len = [], 0
                buf.append(unit)
                buf_len += len(unit) + 1
        _flush_parent()

        # Parents en premier (indexés comme contexte), enfants ensuite (indexés pour retrieval)
        return parents + children


def _normalize_for_chunking(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def _paragraph_units(text: str) -> list[list[str]]:
    paragraphs: list[list[str]] = []
    for paragraph in _BLANK_LINE_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units = _split_paragraph(paragraph)
        if units:
            paragraphs.append(units)
    return paragraphs


def _split_paragraph(paragraph: str) -> list[str]:
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1 and any(_BULLET_RE.match(line) for line in lines):
        units = [_BULLET_RE.sub("", line).strip() for line in lines]
    else:
        units = [part.strip() for part in _SENT_RE.split(" ".join(lines)) if part.strip()]
    return _split_long_units(units)


def _child_units(units: list[str]) -> list[str]:
    """Merge tiny fragments and split oversized children into citable units."""
    merged: list[str] = []
    pending = ""
    min_chars = max(1, settings.child_chunk_min_chars)
    max_chars = max(min_chars, settings.child_chunk_max_chars)

    for unit in _split_long_units(units, max_chars=max_chars):
        if not pending:
            pending = unit
            continue
        if len(pending) < min_chars and len(pending) + len(unit) + 1 <= max_chars:
            pending = f"{pending} {unit}"
            continue
        merged.append(pending)
        pending = unit

    if pending:
        if merged and len(pending) < min_chars and len(merged[-1]) + len(pending) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]} {pending}"
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
