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
import uuid
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import trafilatura
from lingua import LanguageDetectorBuilder
from PIL import Image

from compass.schemas import DocumentMeta, Segment, SourceReliability, TemporalStatus

logger = logging.getLogger(__name__)

_LANG_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()


class DocumentPipeline:
    """Transforme une source brute en liste de ``Segment`` indexables.

    Chaque méthode publique retourne des segments porteurs de leurs métadonnées
    complètes — un document sans date ou sans origine est rejeté, pas réparé.
    """

    def __init__(self, min_chars_for_text_pdf: int = 200) -> None:
        self._min_chars = min_chars_for_text_pdf

    # ------------------------------------------------------------------ ingestion
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
        text = " ".join(text.split())  # normalisation des espaces uniquement
        meta.com
