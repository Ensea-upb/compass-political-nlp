"""C15 — Couche transversale de garde-fous : temporalité, traçabilité, contamination.

ÉTAT DE L'ART RÉUTILISÉ :
    - Traçabilité : structlog (journalisation structurée JSON) + hashlib (stdlib).
    - Contrôle temporel : déjà appliqué nativement par C03 (filtre ``$lte``
      ChromaDB) — ici on fournit la VÉRIFICATION indépendante (defense in depth).
    - Protocole anti-contamination : inspiré des tests de contamination de
      données d'évaluation (Sainz et al. 2023) — sonder si le modèle régurgite
      les scores V-Party sans preuves.

CUSTOM (justifié) : les trois vérificateurs (~30 lignes chacun) sont des
protocoles propres au projet — aucune lib ne vérifie « pas de document
postérieur à l'élection » ni « le modèle récite-t-il V-Party ».
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import litellm
import structlog

from compass.config import settings
from compass.schemas import EvidenceItem

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- temporalité
class TemporalViolation(RuntimeError):
    """Une preuve postérieure à l'élection a atteint le raisonnement."""


def assert_temporal_integrity(evidence: list[EvidenceItem], as_of: date) -> None:
    """Vérification indépendante du contrôle temporel (defense in depth).

    C03 filtre déjà ; ce contrôle re-vérifie au dernier moment, juste avant
    le raisonnement — une fuite temporelle est une erreur FATALE, pas un warning.

    Raises:
        TemporalViolation: si une preuve est datée après ``as_of``.
    """
    leaks = [e.segment.segment_id for e in evidence if e.segment.meta.doc_date > as_of]
    if leaks:
        raise TemporalViolation(
            f"{len(leaks)} preuve(s) postérieure(s) au {as_of} : {leaks[:5]}"
        )


# ----------------------------------------------------------------------- traçabilité
class TraceLogger:
    """Journal d'exécution structuré — un fichier JSONL par cas traité.

    Enregistre : modèle, version, prompts, requêtes, documents (hash), date,
    décisions intermédiaires — la liste exacte du bloc 15.
    """

    def __init__(self, case_id: str) -> None:
        settings.ensure_dirs()
        self._path = settings.trace_dir / f"{case_id}_{datetime.utcnow():%Y%m%dT%H%M%S}.jsonl"
        structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                        structlog.processors.JSONRenderer()])
        self._log = structlog.get_logger()

    def record(self, step: str, **payload: Any) -> None:
        """Trace un événement d'étape (sérialisé JSON, append-only)."""
        entry = {"step": step, **{k: _serializable(v) for k, v in payload.items()}}
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @property
    def path(self) -> Path:
        return self._path


def _serializable(v: Any) -> Any:
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return v


# ----------------------------------------------------------------------- contamination
def contamination_probe(model_name: str, party_name: str, election_year: int,
                        variable_id: str) -> dict:
    """Sonde anti-récitation : le modèle connaît-il le score V-Party par cœur ?

    Protocole (A-2 de l'audit) : demander le score SANS aucune preuve. Si le
    modèle produit le score officiel avec assurance, le cas est marqué
    « contaminé » et doit être exclu (ou pondéré) dans la validation — la
    comparaison ne mesurerait que la mémorisation.

    Returns:
        {model, claims_knowledge, raw} — l'interprétation (comparaison au score
        officiel) se fait dans C14, où l'étalon est disponible.
    """
    prompt = (f"Sans aucun document, quel est le score V-Party '{variable_id}' "
              f"du parti {party_name} pour l'élection de {election_year} ? "
              "Réponds uniquement par un nombre, ou 'inconnu'.")
    resp = litellm.completion(model=model_name, temperature=0.0, max_tokens=10,
                              messages=[{"role": "user", "content": prompt}])
    raw = resp.choices[0].message.content.strip()
    knows = bool(re.fullmatch(r"-?\d+([.,]\d+)?", raw))
    logger.info("Sonde contamination %s/%s : %s", model_name, variable_id, raw)
    return {"model": model_name, "claims_knowledge": knows, "raw": raw}

