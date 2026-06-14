"""Explication d'une trace — l'anti-boîte-noire en production.

AMÉLIORATION (2026-06-12, demande Soro) : le deck pédagogique
``COMPASS_calculs_pas_a_pas.pptx`` montre les calculs d'UN cas développé à la
main ; cet utilitaire fait la même chose AUTOMATIQUEMENT pour chaque cas réel,
en convertissant la trace JSONL produite par C15 en rapport markdown lisible —
étape par étape, avec les nombres, les verdicts et les preuves.

Usage :
    python explain_trace.py data/traces/CIV_PARTIA_civ2020pres_20260612T...jsonl
    -> écrit un .md à côté de la trace.

CUSTOM (assumé) : pure mise en forme — aucune logique de décision ici. Si une
information manque dans le rapport, c'est qu'elle manque dans la trace : la
corriger dans C15/orchestrator, pas ici (la trace reste la source de vérité).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_STEP_TITLES = {
    "case_start": "Cas soumis",
    "dossier": "Constitution du dossier (C04)",
    "sheet": "Fiche du registre (C05)",
    "sufficiency": "Test de suffisance (C07)",
    "active_search": "Recherche active (C08)",
    "diagnosis": "Diagnostic des preuves (C09)",
    "panel": "Panel de juges (C11)",
    "final": "Réponse finale (C13)",
    "case_end": "Clôture du cas",
}


def explain(trace_path: Path) -> Path:
    """Convertit une trace JSONL en rapport markdown pas-à-pas.

    Args:
        trace_path: fichier produit par TraceLogger (C15).

    Returns:
        Chemin du rapport markdown écrit à côté de la trace.
    """
    lines: list[str] = [f"# Rapport d'explication — {trace_path.stem}", ""]
    current_var = None
    with open(trace_path, encoding="utf-8") as fh:
        for raw in fh:
            entry = json.loads(raw)
            step = entry.pop("step", "?")
            var = entry.get("variable")
            if var and var != current_var:
                current_var = var
                lines += [f"\n## Variable `{var}`", ""]
            title = _STEP_TITLES.get(step, step)
            lines.append(f"### {title}")
            for key, value in entry.items():
                if key == "variable":
                    continue
                if isinstance(value, list) and value and isinstance(value[0], str):
                    lines.append(f"- **{key}** :")
                    lines += [f"    - {v}" for v in value]
                else:
                    lines.append(f"- **{key}** : {_fmt(value)}")
            lines.append("")
    out = trace_path.with_suffix(".explained.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False)[:300] + "`"
    return str(value)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python explain_trace.py <trace.jsonl>")
    report = explain(Path(sys.argv[1]))
    print(f"Rapport écrit : {report}")

