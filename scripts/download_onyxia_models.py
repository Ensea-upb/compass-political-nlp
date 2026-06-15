"""Download COMPASS open-weight models for an Onyxia service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compass.config import LLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download every configured model")
    parser.add_argument("--judges", action="store_true", help="Download judge models")
    parser.add_argument("--hyde", action="store_true", help="Download the HyDE model")
    parser.add_argument("--vision", action="store_true", help="Download the vision model")
    parser.add_argument("--dry-run", action="store_true", help="Show downloads without downloading")
    args = parser.parse_args()

    cfg = LLMConfig()
    selected = _selected_models(cfg, args)
    if not selected:
        parser.error("Select at least one of --all, --judges, --hyde, --vision")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    root = Path(os.environ.get("HF_MODELS_DIR", "~/.cache/huggingface/hub")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    for model_id in selected:
        destination = root / model_id.replace("/", "--")
        info = _model_info(api, model_id)
        print(f"\nModel: {model_id}")
        print(f"Destination: {destination}")
        print(f"Approx size: {info['size']}")
        print(f"License: {info['license']}")
        print(f"Auth/gated: {info['gated']}")
        if args.dry_run:
            continue
        snapshot_download(
            repo_id=model_id,
            local_dir=destination,
            token=token,
            resume_download=True,
        )


def _selected_models(cfg: LLMConfig, args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.all or args.judges:
        models.extend(cfg.judge_models)
    if args.all or args.hyde:
        models.append(cfg.hyde_model)
    if args.all or args.vision:
        models.append(cfg.vision_model)
    deduped: list[str] = []
    for model in models:
        if not model:
            continue
        if model not in deduped:
            deduped.append(model)
    return deduped


def _model_info(api: HfApi, model_id: str) -> dict[str, str]:
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return {
            "size": f"unknown ({exc})",
            "license": "unknown",
            "gated": "unknown",
        }

    size = sum((s.size or 0) for s in (info.siblings or []))
    license_name = "unknown"
    card_data = getattr(info, "cardData", None)
    if isinstance(card_data, dict):
        license_name = str(card_data.get("license") or "unknown")
    elif card_data is not None:
        license_name = str(getattr(card_data, "license", "unknown") or "unknown")

    return {
        "size": _format_bytes(size) if size else "unknown",
        "license": license_name,
        "gated": str(getattr(info, "gated", "unknown")),
    }


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


if __name__ == "__main__":
    main()
