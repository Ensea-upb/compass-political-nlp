"""Manifesto Project API helpers for public COMPASS ingestion workflows.

The official API root is https://manifesto-project.wzb.eu/api/v1/. Most
endpoints require an API key; this module reads it from MANIFESTO_API_KEY and
never stores credentials in code or generated outputs.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_API_ROOT = "https://manifesto-project.wzb.eu/api/v1"
DEFAULT_WEB_ROOT = "https://manifesto-project.wzb.eu"


class ManifestoAPIError(RuntimeError):
    """Raised when the Manifesto API cannot return the requested resource."""


@dataclass(frozen=True)
class ManifestoDocument:
    """Resolved Manifesto document metadata used by the ingestion script."""

    key: str
    metadata: dict[str, Any]
    pdf_url: str | None


class ManifestoAPI:
    """Small stdlib client for Manifesto Project API endpoints.

    Authentication follows the official documentation: an API key can be sent
    as request parameter, API_KEY header, or authorization header. We use the
    API_KEY header to avoid logging credentials in URLs.
    """

    def __init__(self, api_key: str | None = None, api_root: str = DEFAULT_API_ROOT) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("MANIFESTO_API_KEY")
        self.api_root = api_root.rstrip("/")

    def metadata(self, keys: list[str], version: str | None = None) -> list[dict[str, Any]]:
        """Return corpus metadata for Manifesto keys such as ``41320_200909``."""
        if not keys:
            return []
        params: list[tuple[str, str]] = []
        for key in keys:
            params.append(("keys[]", key))
        if version:
            params.append(("version", version))
        payload = self._request_json("metadata", params=params)
        return _coerce_items(payload)

    def resolve_documents(
        self,
        keys: list[str],
        *,
        version: str | None = None,
        pdf_field: str | None = None,
    ) -> list[ManifestoDocument]:
        """Fetch metadata and infer original PDF URLs when exposed by the API."""
        items = self.metadata(keys, version=version)
        resolved: list[ManifestoDocument] = []
        for requested_key, item in zip(keys, items):
            key = str(
                item.get("key")
                or item.get("manifesto_key")
                or item.get("manifesto_id")
                or requested_key
            )
            resolved.append(
                ManifestoDocument(
                    key=key,
                    metadata=item,
                    pdf_url=find_pdf_url(item, preferred_field=pdf_field),
                )
            )
        return resolved

    def download_pdf(self, url: str, destination: Path) -> Path:
        """Download a PDF or document URL to ``destination``."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        absolute_url = normalize_manifesto_url(url)
        req = urllib.request.Request(
            absolute_url,
            headers={
                "User-Agent": "compass-political-nlp/0.1",
                **self._auth_headers(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
        except Exception as exc:  # pragma: no cover - network-dependent
            raise ManifestoAPIError(f"Could not download PDF from {absolute_url}: {exc}") from exc
        destination.write_bytes(data)
        return destination

    def _request_json(self, endpoint: str, params: list[tuple[str, str]]) -> Any:
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        data = urllib.parse.urlencode(params).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "compass-political-nlp/0.1",
                **self._auth_headers(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - network-dependent
            raise ManifestoAPIError(f"Manifesto API request failed for {endpoint}: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:300].replace("\n", " ")
            raise ManifestoAPIError(f"Manifesto API returned non-JSON content: {preview}") from exc

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"API_KEY": self.api_key}


def normalize_manifesto_url(url: str) -> str:
    """Normalize relative Manifesto URLs into absolute URLs."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return urllib.parse.urljoin(DEFAULT_WEB_ROOT, url)
    return urllib.parse.urljoin(DEFAULT_WEB_ROOT + "/", url)


def find_pdf_url(payload: Any, preferred_field: str | None = None) -> str | None:
    """Best-effort extraction of an original PDF URL from metadata.

    The Manifesto metadata schema can evolve. This function first honors an
    explicit field supplied by the operator, then recursively scans for PDF-like
    links and document/download URL fields.
    """
    if preferred_field and isinstance(payload, dict):
        value = _lookup_dotted(payload, preferred_field)
        if isinstance(value, str) and value.strip():
            return normalize_manifesto_url(value.strip())

    candidate = _scan_for_pdf(payload)
    if candidate:
        return normalize_manifesto_url(candidate)

    candidate = _scan_for_document_url(payload)
    if candidate:
        return normalize_manifesto_url(candidate)
    return None


def _coerce_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "results", "metadata", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if all(isinstance(v, dict) for v in payload.values()):
            return list(payload.values())
        return [payload]
    return []


def _lookup_dotted(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _scan_for_pdf(value: Any) -> str | None:
    if isinstance(value, str):
        lower = value.lower()
        if ".pdf" in lower:
            return value
        return None
    if isinstance(value, dict):
        for child in value.values():
            found = _scan_for_pdf(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _scan_for_pdf(child)
            if found:
                return found
    return None


def _scan_for_document_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_l = str(key).lower()
            if isinstance(child, str) and child.strip():
                looks_like_url = child.startswith(("http://", "https://", "/"))
                looks_like_document_field = any(
                    token in key_l for token in ("pdf", "url", "download", "document", "file", "original")
                )
                if looks_like_url and looks_like_document_field:
                    return child
            found = _scan_for_document_url(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _scan_for_document_url(child)
            if found:
                return found
    return None