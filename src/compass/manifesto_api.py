"""Thin client for the official Manifesto Project API.

Docs: https://manifesto-project.wzb.eu/information/documents/api

The client stays close to the documented workflow:
get_core -> party_date keys -> metadata -> manifesto_id -> texts_and_annotations.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_API_ROOT = "https://manifesto-project.wzb.eu/api/v1"
DEFAULT_WEB_ROOT = "https://manifesto-project.wzb.eu"
PUBLIC_ENDPOINTS = {
    "list_core_versions",
    "list_metadata_versions",
    "get_core_codebook",
    "get_core_citation",
    "get_corpus_citation",
}


class ManifestoAPIError(RuntimeError):
    """Raised when the Manifesto API cannot return the requested resource."""


@dataclass(frozen=True)
class ManifestoDocument:
    key: str
    metadata: dict[str, Any]
    pdf_url: str | None


@dataclass(frozen=True)
class ManifestoText:
    key: str
    text: str
    payload: dict[str, Any]


class ManifestoAPI:
    """Small stdlib wrapper around documented Manifesto API endpoints."""

    def __init__(self, api_key: str | None = None, api_root: str = DEFAULT_API_ROOT) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("MANIFESTO_API_KEY")
        self.api_root = api_root.rstrip("/")

    def list_core_versions(self, *, kind: str | None = None) -> Any:
        return self._get("list_core_versions", [("kind", kind)] if kind else [])

    def list_metadata_versions(self, *, tag: bool | None = None, details: bool | None = None) -> Any:
        params = []
        if tag is not None:
            params.append(("tag", _bool(tag)))
        if details is not None:
            params.append(("details", _bool(details)))
        return self._get("list_metadata_versions", params)

    def get_core_payload(self, version: str, kind: str = "dta", *, raw: bool = False) -> Any:
        params = [("key", version), ("kind", kind)] + ([("raw", "true")] if raw else [])
        return self._get("get_core", params)

    def get_core_records(self, version: str, kind: str = "dta") -> list[dict[str, Any]]:
        payload = self.get_core_payload(version, kind=kind)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            return _records_from_base64_file(payload["content"], kind)
        return []

    def get_parties(self, version: str, *, list_form: str = "short", raw: bool = False) -> Any:
        params = [("key", version), ("list_form", list_form)] + ([("raw", "true")] if raw else [])
        return self._get("get_parties", params)

    def metadata_payload(self, keys: list[str], version: str | None = None) -> dict[str, Any]:
        if not keys:
            return {"items": [], "missing_items": []}
        params = _key_params(keys) + ([("version", version)] if version else [])
        return _payload_with_items(self._post("metadata", params))

    def metadata(self, keys: list[str], version: str | None = None) -> list[dict[str, Any]]:
        return self.metadata_payload(keys, version).get("items", [])

    def texts_and_annotations_payload(
        self,
        keys: list[str],
        *,
        version: str | None = None,
        translation: str | None = None,
    ) -> dict[str, Any]:
        if not keys:
            return {"items": [], "missing_items": []}
        params = _key_params(keys)
        if version:
            params.append(("version", version))
        if translation:
            params.append(("translation", translation))
        return _payload_with_items(self._post("texts_and_annotations", params))

    def texts_and_annotations(
        self,
        keys: list[str],
        *,
        version: str | None = None,
        translation: str | None = None,
    ) -> list[ManifestoText]:
        payload = self.texts_and_annotations_payload(keys, version=version, translation=translation)
        texts = []
        for item in payload.get("items", []):
            text = extract_manifesto_text(item)
            if text:
                texts.append(ManifestoText(str(item.get("manifesto_id") or item.get("key") or ""), text, item))
        return texts

    def resolve_documents(
        self,
        keys: list[str],
        *,
        version: str | None = None,
        pdf_field: str | None = None,
    ) -> list[ManifestoDocument]:
        docs = []
        for item in self.metadata(keys, version=version):
            key = str(item.get("manifesto_id") or item.get("key") or item.get("manifesto_key") or "")
            docs.append(ManifestoDocument(key, item, find_pdf_url(item, preferred_field=pdf_field)))
        return docs

    def download_pdf(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = normalize_manifesto_url(url)
        try:
            data = self._download_bytes(url)
        except urllib.error.HTTPError as exc:  # pragma: no cover
            if exc.code not in (401, 403) or not self.api_key:
                raise ManifestoAPIError(f"Could not download PDF from {url}: {exc}") from exc
            data = self._download_bytes(_append_query_param(url, "api_key", self.api_key))
        if not data.lstrip().startswith(b"%PDF"):
            raise ManifestoAPIError(f"Manifesto original document endpoint did not return a PDF: {url}")
        destination.write_bytes(data)
        return destination

    def _download_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "compass-political-nlp/0.1", "Accept": "application/pdf,*/*", **self.auth_headers()},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read()

    def _get(self, endpoint: str, params: list[tuple[str, str]]) -> Any:
        return self._request_json(endpoint, params, method="GET")

    def _post(self, endpoint: str, params: list[tuple[str, str]]) -> Any:
        return self._request_json(endpoint, params, method="POST")

    def _request_json(self, endpoint: str, params: list[tuple[str, str]], *, method: str) -> Any:
        if endpoint not in PUBLIC_ENDPOINTS and not self.api_key:
            raise ManifestoAPIError(f"Manifesto API endpoint {endpoint} requires MANIFESTO_API_KEY.")
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        data = urllib.parse.urlencode(params).encode("utf-8")
        if method == "GET":
            url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
            data = None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                **({"Content-Type": "application/x-www-form-urlencoded"} if data else {}),
                "Accept": "application/json",
                "User-Agent": "compass-political-nlp/0.1",
                **self.auth_headers(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover
            raise ManifestoAPIError(f"Manifesto API request failed for {endpoint}: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestoAPIError(f"Manifesto API returned non-JSON content: {raw[:300]!r}") from exc

    def auth_headers(self) -> dict[str, str]:
        return {"API_KEY": self.api_key, "Authorization": f"Token {self.api_key}"} if self.api_key else {}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _key_params(keys: list[str]) -> list[tuple[str, str]]:
    return [("keys[]", key) for key in keys]


def _payload_with_items(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"items": [], "missing_items": []}
    items = payload.get("items")
    missing = payload.get("missing_items")
    return {
        "items": [item for item in items if isinstance(item, dict)] if isinstance(items, list) else [],
        "missing_items": missing if isinstance(missing, list) else [],
    }


def normalize_manifesto_url(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return url
    return urllib.parse.urljoin(DEFAULT_WEB_ROOT + "/", url.lstrip("/"))


def find_pdf_url(payload: dict[str, Any], preferred_field: str | None = None) -> str | None:
    candidates = []
    if preferred_field:
        candidates.append(_lookup_dotted(payload, preferred_field))
    candidates.extend(
        payload.get(field)
        for field in ("url_original", "pdf_url", "document_url", "download_url", "original_url", "url")
    )
    links = payload.get("links")
    if isinstance(links, dict):
        candidates.extend(links.get(field) for field in ("pdf", "original", "original_pdf", "download"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return normalize_manifesto_url(candidate.strip())
    return None


def extract_manifesto_text(payload: dict[str, Any]) -> str:
    chunks = []
    for item in payload.get("items", []):
        if isinstance(item, dict):
            chunks.extend(str(item[field]).strip() for field in ("text", "sentence", "translated_text", "original_text") if isinstance(item.get(field), str) and item[field].strip())
    if not chunks:
        chunks.extend(str(payload[field]).strip() for field in ("text", "content", "manifesto_text") if isinstance(payload.get(field), str) and payload[field].strip())
    return "\n".join(chunks)


def _lookup_dotted(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) if k != key]
    query.append((key, value))
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _records_from_base64_file(content: str, kind: str) -> list[dict[str, Any]]:
    raw = base64.b64decode(content)
    if kind.lower() in {"csv", "txt"}:
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    frame = _read_table(raw, kind)
    return frame.fillna("").astype(str).to_dict(orient="records")


def _read_table(raw: bytes, kind: str):
    import pandas as pd

    suffix = ".xlsx" if kind.lower() in {"xlsx", "xls"} else ".dta"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as fh:
        fh.write(raw)
        fh.flush()
        return pd.read_excel(fh.name) if suffix == ".xlsx" else pd.read_stata(fh.name, convert_categoricals=False)


def debug_core_decoding(content: str, kind: str = "") -> dict[str, Any]:
    raw = base64.b64decode(content)
    summary: dict[str, Any] = {"decoded_bytes": len(raw), "kind": kind}
    try:
        frame = _read_table(raw, kind or "dta")
        summary.update({"read_ok": True, "rows": int(len(frame)), "columns": [str(col) for col in list(frame.columns)[:60]]})
    except Exception as exc:
        summary.update({"read_ok": False, "error": str(exc)})
    return summary
