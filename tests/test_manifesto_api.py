from compass.manifesto_api import ManifestoAPI, ManifestoAPIError, _append_query_param, _records_from_text_payload, extract_manifesto_text, find_pdf_url, normalize_manifesto_url


def test_find_pdf_url_prefers_explicit_field():
    payload = {"links": {"original": "/documents/demo.pdf"}, "other": "https://example.org/nope"}

    assert find_pdf_url(payload, preferred_field="links.original") == "https://manifesto-project.wzb.eu/documents/demo.pdf"


def test_find_pdf_url_scans_nested_pdf_links():
    payload = {"metadata": {"files": [{"href": "https://example.org/manifesto.pdf?download=1"}]}}

    assert find_pdf_url(payload) == "https://example.org/manifesto.pdf?download=1"


def test_find_pdf_url_scans_document_url_fields():
    payload = {"document_url": "/api/documents/123/original"}

    assert find_pdf_url(payload) == "https://manifesto-project.wzb.eu/api/documents/123/original"


def test_normalize_manifesto_url_keeps_absolute_url():
    assert normalize_manifesto_url("https://example.org/a.pdf") == "https://example.org/a.pdf"

def test_append_query_param_replaces_existing_value():
    url = _append_query_param("https://example.org/a.pdf?api_key=old&x=1", "api_key", "new")

    assert url == "https://example.org/a.pdf?x=1&api_key=new"

def test_extract_manifesto_text_from_annotations():
    payload = {"items": [{"sentence": "First sentence."}, {"text": "Second sentence."}]}

    assert extract_manifesto_text(payload) == "First sentence.\nSecond sentence."

def test_records_from_text_payload_returns_empty_on_malformed_csv():
    raw = b"party,date\r\n41320,200909\r\nmalformed\rline,still_bad"

    assert _records_from_text_payload(raw) == []

def test_metadata_uses_official_post_keys_and_preserves_missing_items(monkeypatch):
    calls = []
    api = ManifestoAPI(api_key="test-key")

    def fake_request(endpoint, params, method="POST", **kwargs):
        calls.append((endpoint, params, method, kwargs))
        return {
            "items": [{"manifesto_id": "41320_2009", "party": 41320}],
            "missing_items": ["99999_200909"],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)

    payload = api.metadata_payload(["41320_200909", "99999_200909"], version="2024-1")

    assert payload["items"][0]["manifesto_id"] == "41320_2009"
    assert payload["missing_items"] == ["99999_200909"]
    assert calls == [
        (
            "metadata",
            [("keys[]", "41320_200909"), ("keys[]", "99999_200909"), ("version", "2024-1")],
            "POST",
            {},
        )
    ]


def test_texts_and_annotations_uses_manifesto_ids_and_translation(monkeypatch):
    calls = []
    api = ManifestoAPI(api_key="test-key")

    def fake_request(endpoint, params, method="POST", **kwargs):
        calls.append((endpoint, params, method, kwargs))
        return {
            "items": [{"manifesto_id": "41320_2009", "items": [{"text": "A sentence."}]}],
            "missing_items": [],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)

    texts = api.texts_and_annotations(["41320_2009"], version="2024-1", translation="en")

    assert texts[0].key == "41320_2009"
    assert texts[0].text == "A sentence."
    assert calls == [
        (
            "texts_and_annotations",
            [("keys[]", "41320_2009"), ("version", "2024-1"), ("translation", "en")],
            "POST",
            {},
        )
    ]


def test_public_version_endpoints_do_not_require_api_key(monkeypatch):
    calls = []
    api = ManifestoAPI(api_key=None)

    def fake_request(endpoint, params, method="POST", **kwargs):
        calls.append((endpoint, params, method, kwargs))
        return {"ok": True}

    monkeypatch.setattr(api, "_request_json", fake_request)

    api.list_core_versions()
    api.list_metadata_versions(tag=True, details=True)

    assert calls[0] == ("list_core_versions", [], "GET", {"require_api_key": False})
    assert calls[1] == (
        "list_metadata_versions",
        [("tag", "true"), ("details", "true")],
        "GET",
        {"require_api_key": False},
    )


def test_protected_endpoint_requires_api_key_before_network():
    api = ManifestoAPI(api_key=None)

    try:
        api.metadata_payload(["41320_200909"], version="2024-1")
    except ManifestoAPIError as exc:
        assert "MANIFESTO_API_KEY" in str(exc)
    else:
        raise AssertionError("metadata endpoint should require an API key")


def test_resolve_documents_uses_manifesto_id_without_zip_shift(monkeypatch):
    api = ManifestoAPI(api_key="test-key")

    def fake_metadata(keys, version=None):
        return [{"manifesto_id": "41320_2009", "document_url": "/down/originals/41320_2009.pdf"}]

    monkeypatch.setattr(api, "metadata", fake_metadata)

    docs = api.resolve_documents(["41320_200909", "missing_key"], version="2024-1")

    assert len(docs) == 1
    assert docs[0].key == "41320_2009"
    assert docs[0].pdf_url == "https://manifesto-project.wzb.eu/down/originals/41320_2009.pdf"