from compass.manifesto_api import _append_query_param, extract_manifesto_text, find_pdf_url, normalize_manifesto_url


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