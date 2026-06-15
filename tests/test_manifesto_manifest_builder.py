from datetime import date

from scripts.build_manifesto_manifest import build_manifest_rows, doc_date_from_manifesto_date


def test_doc_date_from_manifesto_date_month():
    assert doc_date_from_manifesto_date("200909") == date(2009, 9, 1)


def test_build_manifest_rows_filters_country_and_election():
    records = [
        {"country": "41", "party": "41320", "date": "200909"},
        {"country": "42", "party": "99999", "date": "200909"},
    ]

    rows = build_manifest_rows(
        records,
        metadata_version="2024-1",
        country_iso3="DEU",
        country_code="41",
        election_date="200909",
        language="de",
    )

    assert rows == [
        {
            "key": "41320_200909",
            "metadata_version": "2024-1",
            "country_iso3": "DEU",
            "party_id": "41320",
            "election_id": "DEU_2009",
            "doc_date": "2009-09-01",
            "doc_type": "manifesto",
            "language": "de",
            "reliability": "official",
            "pdf_url": "",
        }
    ]