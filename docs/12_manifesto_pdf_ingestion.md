# Manifesto Project PDF ingestion

COMPASS follows the official Manifesto Project API workflow: build `party_date` keys from the core dataset, query `metadata`, use the returned `manifesto_id` for `texts_and_annotations`, and only download original PDFs when metadata exposes a downloadable document URL. If the original PDF endpoint is blocked with HTTP 401/403, the script automatically falls back to the official `texts_and_annotations` endpoint and ingests the machine-readable manifesto text.


## Official API workflow

The relevant official endpoints are under `https://manifesto-project.wzb.eu/api/v1/`:

1. `list_core_versions` / `get_core` identify core dataset releases and party-election rows.
2. `metadata` accepts `keys[]` values such as `41320_200909` and returns corpus metadata plus a `manifesto_id` such as `41320_2009`.
3. `texts_and_annotations` should normally be called with the returned `manifesto_id`; it can also request an optional `translation`.
4. Original PDFs are treated as a best-effort document source when metadata exposes a URL. Some `/down/originals/...pdf` links can still return HTTP 403 even with a valid API key, so the text fallback is part of the normal ingestion design.

The client sends protected API requests with the `API_KEY` header from `MANIFESTO_API_KEY` and uses POST for `metadata` and `texts_and_annotations`, as recommended for parameter-heavy requests.
## 1. Prerequisites

Create a Manifesto Project account, generate an API key, then expose it only as an environment variable:

```bash
export MANIFESTO_API_KEY="your_manifesto_api_key"
```

The key is never stored in the repository. Downloaded PDFs are written under `data/manifesto_pdfs/`, and `data/` is ignored by Git.

## 2. One PDF from API metadata

```bash
export COMPASS_HF_DEVICE=cpu
python examples/run_manifesto_pdf_ingestion.py \
  --keys 41320_200909 \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --party-id 41320 \
  --election-id DEU_2009 \
  --doc-date 2009-09-01 \
  --dry-run \
  --print-metadata
```

First run with `--dry-run`. Manifesto metadata usually exposes original documents through `url_original`. If your metadata version uses another field, inspect the printed metadata and rerun with the field containing the original PDF URL:

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --keys 41320_200909 \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --party-id 41320 \
  --election-id DEU_2009 \
  --doc-date 2009-09-01 \
  --pdf-field links.original_pdf
```

The exact field name can vary with the metadata schema, but the client intentionally checks only explicit URL fields such as `url_original` instead of guessing from arbitrary nested values.

## 3. Batch ingestion from a CSV manifest

Use a CSV file with at least:

```text
key,country_iso3,doc_date
```

Recommended columns:

```text
key,metadata_version,country_iso3,party_id,election_id,doc_date,doc_type,language,reliability,pdf_url
```

Then run:

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --manifest examples/manifesto_pdf_manifest_sample.csv \
  --reset
```

If `pdf_url` is provided in the CSV, the script downloads it directly. If it is empty, the script resolves the URL through the Manifesto API metadata endpoint.


## 4. Build the CSV automatically

For a quick validated batch, start from:

```bash
python examples/run_manifesto_pdf_ingestion.py \
  --manifest examples/manifesto_manifest_deu_2009.csv \
  --translation en
```

To generate a manifest from the Manifesto core dataset:

```bash
python scripts/build_manifesto_manifest.py \
  --core-version MPDS2024a \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --country-code 41 \
  --election-date 200909 \
  --language de \
  --output data/manifests/deu_2009.csv
```

You can also start from a local core CSV:

```bash
python scripts/build_manifesto_manifest.py \
  --core-csv data/mp_core.csv \
  --metadata-version 2024-1 \
  --country-iso3 DEU \
  --country-code 41 \
  --election-date 200909 \
  --language de \
  --output data/manifests/deu_2009.csv
```
## 5. PDF blocked: automatic text fallback

Some original document URLs such as `/down/originals/...pdf` can return HTTP 403 even when the metadata API accepts your key. In that case the script continues automatically with `texts_and_annotations` unless `--no-text-fallback` is passed.

The fallback writes text files under `data/manifesto_texts/` and indexes them with `DocumentPipeline.ingest_text()`. Use `--translation en` if you want the API-provided English translation when available.

## 6. What the script does

```text
Manifesto core dataset or CSV manifest
-> party_date key, for example 41320_200909
-> Manifesto API metadata
-> manifesto_id, for example 41320_2009
-> original PDF download when exposed, otherwise texts_and_annotations
-> DocumentPipeline.ingest_pdf() or DocumentPipeline.ingest_text()
-> CountryMemory.add_documents()
-> outputs/manifesto_pdf_ingestion_report.json
```

This is the real COMPASS ingestion path: PyMuPDF extracts text PDFs, OCR is attempted for scanned pages, metadata is converted into `DocumentMeta`, then parent/child segments are indexed in ChromaDB.

## 7. Onyxia notes

Onyxia should keep persistence enabled because the Manifesto corpus and embedding index can grow quickly. Use at least a persistent volume large enough for the PDFs plus `data/manifesto_ingestion/chroma/`.

For a first test on the validated small GPU profile, keep embeddings on CPU:

```bash
export COMPASS_HF_DEVICE=cpu
```

The PDF ingestion step does not require the local vLLM server. vLLM is needed later for reasoning or judge-panel steps.
