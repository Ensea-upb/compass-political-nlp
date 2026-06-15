# Manifesto Project PDF ingestion

COMPASS first tries to ingest original Manifesto Project PDFs when the API metadata exposes a downloadable document URL. If the original PDF endpoint is blocked with HTTP 401/403, the script automatically falls back to the official `texts_and_annotations` API and ingests the machine-readable manifesto text.

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

First run with `--dry-run`. If the report says `missing_pdf_url`, inspect the printed metadata and rerun with the field containing the original PDF URL:

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

The exact field name can vary with the metadata schema. The script also searches nested metadata automatically for PDF-like URLs.

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

## 4. PDF blocked: automatic text fallback

Some original document URLs such as `/down/originals/...pdf` can return HTTP 403 even when the metadata API accepts your key. In that case the script continues automatically with `texts_and_annotations` unless `--no-text-fallback` is passed.

The fallback writes text files under `data/manifesto_texts/` and indexes them with `DocumentPipeline.ingest_text()`. Use `--translation en` if you want the API-provided English translation when available.

## 5. What the script does

```text
Manifesto key or CSV manifest
-> Manifesto API metadata
-> original PDF download into data/manifesto_pdfs/
-> DocumentPipeline.ingest_pdf()
-> CountryMemory.add_documents()
-> outputs/manifesto_pdf_ingestion_report.json
```

This is the real COMPASS ingestion path: PyMuPDF extracts text PDFs, OCR is attempted for scanned pages, metadata is converted into `DocumentMeta`, then parent/child segments are indexed in ChromaDB.

## 6. Onyxia notes

Onyxia should keep persistence enabled because the Manifesto corpus and embedding index can grow quickly. Use at least a persistent volume large enough for the PDFs plus `data/manifesto_ingestion/chroma/`.

For a first test on the validated small GPU profile, keep embeddings on CPU:

```bash
export COMPASS_HF_DEVICE=cpu
```

The PDF ingestion step does not require the local vLLM server. vLLM is needed later for reasoning or judge-panel steps.