# Onyxia Runbook

This runbook validates the public repository against the real COMPASS architecture.

## 1. Clone

```bash
git clone https://github.com/Ensea-upb/compass-political-nlp.git
cd compass-political-nlp
```

## 2. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-demo.txt
```

The smoke test uses the lightweight demo dependencies only.

## 3. Smoke Test

```bash
pip install -r requirements-test.txt
python examples/run_real_architecture.py smoke
```

This checks the real schemas, registry, temporal guardrail and aggregation without downloading large models.

Expected ending:

```text
Smoke status: passed
```

## 4. Full Architecture Test

Install the complete research stack before the full architecture run:

```bash
pip install -r requirements-full.txt
```

The full run downloads NLP models on first execution. On Onyxia, prefer a service with enough memory for `sentence-transformers`, `transformers` and `torch`.

```bash
python examples/run_real_architecture.py full --reset
```

This seeds a synthetic country-party-election case, creates local SQLite and Chroma stores under `data/onyxia_real_architecture`, ingests synthetic text via C01, loads the V-Party registry, and calls the real `CompassRunner` on `v2pavote`.

Expected ending:

```text
Full status: completed
v2pavote: score=42.5, ...
```

## 5. If Something Fails

- Missing Python package in smoke/test mode: rerun `pip install -r requirements-test.txt`.
- Missing Python package in full mode: rerun `pip install -r requirements-full.txt`.
- Model download blocked: allow outbound internet or pre-populate the Hugging Face cache in the Onyxia service.
- Memory error: restart with more RAM.
- Tesseract error: the synthetic full example does not need OCR, so Tesseract is only required for PDF scans.
