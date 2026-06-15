# Onyxia Hugging Face Models

COMPASS defaults to the validated `onyxia-16gb` open-weight profile served through a local OpenAI-compatible endpoint. This avoids requiring OpenAI, Anthropic, or Mistral API keys while staying runnable on the tested NVIDIA A2 16 GB service.

## Runtime Profiles

### `onyxia-16gb` operational profile

This is the default public configuration because it is the profile validated on Onyxia with an NVIDIA A2 16 GB GPU:

```bash
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_ENABLED=false
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu
```

The GPU is reserved for vLLM. Embeddings, NLI, and reranking run on CPU to avoid VRAM contention.

### `onyxia-large` research extension

Use this only on larger multi-GPU services:

```bash
export COMPASS_JUDGE_MODELS="Qwen/Qwen3-32B,mistralai/Mistral-Small-3.1-24B-Instruct-2503,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export COMPASS_HYDE_ENABLED=true
export COMPASS_HYDE_MODEL=Qwen/Qwen3-14B
export COMPASS_VISION_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
```

The large profile is a research target, not the operational default.

## 1. Download Models

Install the Onyxia runtime dependencies:

```bash
pip install -r requirements-onyxia.txt
```

Dry run:

```bash
python scripts/download_onyxia_models.py --all --dry-run
```

Download only the judge models:

```bash
export HF_MODELS_DIR=/home/onyxia/work/models
python scripts/download_onyxia_models.py --judges
```

Download the HyDE model:

```bash
python scripts/download_onyxia_models.py --hyde
```

Download the vision model only after setting `COMPASS_VISION_MODEL` for a large-profile run:

```bash
python scripts/download_onyxia_models.py --vision
```

If a model requires authentication, set `HF_TOKEN` in the environment. The script calls `huggingface_hub.login()` when `HF_TOKEN` is available. No token is stored in the repository.

## Validated Onyxia Installation

The following installation sequence was validated on an Onyxia `vscode-tensorflow-gpu` service with an NVIDIA A2 16 GB GPU and a 100 Gi persistent volume:

```bash
cd ~/work/compass-political-nlp
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-onyxia.txt
```

The Onyxia runtime file pins the FastAPI/Starlette/Prometheus stack used by vLLM. These pins avoid the server-side error:

```text
AttributeError: '_IncludedRouter' object has no attribute 'path'
```

After installation, verify the vLLM server with:

```bash
curl http://localhost:8000/v1/models
```

A successful response should list the served model, for example `Qwen/Qwen2.5-3B-Instruct`.

## 2. Serve With vLLM

Operational `onyxia-16gb` example:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --enforce-eager
```

Research `onyxia-large` example, only for high-memory multi-GPU services:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768
```

For a single large Onyxia service, start one model per service or change the served model between runs. COMPASS expects an OpenAI-compatible `/v1` endpoint.

## 3. Run COMPASS in Local Mode

Operational `onyxia-16gb` run:

```bash
export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_ENABLED=false
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu

python examples/run_real_architecture.py full --reset
```

Override judge models with a comma-separated list only for the research profile:

```bash
export COMPASS_JUDGE_MODELS="Qwen/Qwen3-32B,mistralai/Mistral-Small-3.1-24B-Instruct-2503,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export COMPASS_HYDE_ENABLED=true
export COMPASS_HYDE_MODEL=Qwen/Qwen3-14B
```

## 4. Approximate Onyxia Resources

These figures are practical starting points, not hard guarantees:

| Model | Role | Suggested GPU profile |
| --- | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | default judge and optional HyDE substitute | NVIDIA A2 16 GB with reduced context |
| `Qwen/Qwen3-14B` | HyDE | 1-2 high-memory GPUs, or quantized single GPU |
| `Qwen/Qwen3-32B` | judge | 2-4 high-memory GPUs |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | judge | 2-4 high-memory GPUs |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | judge | 2-4 high-memory GPUs |
| `Qwen/Qwen2.5-VL-32B-Instruct` | vision/OCR extension | 4 high-memory GPUs recommended |

Use vLLM tensor parallelism when the model does not fit on one GPU.

## 5. Substitute Lighter Models

If resources are limited, keep the same interface and override the model IDs:

```bash
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-7B-Instruct
export COMPASS_JUDGE_MODELS="Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
```

Document any substitution and rerun the pilot ablations before treating scores as research outputs.

## Troubleshooting vLLM 500 Errors

If vLLM returns HTTP 500 for both `/v1/chat/completions` and `/v1/completions`, check the server logs. If the traceback mentions `prometheus_fastapi_instrumentator` and `_IncludedRouter`, reinstall the tested runtime pins:

```bash
pip install --force-reinstall \
  "fastapi==0.115.14" \
  "starlette==0.46.2" \
  "prometheus-fastapi-instrumentator==7.1.0"

pip show fastapi starlette prometheus-fastapi-instrumentator
```

Then restart vLLM completely before rerunning COMPASS.