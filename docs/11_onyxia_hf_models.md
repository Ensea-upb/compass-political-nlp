# Onyxia Hugging Face Models

COMPASS defaults to open-weight Hugging Face models served through a local
OpenAI-compatible endpoint. This avoids requiring OpenAI, Anthropic or Mistral
API keys for the research pipeline.

## 1. Download Models

Install the full dependencies:

```bash
pip install -r requirements-full.txt
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

Download the vision model:

```bash
python scripts/download_onyxia_models.py --vision
```

If a model requires authentication, set `HF_TOKEN` in the environment. The
script calls `huggingface_hub.login()` when `HF_TOKEN` is available. No token is
stored in the repository.

## 2. Serve With vLLM

Example for one judge model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768
```

For a single large Onyxia service, start one model per service or change the
served model between runs. COMPASS expects an OpenAI-compatible `/v1` endpoint.

## 3. Run COMPASS in Local Mode

```bash
export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_HYDE_ENABLED=true

python examples/run_real_architecture.py full --reset
```

Override judge models with a comma-separated list:

```bash
export COMPASS_JUDGE_MODELS="Qwen/Qwen3-32B,mistralai/Mistral-Small-3.1-24B-Instruct-2503,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

## 4. Approximate Onyxia Resources

These figures are practical starting points, not hard guarantees:

| Model | Role | Suggested GPU profile |
| --- | --- | --- |
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

The scientific rule is unchanged: document any substitution and rerun the pilot
ablations before treating scores as research outputs.

## Small-GPU Onyxia profile: NVIDIA A2 16 GB

If the service exposes a 16 GB GPU, reserve the GPU for vLLM and run the
COMPASS-side Hugging Face components on CPU:

```bash
# Terminal 1: vLLM on GPU
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

```bash
# Terminal 2: COMPASS on CPU for HF classifiers/re-rankers, HTTP to vLLM
export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu

python examples/run_real_architecture.py full --reset
```

This avoids loading the NLI classifier, reranker, or embeddings on the same GPU
that already hosts vLLM.
