# Modèles Hugging Face sur Onyxia

COMPASS utilise des modèles open-weight servis localement par une API compatible OpenAI. Aucune clé OpenAI, Anthropic ou Mistral n'est nécessaire en mode local.

## Profils

### Profil opérationnel 16 Go

```bash
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_ENABLED=false
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HF_DEVICE=cpu
```

Le GPU sert vLLM. Les embeddings, le NLI et le reranking utilisent le CPU.

### Profil de recherche multi-GPU

```bash
export COMPASS_JUDGE_MODELS="Qwen/Qwen3-32B,mistralai/Mistral-Small-3.1-24B-Instruct-2503,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export COMPASS_HYDE_ENABLED=true
export COMPASS_HYDE_MODEL=Qwen/Qwen3-14B
export COMPASS_VISION_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
```

Ce profil n'est pas le défaut validé sur GPU 16 Go.

## Inspecter les téléchargements

```bash
python scripts/download_onyxia_models.py --all --dry-run
```

Le script affiche le dépôt Hugging Face, la destination, la taille approximative, la licence et le statut d'accès. `--dry-run` ne télécharge rien.

Sélectionner une famille :

```bash
python scripts/download_onyxia_models.py --judges --dry-run
python scripts/download_onyxia_models.py --hyde --dry-run
python scripts/download_onyxia_models.py --vision --dry-run
```

## Télécharger

```bash
export HF_MODELS_DIR=$HOME/.cache/huggingface/hub
python scripts/download_onyxia_models.py --judges
```

Pour un modèle public, `HF_TOKEN` est facultatif mais augmente les limites de téléchargement :

```bash
export HF_TOKEN="votre_token"
```

Ne jamais inscrire ce token dans le dépôt.

## Lancer vLLM sur un GPU

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

Sur plusieurs GPU :

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

`--tensor-parallel-size` doit correspondre au nombre de GPU visibles.

## Vérifier l'API

```bash
curl http://localhost:8000/v1/models
```

Test minimal :

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Réponds seulement: OK"}],
    "max_tokens": 10,
    "temperature": 0
  }'
```

## Configurer COMPASS

```bash
export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
```

## Ressources approximatives

| Modèle | Poids approximatifs | Usage conseillé |
| --- | ---: | --- |
| Qwen2.5-3B-Instruct | 6 à 8 Go | GPU 16 Go, chat opérationnel |
| Qwen3-14B | environ 28 Go | GPU plus grand ou quantification |
| Mistral Small 24B | plusieurs dizaines de Go | multi-GPU |
| Qwen/DeepSeek 32B | environ 60 Go en pleine précision | multi-GPU |

La taille du dépôt n'est pas la VRAM exacte : vLLM réserve aussi le cache KV et les structures d'exécution.

## Erreurs fréquentes

- `No module named vllm` : mauvais environnement ou dépendance absente.
- `CUDA out of memory` : modèle trop grand, longueur de contexte excessive ou GPU déjà occupé.
- `model not found` : la variable COMPASS ne correspond pas au nom exposé par `/v1/models`.
- `400 Bad Request` : vérifier le contexte maximal et `max_tokens`.
