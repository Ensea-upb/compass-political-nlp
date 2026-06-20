# Guide d'exécution sur Onyxia

Ce guide décrit l'installation validée du dépôt public, depuis le clone jusqu'au chat connecté à un serveur vLLM local.

## 1. Préparer le service

Utiliser un service VS Code GPU avec :

- un GPU NVIDIA ;
- un volume persistant activé ;
- suffisamment de RAM pour les embeddings et le reranking CPU ;
- les ports `8000` pour vLLM et `41771` pour le chat si l'interface Onyxia exige leur exposition.

Le profil validé utilise un GPU de 16 Go et `Qwen/Qwen2.5-3B-Instruct`.

## 2. Cloner et créer l'environnement

```bash
cd ~/work
git clone https://github.com/Ensea-upb/compass-political-nlp.git
cd compass-political-nlp

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-onyxia.txt
pip install -r requirements-test.txt
```

Si le dépôt existe déjà :

```bash
cd ~/work/compass-political-nlp
git pull
source .venv/bin/activate
```

## 3. Vérifier le dépôt sans modèle

```bash
python examples/run_demo.py
python examples/run_real_architecture.py smoke
python -m pytest
```

Le smoke test vérifie les schémas, le registre, le garde-fou temporel, le diagnostic typé et l'agrégation sans exiger ChromaDB ou vLLM en fonctionnement.

## 4. Vérifier le GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Lancer vLLM

Dans un premier terminal :

```bash
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

Vérifier le serveur :

```bash
curl http://localhost:8000/v1/models
```

Le nom retourné doit être utilisé dans `COMPASS_JUDGE_MODELS` et `COMPASS_HYDE_MODEL`.

## 6. Configurer COMPASS

Dans un second terminal :

```bash
cd ~/work/compass-political-nlp
source .venv/bin/activate

export COMPASS_LLM_BACKEND=local
export COMPASS_LLM_API_BASE=http://localhost:8000/v1
export COMPASS_LLM_API_KEY=EMPTY
export COMPASS_JUDGE_MODELS=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_MODEL=Qwen/Qwen2.5-3B-Instruct
export COMPASS_HYDE_ENABLED=false
export COMPASS_HF_DEVICE=cpu
```

Le GPU est réservé à vLLM. Les embeddings et le cross-encoder restent sur CPU pour éviter une concurrence VRAM.

## 7. Configurer les données persistantes

```bash
export COMPASS_DATA_DIR=$PWD/data/manifesto_ingestion
export COMPASS_CHROMA_DIR=$PWD/data/manifesto_ingestion/chroma
export COMPASS_SQLITE_PATH=$PWD/data/manifesto_ingestion/compass_structured.db
export COMPASS_TRACE_DIR=$PWD/data/manifesto_ingestion/traces
```

Ces chemins doivent pointer vers la même campagne d'ingestion.

## 8. Lancer le chat

Le chat actuel exige encore un pays et une date limite :

```bash
python apps/chat_web.py \
  --country DEU \
  --as-of 2009-09-27 \
  --party 41320 \
  --port 41771
```

Ouvrir ensuite le port `41771` depuis l'interface Onyxia. Le chantier de chat mondial sans pays obligatoire est différé dans la roadmap.

Le chat exécute les questions longues en arrière-plan. L'appel `POST /ask`
retourne immédiatement un identifiant, puis l'interface interroge
`GET /result/<identifiant>` jusqu'à la fin du pipeline. Ce fonctionnement évite
qu'un proxy Onyxia interrompe la première question pendant le chargement CPU du
cross-encoder ou du modèle NLI. Le bouton affiche `Analyse...` pendant ce temps.

La première réponse reste généralement plus lente que les suivantes : les
modèles de reranking et de validation sont chargés en mémoire à leur première
utilisation.

## 9. Diagnostic rapide

```bash
curl http://localhost:8000/v1/models
git log -1 --oneline
python -m pytest tests/test_chat_engine.py tests/test_chat_web.py -q
```

- `ModuleNotFoundError: vllm` : installer `requirements-onyxia.txt` dans l'environnement actif.
- `CUDA out of memory` : utiliser le modèle 3B, réduire `gpu-memory-utilization` ou arrêter les autres processus GPU.
- `400 Bad Request` : vérifier `max-model-len`, le nom du modèle et que le dépôt contient le budget de prompt compact.
- `Gateway Timeout` avec une ancienne version du dépôt : faire `git pull`,
  redémarrer `apps/chat_web.py` et vérifier que l'interface appelle
  `./result/<identifiant>`. Le chat récent ne garde plus la requête `/ask`
  ouverte pendant toute l'analyse.
- fallback extractif : consulter la note technique et les logs du terminal vLLM.
