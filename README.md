# ctranslate2-web-server

FastAPI server providing CTranslate2 inference with an OpenAI-compatible API. Supports all Gemma 3 models from HuggingFace, converting them to CTranslate2 format (int8) on first use.

## Requirements

- Docker
- NVIDIA Container Toolkit (GPU only)

## Dockerfiles

| File | Base | Use case |
|---|---|---|
| `Dockerfile.cpu` | `python:3.14-slim` | Standard CPU image |
| `Dockerfile.gpu` | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` | NVIDIA GPU image |

## Build

```bash
make build-cpu   # CPU image
make build-gpu   # GPU image
make build       # both
```

To bake one or more models into the image at build time, pass `BUILD_MODELS` as a space-separated list of HuggingFace model IDs:

```bash
make build-cpu BUILD_MODELS="google/gemma-3-270m-it"
make build-cpu BUILD_MODELS="google/gemma-3-270m-it google/gemma-3-4b-it"
```

The models are downloaded, converted to CTranslate2 int8 format, and stored inside the image under `/models`. When the container starts those models are available immediately with no conversion step on first request.

You can also pass the build arg directly to Docker:

```bash
docker build -f Dockerfile.cpu \
  --build-arg BUILD_MODELS="google/gemma-3-270m-it" \
  -t ctranslate2-web-server-cpu .
```

> **Note:** `HF_TOKEN` must be set in the environment if the models require authentication (e.g. gated models). Pass it with `--build-arg HF_TOKEN=$HF_TOKEN`.

## Run

```bash
make run-cpu     # CPU
make run-gpu     # GPU (requires NVIDIA runtime)
```

Models are stored inside the image under `/models`.

## Pre-built images

Images are published to the GitHub Container Registry on every push:

```
ghcr.io/jordimas/ctranslate2-web-server-cpu:latest
ghcr.io/jordimas/ctranslate2-web-server-gpu:latest
```

Pull and run the CPU image:

```bash
docker pull ghcr.io/jordimas/ctranslate2-web-server-cpu:latest
docker run --rm -p 8015:8015 -e HF_TOKEN=$HF_TOKEN ghcr.io/jordimas/ctranslate2-web-server-cpu:latest
```

For GPU:

```bash
docker pull ghcr.io/jordimas/ctranslate2-web-server-gpu:latest
docker run --rm --gpus all -p 8015:8015 -e HF_TOKEN=$HF_TOKEN ghcr.io/jordimas/ctranslate2-web-server-gpu:latest
```

## API

### List models

```
GET /v1/models
```

Returns all Gemma 3 models available on HuggingFace.

### Text completion

```bash
curl http://localhost:8015/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "google/gemma-3-270m-it", "prompt": "Once upon a time", "max_tokens": 100}'
```

### Chat completion

```bash
curl http://localhost:8015/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "google/gemma-3-270m-it", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The first request for a model triggers an automatic download and conversion. Subsequent requests use the cached converted model.

## Sample

`sample/eval_flores_en_ca.py` evaluates English→Catalan translation on [FLORES-200](https://huggingface.co/datasets/facebook/flores) scored with BLEU. It uses the OpenAI Python SDK — switching to this server requires only a `--url` flag:

```bash
# Start the server
make run-cpu

# Against OpenAI
python sample/eval_flores_en_ca.py --model gpt-4o-mini

# Against this server
python sample/eval_flores_en_ca.py \
  --url http://localhost:8015/v1 \
  --model google/gemma-3-4b-it
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/models` | Directory to store converted models |
| `DEVICE` | `cpu` / `cuda` | Inference device (set by image) |
