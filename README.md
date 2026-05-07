# ctranslate2-web-server

FastAPI server providing CTranslate2 inference with an OpenAI-compatible API. Supports all Gemma 3 models from HuggingFace, converting them to CTranslate2 format (int8) on first use.

## Requirements

- Docker
- NVIDIA Container Toolkit (GPU only)

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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/models` | Directory to store converted models |
| `DEVICE` | `cpu` / `cuda` | Inference device (set by image) |
