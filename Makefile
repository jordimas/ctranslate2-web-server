IMAGE        ?= ctranslate2-web-server
TAG          ?= latest
PORT         ?= 8015
BUILD_MODELS ?=

_build_args = $(if $(BUILD_MODELS),--build-arg BUILD_MODELS="$(BUILD_MODELS)")

build-cpu:
	docker build -f Dockerfile.cpu $(_build_args) -t $(IMAGE)-cpu:$(TAG) .

build-gpu:
	docker build -f Dockerfile.gpu $(_build_args) -t $(IMAGE)-gpu:$(TAG) .

build: build-cpu build-gpu

build-cpu-12b:
	docker build -f Dockerfile.cpu --build-arg BUILD_MODELS="google/gemma-3-12b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-cpu:$(TAG) .

build-cpu-1b:
	docker build -f Dockerfile.cpu --build-arg BUILD_MODELS="google/gemma-3-1b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-cpu:$(TAG) .

build-cpu-27b:
	docker build -f Dockerfile.cpu --build-arg BUILD_MODELS="google/gemma-3-27b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-cpu:$(TAG) .

build-gpu-12b:
	docker build --no-cache -f Dockerfile.gpu --build-arg BUILD_MODELS="google/gemma-3-12b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-gpu:$(TAG) .

build-gpu-1b:
	docker build -f Dockerfile.gpu --build-arg BUILD_MODELS="google/gemma-3-1b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-gpu:$(TAG) .

build-gpu-4b:
	docker build -f Dockerfile.gpu --build-arg BUILD_MODELS="google/gemma-3-4b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-gpu:$(TAG) .

build-cpu-4b:
	docker build -f Dockerfile.cpu --build-arg BUILD_MODELS="google/gemma-3-4b-it" --build-arg HF_TOKEN=$(HF_TOKEN) -t $(IMAGE)-gpu:$(TAG) .

build-cpu-eval-all:
	docker build --no-cache -f Dockerfile.cpu \
		--build-arg BUILD_MODELS="$(shell tr '\n' ' ' < sample/eval_models.txt)" \
		--build-arg HF_TOKEN=$(HF_TOKEN) \
		-t $(IMAGE)-cpu:$(TAG) .

run-cpu:
	docker run --rm -p $(PORT):$(PORT) -e HF_TOKEN=$(HF_TOKEN) $(IMAGE)-cpu:$(TAG)

run-gpu:
	docker run --rm --gpus all -p $(PORT):$(PORT) -e HF_TOKEN=$(HF_TOKEN) $(IMAGE)-gpu:$(TAG)

.PHONY: build build-cpu build-gpu build-cpu-12b build-cpu-1b build-gpu-12b build-gpu-1b run-cpu run-gpu build-cpu-eval-all
