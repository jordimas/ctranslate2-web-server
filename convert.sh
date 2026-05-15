#!/bin/bash
set -e
uv run ct2-transformers-converter --low_cpu_mem_usage --model google/gemma-4-31b-it --output_dir "${MODELS_DIR:-/models}/google-gemma-4-31b-it" --quantization int8 --force
