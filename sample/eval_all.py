"""
Run eval_flores_en_ca.py for all gemma-3-it models and report BLEU scores + timing.
"""
import subprocess
import time
import re
import os

_models_file = os.path.join(os.path.dirname(__file__), "eval_models.txt")
MODELS = [line.strip() for line in open(_models_file) if line.strip()]

URL = "http://0.0.0.0:8015/v1"

results = []
server_info = None

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Running: {model}")
    print('='*60, flush=True)

    env = {**os.environ, "OPENAI_API_KEY": "NO_KEY"}
    start = time.time()
    is_last = model == MODELS[-1]
    cmd = ["uv", "run", "eval_flores_en_ca.py", "--url", URL, "--model", model]
    if is_last:
        cmd.append("--info")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    proc.wait()
    elapsed = time.time() - start

    full_output = "".join(output_lines)
    bleu_match = re.search(r"BLEU\s*=\s*([\d.]+)", full_output)
    bleu = bleu_match.group(1) if bleu_match else "N/A"
    info_match = re.search(r"ctranslate2 .+", full_output)
    if info_match:
        server_info = info_match.group(0)
    results.append((model, bleu, elapsed))

print(f"\n{'='*60}")
print(f"{'Model':<30} {'BLEU':>8} {'Time':>10}")
print('-'*60)
for model, bleu, t in results:
    mins, secs = divmod(int(t), 60)
    print(f"{model:<30} {bleu:>8} {mins:>6}m{secs:02d}s")
print('='*60)
if server_info:
    print(server_info)
