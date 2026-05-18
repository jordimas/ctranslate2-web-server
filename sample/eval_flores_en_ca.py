"""
Eval: English -> Catalan translation using OpenAI, scored with BLEU on FLORES-200.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datasets import load_dataset
from openai import OpenAI
from sacrebleu.metrics import BLEU

parser = argparse.ArgumentParser()
parser.add_argument("--url", default=None, help="OpenAI-compatible base URL")
parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
parser.add_argument("--workers", type=int, default=1, help="Concurrent requests")
parser.add_argument("--info", action="store_true", help="Show server library versions")
args = parser.parse_args()

if args.info and args.url:
    try:
        base = args.url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = requests.get(f"{base}/info")
        if resp.status_code != 404:
            data = resp.json()
            print(f"ctranslate2 {data['ctranslate2']}, transformers {data['transformers']}")
    except Exception:
        pass

N = 200
client = OpenAI(base_url=args.url)

ds = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
samples = ds.select(range(N))

def translate(i, text):
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Translate the following English text to Catalan. Output only the translation."},
            {"role": "user", "content": text},
        ],
        temperature=0,
        extra_body={"reasoning_effort": "none"},
    )
    tokens = response.usage.completion_tokens if response.usage else 0
    return i, response.choices[0].message.content.strip(), tokens

hypotheses = [None] * N
completed = 0
total_tokens = 0
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(translate, i, row["sentence_eng_Latn"]): i for i, row in enumerate(samples)}
    for future in as_completed(futures):
        i, translation, tok_count = future.result()
        hypotheses[i] = translation
        total_tokens += tok_count
        print(f"{i} - {translation}")
        completed += 1
        if completed % (N // 10) == 0:
            print(f"{completed * 100 // N}%")

references = [row["sentence_cat_Latn"] for row in samples]

bleu = BLEU()
result = bleu.corpus_score(hypotheses, [references])
print(result)
print(f"Tokens: {total_tokens}")
