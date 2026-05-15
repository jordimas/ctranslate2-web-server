import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ctranslate2
import transformers
from transformers import AutoTokenizer

logger.info("ctranslate2 %s | transformers %s", ctranslate2.__version__, transformers.__version__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
DEVICE = os.environ.get("DEVICE", "cpu")

_cache: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id not in _cache:
        ct2 = MODELS_DIR / model_id.replace("/", "-")
        if not (ct2 / "model.bin").exists():
            raise RuntimeError(f"Model {model_id} not found at {ct2}. Convert it first.")
        logger.info("Loading model %s...", model_id)
        cpu_count = os.cpu_count()
        ct2_kwargs = {} if DEVICE == "cuda" else {"inter_threads": max(1, cpu_count // 4), "intra_threads": 4}
        _cache[model_id] = (
            ctranslate2.Generator(str(ct2), device=DEVICE, compute_type="int8", **ct2_kwargs),
            AutoTokenizer.from_pretrained(model_id),
        )
        logger.info("Model %s loaded.", model_id)
    return _cache[model_id]


def _end_tokens(tok):
    candidates = ["<eos>", "<end_of_turn>", "<turn|>", "<|end_of_turn|>", "<|eot_id|>"]
    added = tok.get_added_vocab()
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in candidates:
        if t in added:
            ids.add(added[t])
    return [tok.convert_ids_to_tokens([i])[0] for i in ids if i is not None]


def translate(model_id, messages, max_tokens=512):
    gen, tok = _load(model_id)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    logger.info("Prompt: %s", prompt)
    token_ids = tok.encode(prompt, add_special_tokens=False)
    result = gen.generate_batch(
        [tok.convert_ids_to_tokens(token_ids)],
        max_length=max_tokens,
        sampling_temperature=1e-6,
        sampling_topp=1.0,
        include_prompt_in_result=False,
        #end_token=_end_tokens(tok),
    )
    return tok.decode(result[0].sequences_ids[0], skip_special_tokens=True)


def main():
    model_id = "google/gemma-4-31b-it"
    messages = [
        {
            "role": "system",
            "content": "Translate the following English text to Catalan. Output only the translation.",
        },
        {
            "role": "user",
            "content": "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.",
        },
    ]
    t0 = time.time()
    text = translate(model_id, messages)
    elapsed = time.time() - t0
    print(text)
    logger.info("Elapsed: %.2fs", elapsed)
    _, tok = _load(model_id)
    print(f"BOS token: {tok.bos_token!r} (id={tok.bos_token_id})")


if __name__ == "__main__":
    main()