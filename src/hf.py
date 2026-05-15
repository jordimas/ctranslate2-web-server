import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoProcessor, AutoModelForCausalLM

_cache: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id not in _cache:
        logger.info("Loading model %s...", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        model.eval()
        _cache[model_id] = (model, processor)
        logger.info("Model %s loaded.", model_id)
    return _cache[model_id]


def _generate(model_id, inputs, max_tokens, temperature, top_p, stop):
    model, processor = _load(model_id)
    input_len = inputs["input_ids"].shape[-1]
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 1e-6 else 1.0,
            top_p=top_p,
            top_k=64,
            do_sample=temperature > 1e-6,
        )
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    text = processor.parse_response(raw)
    finish = "length"
    for s in (stop or []):
        if s in text:
            text, finish = text[:text.index(s)], "stop"
            break
    return text, finish


def chat(model_id, messages, max_tokens, temperature, top_p, stop):
    model, processor = _load(model_id)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    logger.info("Prompt: %s", prompt)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    return _generate(model_id, inputs, max_tokens, temperature, top_p, stop)


def main():
    model_id = "google/gemma-4-31b-it"
    messages = [
        {"role": "system", "content": "Translate the following English text to Catalan. Output only the translation."},
        {"role": "user", "content": "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes."},
    ]
    t0 = time.time()
    text, finish = chat(model_id, messages, max_tokens=512, temperature=0.0, top_p=1.0, stop=[])
    elapsed = time.time() - t0
    print(text)
    logger.info("Finish reason: %s | elapsed: %.2fs", finish, elapsed)


if __name__ == "__main__":
    main()
