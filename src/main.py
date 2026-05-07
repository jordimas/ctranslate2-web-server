import logging
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ctranslate2
import transformers
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODELS_DIR.exists():
        logger.info("Models directory %s does not exist; no cached models.", MODELS_DIR)
    else:
        cached = [d.name for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "model.bin").exists()]
        if cached:
            logger.info("Cached models available (%d): %s", len(cached), ", ".join(sorted(cached)))
        else:
            logger.info("No cached models found in %s", MODELS_DIR)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(
        "422 Unprocessable Content %s %s\nBody: %s\nErrors: %s",
        request.method, request.url.path,
        body.decode("utf-8", errors="replace"),
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
DEVICE = os.environ.get("DEVICE", "cpu")
_lock = threading.Lock()
_cache: dict[str, tuple] = {}

cpu_count = os.cpu_count()  # logical cores

# General-purpose default
inter_threads = max(1, cpu_count // 4)
intra_threads = 4


# --- model loading ---

def _load(model_id: str):
    with _lock:
        if model_id not in _cache:
            ct2 = MODELS_DIR / model_id.replace("/", "-")
            if not (ct2 / "model.bin").exists():
                ct2.mkdir(parents=True, exist_ok=True)
                logger.info("Converting model %s to CTranslate2 format...", model_id)
                r = subprocess.run(
                    ["ct2-transformers-converter", "--model", model_id,
                     "--output_dir", str(ct2), "--quantization", "int8", "--force"],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr)
                logger.info("Conversion of model %s complete.", model_id)
            _cache[model_id] = (
                ctranslate2.Generator(str(ct2), device=DEVICE, inter_threads=inter_threads, intra_threads=intra_threads),
                AutoTokenizer.from_pretrained(model_id),
            )
    return _cache[model_id]


def _generate(model_id, token_ids, max_tokens, temperature, top_p, stop):
    gen, tok = _load(model_id)
    tokens = tok.convert_ids_to_tokens(token_ids)
    result = gen.generate_batch(
        [tokens], max_length=max_tokens,
        sampling_temperature=max(temperature, 1e-6),
        sampling_topp=top_p,
        include_prompt_in_result=False,
    )
    text = tok.decode(result[0].sequences_ids[0], skip_special_tokens=True)
    finish = "length"
    for s in (stop or []):
        if s in text:
            text, finish = text[:text.index(s)], "stop"
            break
    return text, finish


def _ntokens(model_id, text):
    _, tok = _load(model_id)
    return len(tok.encode(text))


# --- schemas ---

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "google"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class CompletionReq(BaseModel):
    model: str
    prompt: Union[str, list[str]]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResp(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ChatMsg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str
    messages: list[ChatMsg]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    reasoning: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMsg
    finish_reason: Optional[str] = None


class ChatResp(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage


# --- routes ---

@app.get("/info")
def info():
    return {"ctranslate2": ctranslate2.__version__, "transformers": transformers.__version__}


@app.get("/v1/models", response_model=ModelList)
@app.post("/v1/models", response_model=ModelList)
def list_models():
    logger.info("INPUT  GET /v1/models (no body)")
    models = HfApi().list_models(author="google", search="gemma-3", cardData=False)
    data = sorted(
        [ModelCard(id=m.id) for m in models if "gemma-3" in m.id.lower()],
        key=lambda x: x.id,
    )
    resp = ModelList(data=data)
    logger.info("OUTPUT /v1/models -> %d models", len(resp.data))
    return resp


@app.post("/v1/completions", response_model=CompletionResp)
@app.get("/v1/completions", response_model=CompletionResp)
def completions(req: CompletionReq):
    logger.info("INPUT  POST /v1/completions -> %s", req.model_dump_json())
    if req.stream:
        raise HTTPException(501, "Streaming not supported")
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    stop = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    choices, pt, ct = [], 0, 0
    for i, p in enumerate(prompts):
        _, tok = _load(req.model)
        token_ids = tok.encode(p)
        text, finish = _generate(req.model, token_ids, req.max_tokens, req.temperature, req.top_p, stop)
        pt += len(token_ids)
        ct += _ntokens(req.model, text)
        choices.append(CompletionChoice(text=text, index=i, finish_reason=finish))
    resp = CompletionResp(
        model=req.model, choices=choices,
        usage=Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
    )
    logger.info("OUTPUT /v1/completions -> %s", resp.model_dump_json())
    return resp


@app.post("/v1/chat/completions", response_model=ChatResp)
@app.get("/v1/chat/completions", response_model=ChatResp)
def chat_completions(req: ChatReq):
    logger.info("INPUT  POST /v1/chat/completions -> %s", req.model_dump_json())
    if req.stream:
        raise HTTPException(501, "Streaming not supported")
    stop = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    _, tok = _load(req.model)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    is_qwen = "qwen" in req.model.lower()
    disable_thinking = is_qwen and req.reasoning not in ("low", "medium", "high")
    if tok.chat_template is not None:
        extra = {"enable_thinking": False} if disable_thinking else {}
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        logger.info("Prompt: -> %s", prompt)
        token_ids = tok.encode(prompt, add_special_tokens=False)
    else:
        parts = [f"{m.role}: {m.content}" for m in req.messages]
        parts.append("assistant:")
        token_ids = tok.encode("\n".join(parts))
    text, finish = _generate(req.model, token_ids, req.max_tokens, req.temperature, req.top_p, stop)
    resp = ChatResp(
        model=req.model,
        choices=[ChatChoice(index=0, message=ChatMsg(role="assistant", content=text), finish_reason=finish)],
        usage=Usage(
            prompt_tokens=len(token_ids),
            completion_tokens=_ntokens(req.model, text),
            total_tokens=len(token_ids) + _ntokens(req.model, text),
        ),
    )
    logger.info("OUTPUT /v1/chat/completions -> %s", resp.model_dump_json())
    return resp
