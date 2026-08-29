#!/usr/bin/env python3
"""
nathwaniGPT —w

Usage:
    python serve/api.py
    python serve/api.py --weights weights/mistral-7b --port 8080

Endpoints:
    POST /v1/chat/completions   (streaming and non-streaming)
    GET  /v1/models
    GET  /health
"""

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.loader import load_model
from engine.sampler import sample_token
from engine.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------

_model = None
_tokenizer: Optional[Tokenizer] = None
_device: str = "cpu"
_weights_path: Path = Path("weights/phi4-mini")
_stop_ids: set[int] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device, _stop_ids
    print(f"Loading model from {_weights_path} …")
    _model, _, _device = load_model(_weights_path)
    _tokenizer = Tokenizer(_weights_path)
    _stop_ids = _tokenizer.stop_ids
    print(f"Stop token IDs: {_stop_ids}")
    print("API ready.")
    yield


app = FastAPI(title="nathwaniGPT API", version="2.0.0-alpha", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nathwaniGPT"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = False


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _prefill(messages: list[dict]) -> tuple[torch.Tensor, list, int]:
    prompt = _tokenizer.apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        add_generation_prompt=True,
    )
    input_ids = _tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    logits, kv_caches = _model(input_tensor, start_pos=0)
    return logits, kv_caches, len(input_ids)


def _step(logits, kv_caches, start_pos, req: ChatRequest) -> tuple[int, torch.Tensor, list, int]:
    next_token = sample_token(
        logits[0, -1],
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=_device)
    logits, kv_caches = _model(input_tensor, kv_caches=kv_caches, start_pos=start_pos)
    return next_token, logits, kv_caches, start_pos + 1


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": "nathwaniGPT", "object": "model"}],
        "object": "list",
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream(messages, req),
            media_type="text/event-stream",
        )

    # Non-streaming
    logits, kv_caches, start_pos = _prefill(messages)
    result: list[int] = []

    for _ in range(req.max_tokens):
        next_token, logits, kv_caches, start_pos = _step(logits, kv_caches, start_pos, req)
        if next_token in _stop_ids:
            break
        result.append(next_token)

    text = _tokenizer.decode(result)
    return {
        "id": "chatcmpl-nathwani",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


async def _stream(messages: list[dict], req: ChatRequest) -> AsyncIterator[bytes]:
    logits, kv_caches, start_pos = await asyncio.get_event_loop().run_in_executor(
        None, _prefill, messages
    )

    for _ in range(req.max_tokens):
        next_token, logits, kv_caches, start_pos = _step(logits, kv_caches, start_pos, req)
        if next_token in _stop_ids:
            break

        token_str = _tokenizer.decode_token(next_token)
        chunk = {
            "choices": [{
                "delta": {"content": token_str},
                "index": 0,
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0)

    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/phi4-mini")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    _weights_path = Path(args.weights)
    uvicorn.run(app, host=args.host, port=args.port)
