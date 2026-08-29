#!/usr/bin/env python3
"""
nathwaniGPT v2.0 — API server

Weights:  weights/phi4-mini/          (microsoft/Phi-4-mini-instruct)
Engine:   engine/loader.py            (loads safetensors)
          engine/model.py             (transformer)
          engine/sampler.py           (sampling)
          engine/tokenizer.py         (HF tokenizer wrapper)
App:      serve/api.py                (FastAPI app, routes, streaming)
Config:   configs/phi4-mini.json      (inference defaults)

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions         (streaming + non-streaming)

Run:
    python models/v2.0/serve.py
    python models/v2.0/serve.py --port 9000
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import serve.api as api
import uvicorn

WEIGHTS = ROOT / "weights" / "phi4-mini"


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT v2.0 API server")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()

    api._weights_path = Path(args.weights)
    uvicorn.run(api.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
