#!/usr/bin/env python3
"""
nathwaniGPT v2.0 — single-shot generation (non-interactive)

Weights:  weights/phi4-mini/
Engine:   engine/loader.py, engine/model.py, engine/sampler.py, engine/tokenizer.py
Config:   configs/phi4-mini.json

Useful for scripting and piping — prints the response and exits.

Usage:
    python models/v2.0/generate.py "What is the Riemann hypothesis?"
    python models/v2.0/generate.py --system "Reply only in haiku." "Explain recursion"
    echo "Summarise the theory of relativity" | python models/v2.0/generate.py -
    python models/v2.0/generate.py "hello" --temperature 0 --max-tokens 50
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine.loader import load_model
from engine.sampler import sample_token
from engine.tokenizer import Tokenizer

WEIGHTS = ROOT / "weights" / "phi4-mini"
_cfg = json.loads((ROOT / "configs" / "phi4-mini.json").read_text())
DEFAULTS = _cfg["nathwanigpt_defaults"]

DEFAULT_SYSTEM = (
    "You are nathwaniGPT, a sharp and highly capable AI assistant. "
    "You think carefully before responding. You never pad responses or repeat yourself. "
    "You treat the user as an intelligent adult."
)


def generate(model, tokenizer, system: str, prompt: str, device: str, args) -> str:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = tokenizer.encode(text)

    stop_ids = tokenizer.stop_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, kv_caches = model(input_tensor, start_pos=0)
    start_pos = len(input_ids)

    result: list[int] = []
    generated: list[int] = []

    for _ in range(args.max_tokens):
        next_token = sample_token(
            logits[0, -1],
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            generated=generated,
        )
        generated.append(next_token)
        if next_token in stop_ids:
            break
        result.append(next_token)

        if args.stream:
            print(tokenizer.decode_token(next_token), end="", flush=True)

        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, kv_caches = model(input_tensor, kv_caches=kv_caches, start_pos=start_pos)
        start_pos += 1

    if args.stream:
        print()

    return tokenizer.decode(result)


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT v2.0 single-shot generation")
    parser.add_argument("prompt", nargs="?", default="-",
                        help="Prompt text, or '-' to read from stdin")
    parser.add_argument("--system",      default=DEFAULT_SYSTEM)
    parser.add_argument("--weights",     default=str(WEIGHTS))
    parser.add_argument("--max-tokens",  type=int,   default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--top-k",       type=int,   default=DEFAULTS["top_k"])
    parser.add_argument("--top-p",       type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--rep-penalty", type=float, default=DEFAULTS["repetition_penalty"])
    parser.add_argument("--stream",      action="store_true",
                        help="Stream tokens as they are generated")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress loading messages")
    args = parser.parse_args()

    prompt = sys.stdin.read().strip() if args.prompt == "-" else args.prompt
    if not prompt:
        parser.error("No prompt provided.")

    if not args.quiet:
        print(f"[nathwaniGPT v2.0] loading from {args.weights} …", file=sys.stderr)

    weights_path = Path(args.weights)
    model, _, device = load_model(weights_path)
    tokenizer = Tokenizer(weights_path)

    response = generate(model, tokenizer, args.system, prompt, device, args)

    if not args.stream:
        print(response)


if __name__ == "__main__":
    main()
