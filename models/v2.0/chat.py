#!/usr/bin/env python3
"""
nathwaniGPT v2.0 — interactive chat

Weights:  weights/phi4-mini/          (microsoft/Phi-4-mini-instruct)
Engine:   engine/loader.py            (loads safetensors)
          engine/model.py             (transformer)
          engine/sampler.py           (sampling)
          engine/tokenizer.py         (HF tokenizer wrapper)
Chat:     serve/cli.py                (generate loop + SYSTEM_PROMPT)
Config:   configs/phi4-mini.json      (inference defaults)

Run:
    python models/v2.0/chat.py
    python models/v2.0/chat.py --max-tokens 2048 --temperature 0.5
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine.loader import load_model
from engine.tokenizer import Tokenizer
from serve.cli import BANNER, SYSTEM_PROMPT, generate

WEIGHTS = ROOT / "weights" / "phi4-mini"
_cfg = json.loads((ROOT / "configs" / "phi4-mini.json").read_text())
DEFAULTS = _cfg["nathwanigpt_defaults"]


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT v2.0 chat")
    parser.add_argument("--weights",      default=str(WEIGHTS))
    parser.add_argument("--max-tokens",   type=int,   default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature",  type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--top-k",        type=int,   default=DEFAULTS["top_k"])
    parser.add_argument("--top-p",        type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--rep-penalty",  type=float, default=DEFAULTS["repetition_penalty"])
    args = parser.parse_args()

    weights_path = Path(args.weights)
    print(f"\nLoading nathwaniGPT v2.0 from {weights_path} …")
    model, _, device = load_model(weights_path)
    tokenizer = Tokenizer(weights_path)
    print(BANNER)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        print("nathwaniGPT: ", end="", flush=True)

        response = generate(
            model, tokenizer, messages, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
        )
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
