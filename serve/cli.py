#!/usr/bin/env python3
"""
nathwaniGPT — interactive chat CLI

Usage:
    python serve/cli.py
    python serve/cli.py --weights weights/mistral-7b --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.loader import load_model
from engine.sampler import sample_token
from engine.tokenizer import Tokenizer
from engine.context import ConversationContext

SYSTEM_PROMPT = (
    "You are nathwaniGPT, a sharp and highly capable AI assistant. "
    "You think carefully before responding — when a problem is complex, "
    "you reason through it step by step before giving your answer. "
    "When the answer is simple, you give it directly without theatrics. "
    "You never pad responses, add unnecessary caveats, or repeat yourself. "
    "When you are uncertain, you say so plainly. "
    "When you disagree with a premise, you say so and explain why. "
    "You treat the user as an intelligent adult."
)

BANNER = """
╔═══════════════════════════════╗
║       nathwaniGPT  v2.0       ║
║   your model. your weights.   ║
╚═══════════════════════════════╝
Type your message. Ctrl+C or Ctrl+D to exit.
"""


def generate(
    model,
    tokenizer: Tokenizer,
    messages: list[dict],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt)
    # Only track tokens the model generated — not prompt tokens.
    # Including prompt tokens in the repetition penalty penalises every common
    # word in the system prompt, producing incoherent output.
    generated_ids: list[int] = []
    result_tokens: list[int] = []

    stop_ids = tokenizer.stop_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, kv_caches = model(input_tensor, start_pos=0)
    start_pos = len(input_ids)

    for _ in range(max_new_tokens):
        next_token = sample_token(
            logits[0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated=generated_ids,
        )
        generated_ids.append(next_token)

        if next_token in stop_ids:
            break

        result_tokens.append(next_token)
        token_str = tokenizer.decode_token(next_token)
        print(token_str, end="", flush=True)

        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, kv_caches = model(input_tensor, kv_caches=kv_caches, start_pos=start_pos)
        start_pos += 1

    print()
    return tokenizer.decode(result_tokens)


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT chat")
    parser.add_argument("--weights", default="weights/phi4-mini", help="Path to weights directory")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-limit", type=int, default=16384,
                        help="Token budget for conversation history (default: 16384)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    print(f"\nLoading nathwaniGPT from {weights_path} …")
    model, cfg, device = load_model(weights_path)
    tokenizer = Tokenizer(weights_path)
    print(BANNER)

    ctx = ConversationContext(tokenizer, max_tokens=args.context_limit, system_prompt=SYSTEM_PROMPT)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("/reset", "/clear"):
            ctx.reset()
            print("(conversation cleared)")
            continue

        if not user_input:
            continue

        ctx.add("user", user_input)
        print("nathwaniGPT: ", end="", flush=True)

        response = generate(
            model, tokenizer, ctx.messages, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
        )
        ctx.add("assistant", response)
        print(f"  [{ctx.token_count}/{args.context_limit} tokens]")


if __name__ == "__main__":
    main()
