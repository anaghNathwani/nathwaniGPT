#!/usr/bin/env python3
"""
nathwaniGPT — quality evaluation

Runs the model against a JSONL test set and reports:
  - Exact match accuracy
  - Contains match accuracy  (expected string appears anywhere in output)
  - Mean generation length
  - Optionally: perplexity (NLL over expected completions)

Engine:  engine/loader.py, engine/model.py, engine/sampler.py, engine/tokenizer.py
Weights: weights/phi4-mini/ (default)

Dataset format (one JSON object per line):
  {"prompt": "What is 2+2?", "expected": "4"}
  {"messages": [{"role": "user", "content": "..."}], "expected": "..."}

Usage:
    python scripts/eval.py --dataset evals/test.jsonl
    python scripts/eval.py --dataset evals/test.jsonl --perplexity
    python scripts/eval.py --dataset evals/test.jsonl --weights weights/mistral-7b --out results.jsonl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.loader import load_model
from engine.sampler import sample_token
from engine.tokenizer import Tokenizer

DEFAULT_WEIGHTS = ROOT / "weights" / "phi4-mini"
DEFAULT_SYSTEM  = "You are a helpful assistant. Answer concisely and accurately."


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_messages(row: dict) -> list[dict]:
    if "messages" in row:
        return row["messages"]
    return [
        {"role": "system",  "content": DEFAULT_SYSTEM},
        {"role": "user",    "content": row["prompt"]},
    ]


def _generate(model, tokenizer, messages, device, args) -> str:
    import torch

    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids   = tokenizer.encode(prompt_text)
    stop_ids    = tokenizer.stop_ids

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
            generated=generated,
        )
        generated.append(next_token)
        if next_token in stop_ids:
            break
        result.append(next_token)
        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, kv_caches = model(input_tensor, kv_caches=kv_caches, start_pos=start_pos)
        start_pos += 1

    return tokenizer.decode(result)


def _perplexity(model, tokenizer, messages, expected: str, device) -> float:
    """Compute the mean NLL of the expected completion given the prompt."""
    import torch

    prompt_text   = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_ids    = tokenizer.encode(prompt_text)
    expected_ids  = tokenizer.encode(expected, add_special_tokens=False)

    if not expected_ids:
        return float("nan")

    full_ids = prompt_ids + expected_ids
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(input_tensor, start_pos=0)

    import torch.nn.functional as F
    log_probs = F.log_softmax(logits[0], dim=-1)
    nll = 0.0
    for i, tok in enumerate(expected_ids):
        pos = len(prompt_ids) - 1 + i
        nll -= log_probs[pos, tok].item()

    return math.exp(nll / len(expected_ids))


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT quality eval")
    parser.add_argument("--dataset",     required=True,           help="JSONL test file")
    parser.add_argument("--weights",     default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--max-tokens",  type=int,   default=256)
    parser.add_argument("--temperature", type=float, default=0.0,  help="0 = greedy (default for eval)")
    parser.add_argument("--top-k",       type=int,   default=1)
    parser.add_argument("--top-p",       type=float, default=1.0)
    parser.add_argument("--perplexity",  action="store_true",     help="Also compute perplexity per example")
    parser.add_argument("--out",         default=None,            help="Write per-example results to JSONL")
    parser.add_argument("--limit",       type=int,   default=None, help="Only evaluate first N examples")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    rows = _load_dataset(dataset_path)
    if args.limit:
        rows = rows[: args.limit]

    print(f"\nLoading model from {args.weights} …")
    model, _, device = load_model(Path(args.weights))
    tokenizer = Tokenizer(Path(args.weights))
    print(f"Evaluating {len(rows)} examples on {device}\n")

    out_file = open(args.out, "w") if args.out else None
    exact_matches    = 0
    contains_matches = 0
    total_length     = 0
    perplexities     = []
    t_start          = time.perf_counter()

    for i, row in enumerate(rows):
        expected = str(row.get("expected", ""))
        messages = _build_messages(row)

        output = _generate(model, tokenizer, messages, device, args)
        total_length += len(output.split())

        exact   = output.strip().lower() == expected.strip().lower()
        contains = expected.strip().lower() in output.lower()
        exact_matches    += int(exact)
        contains_matches += int(contains)

        ppl = None
        if args.perplexity:
            ppl = _perplexity(model, tokenizer, messages, expected, device)
            perplexities.append(ppl)

        result = {
            "i":        i,
            "prompt":   row.get("prompt", ""),
            "expected": expected,
            "output":   output,
            "exact":    exact,
            "contains": contains,
        }
        if ppl is not None:
            result["perplexity"] = round(ppl, 3)

        if out_file:
            out_file.write(json.dumps(result) + "\n")

        status = "✓" if exact else ("~" if contains else "✗")
        ppl_str = f"  ppl={ppl:.1f}" if ppl is not None else ""
        print(f"  [{i+1:>4}/{len(rows)}] {status}  {output[:60]!r}{ppl_str}")

    elapsed = time.perf_counter() - t_start
    n = len(rows)
    sep = "─" * 48

    print(f"\n{sep}")
    print(f"  Results on {dataset_path.name}  ({n} examples, {elapsed:.1f}s)")
    print(sep)
    print(f"  Exact match    : {exact_matches}/{n}  ({100*exact_matches/n:.1f}%)")
    print(f"  Contains match : {contains_matches}/{n}  ({100*contains_matches/n:.1f}%)")
    print(f"  Avg output len : {total_length/n:.1f} words")
    if perplexities:
        print(f"  Mean perplexity: {sum(perplexities)/len(perplexities):.2f}")
    print(sep)

    if out_file:
        out_file.close()
        print(f"\n  Per-example results saved to: {args.out}")


if __name__ == "__main__":
    main()
