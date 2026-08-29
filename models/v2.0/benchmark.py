#!/usr/bin/env python3
"""
nathwaniGPT v2.0 — speed and memory benchmark

Weights:  weights/phi4-mini/
Engine:   engine/loader.py, engine/model.py, engine/sampler.py, engine/tokenizer.py
Config:   configs/phi4-mini.json

Measures:
  - Model load time
  - Prefill latency  (time to produce the first token)
  - Decode throughput (tokens/sec for subsequent tokens)
  - Peak memory usage

Usage:
    python models/v2.0/benchmark.py
    python models/v2.0/benchmark.py --decode-tokens 100 --runs 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "weights" / "phi4-mini"
_cfg = json.loads((ROOT / "configs" / "phi4-mini.json").read_text())
DEFAULTS = _cfg["nathwanigpt_defaults"]

BENCH_PROMPT = (
    "Explain the key differences between transformers and recurrent neural networks, "
    "including their trade-offs in terms of parallelism, memory, and sequence length handling."
)


def _peak_memory_mb(device: str) -> float:
    import torch
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def _sync(device: str):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_benchmark(args):
    import torch
    from engine.loader import load_model
    from engine.sampler import sample_token
    from engine.tokenizer import Tokenizer

    # ── Load ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, cfg, device = load_model(Path(args.weights))
    tokenizer = Tokenizer(Path(args.weights))
    load_time = time.perf_counter() - t0
    print(f"\n{'─'*52}")
    print(f"  nathwaniGPT v2.0 — benchmark")
    print(f"{'─'*52}")
    print(f"  Device : {device}")
    print(f"  Params : {model.n_params()/1e9:.2f}B")
    print(f"  Load   : {load_time:.2f}s")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": BENCH_PROMPT},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids   = tokenizer.encode(prompt_text)
    print(f"  Prompt : {len(input_ids)} tokens")
    print(f"  Decode : {args.decode_tokens} tokens × {args.runs} run(s)")
    print(f"{'─'*52}\n")

    prefill_times  = []
    decode_times   = []
    stop_ids       = tokenizer.stop_ids

    for run in range(args.runs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ── Prefill ───────────────────────────────────────────────────────────
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        _sync(device)
        t_pre = time.perf_counter()
        logits, kv_caches = model(input_tensor, start_pos=0)
        _sync(device)
        prefill_ms = (time.perf_counter() - t_pre) * 1000
        prefill_times.append(prefill_ms)

        # First token
        generated: list[int] = []
        next_token = sample_token(logits[0, -1], temperature=0.0)  # greedy for reproducibility
        generated.append(next_token)
        start_pos = len(input_ids)

        # ── Decode ────────────────────────────────────────────────────────────
        _sync(device)
        t_dec = time.perf_counter()
        tokens_generated = 1

        for _ in range(args.decode_tokens - 1):
            if next_token in stop_ids:
                break
            inp = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, kv_caches = model(inp, kv_caches=kv_caches, start_pos=start_pos)
            next_token = sample_token(logits[0, -1], temperature=0.0, generated=generated)
            generated.append(next_token)
            start_pos += 1
            tokens_generated += 1

        _sync(device)
        decode_elapsed = time.perf_counter() - t_dec
        decode_times.append(tokens_generated / decode_elapsed)

        peak_mb = _peak_memory_mb(device)
        print(f"  Run {run+1}: prefill {prefill_ms:.1f}ms | "
              f"decode {tokens_generated/decode_elapsed:.1f} tok/s"
              + (f" | peak {peak_mb:.0f}MB" if peak_mb else ""))

    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode  = sum(decode_times)  / len(decode_times)

    print(f"\n{'─'*52}")
    print(f"  Avg prefill latency : {avg_prefill:.1f} ms")
    print(f"  Avg decode speed    : {avg_decode:.1f} tok/s")
    if _peak_memory_mb(device):
        print(f"  Peak memory         : {_peak_memory_mb(device):.0f} MB")
    print(f"{'─'*52}\n")


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT v2.0 benchmark")
    parser.add_argument("--weights",       default=str(WEIGHTS))
    parser.add_argument("--decode-tokens", type=int, default=50,
                        help="Number of tokens to generate per decode run")
    parser.add_argument("--runs",          type=int, default=2,
                        help="Number of benchmark runs to average over")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
