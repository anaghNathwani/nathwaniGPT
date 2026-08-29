#!/usr/bin/env python3
"""
nathwaniGPT v2.0 — model inspector

Weights:  weights/phi4-mini/          (reads config.json and safetensors index only)
Config:   configs/phi4-mini.json

Shows architecture, parameter counts, and memory estimates for each dtype
without loading the full model into memory.

Usage:
    python models/v2.0/inspect.py
    python models/v2.0/inspect.py --weights weights/phi4-mini
    python models/v2.0/inspect.py --full     # also load model and verify param count
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "weights" / "phi4-mini"

DTYPE_BYTES = {
    "float32 (fp32)": 4,
    "bfloat16 (bf16)": 2,
    "float16 (fp16)": 2,
    "int8":  1,
    "int4":  0.5,
}


def _count_params_from_index(weights_dir: Path) -> tuple[int, dict[str, int]]:
    """Parse safetensors index to count params without loading tensors."""
    index_path = weights_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file — read the safetensors header
        from safetensors import safe_open
        with safe_open(weights_dir / "model.safetensors", framework="pt") as f:
            names = f.keys()
            total = sum(1 for _ in names)  # rough; can't get shapes without loading
        return 0, {}

    with open(index_path) as f:
        index = json.load(f)

    # safetensors index only has the file mapping, not shapes — load metadata headers
    from safetensors import safe_open
    shapes: dict[str, tuple] = {}
    seen_shards: set[str] = set()
    for tensor_name, shard_file in index["weight_map"].items():
        if shard_file not in seen_shards:
            with safe_open(weights_dir / shard_file, framework="pt") as f:
                for k in f.keys():
                    shapes[k] = f.get_slice(k).get_shape()
            seen_shards.add(shard_file)

    breakdown: dict[str, int] = {}
    total = 0
    for name, shape in shapes.items():
        n = 1
        for d in shape:
            n *= d
        total += n

        # Bucket by layer type
        if "embed_tokens" in name or "lm_head" in name:
            bucket = "embedding / lm_head"
        elif "self_attn" in name:
            bucket = "attention"
        elif "mlp" in name:
            bucket = "MLP"
        elif "norm" in name:
            bucket = "layer norms"
        else:
            bucket = "other"
        breakdown[bucket] = breakdown.get(bucket, 0) + n

    return total, breakdown


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT v2.0 inspector")
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--full", action="store_true",
                        help="Load the full model to verify parameter count")
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    hf_cfg_path = weights_dir / "config.json"
    our_cfg_path = ROOT / "configs" / "phi4-mini.json"

    if not hf_cfg_path.exists():
        print(f"No config.json found at {weights_dir}. Run: python scripts/download.py phi4-mini")
        sys.exit(1)

    with open(hf_cfg_path) as f:
        hf = json.load(f)
    with open(our_cfg_path) as f:
        ours = json.load(f)

    hidden  = hf["hidden_size"]
    layers  = hf["num_hidden_layers"]
    heads   = hf["num_attention_heads"]
    kv_heads = hf.get("num_key_value_heads", heads)
    inter   = hf["intermediate_size"]
    vocab   = hf["vocab_size"]
    max_ctx = hf["max_position_embeddings"]
    head_dim = hidden // heads

    sep = "─" * 54

    print(f"\n{sep}")
    print(f"  nathwaniGPT v2.0 — model inspector")
    print(f"  Base: {ours.get('base_repo', 'microsoft/Phi-4-mini-instruct')}")
    print(sep)

    print(f"\n  Architecture")
    print(f"  {'Layers':<28} {layers}")
    print(f"  {'Hidden size':<28} {hidden:,}")
    print(f"  {'Attention heads':<28} {heads} Q / {kv_heads} KV  (GQA ratio {heads//kv_heads}×)")
    print(f"  {'Head dim':<28} {head_dim}")
    print(f"  {'FFN intermediate size':<28} {inter:,}")
    print(f"  {'Vocab size':<28} {vocab:,}")
    print(f"  {'Max context':<28} {max_ctx:,} tokens")
    print(f"  {'Activation':<28} SwiGLU")
    print(f"  {'Position encoding':<28} RoPE (θ={hf.get('rope_theta', 10000.0):,.0f})")
    print(f"  {'Tied embeddings':<28} {hf.get('tie_word_embeddings', False)}")

    # ── Parameter count ───────────────────────────────────────────────────────
    print(f"\n  Parameters (from safetensors index)")
    try:
        total, breakdown = _count_params_from_index(weights_dir)
        if total:
            for bucket, count in sorted(breakdown.items(), key=lambda x: -x[1]):
                pct = 100 * count / total
                print(f"  {'  ' + bucket:<30} {count/1e6:>7.1f}M  ({pct:.1f}%)")
            print(f"  {'  TOTAL':<30} {total/1e9:>7.2f}B")
        else:
            print("  (could not determine — safetensors index missing shapes)")
    except Exception as e:
        print(f"  (skipped: {e})")

    # ── Memory estimates ──────────────────────────────────────────────────────
    if total:
        print(f"\n  Memory estimates (weights only, no KV cache)")
        for dtype, bpp in DTYPE_BYTES.items():
            mb = total * bpp / 1e6
            print(f"  {'  ' + dtype:<30} {mb:>8,.0f} MB  ({mb/1024:.1f} GB)")

    # ── KV cache estimate ─────────────────────────────────────────────────────
    print(f"\n  KV cache per token (bf16)")
    kv_per_token = 2 * layers * kv_heads * head_dim * 2  # 2 for K+V, 2 bytes for bf16
    print(f"  {'  bytes / token':<30} {kv_per_token:,}")
    for ctx in (1024, 4096, 16384, 32768):
        mb = kv_per_token * ctx / 1e6
        print(f"  {'  @ ' + f'{ctx:,} tokens':<30} {mb:.0f} MB")

    # ── Inference defaults ────────────────────────────────────────────────────
    print(f"\n  Inference defaults (configs/phi4-mini.json)")
    for k, v in ours.get("nathwanigpt_defaults", {}).items():
        print(f"  {'  ' + k:<30} {v}")

    # ── Optional: load and verify ─────────────────────────────────────────────
    if args.full:
        print(f"\n  Full load verification …")
        from engine.loader import load_model
        model, _, device = load_model(weights_dir)
        actual = model.n_params()
        match = "✓" if abs(actual - total) / total < 0.001 else "✗ mismatch"
        print(f"  {'  Loaded params':<30} {actual/1e9:.2f}B  {match}")
        print(f"  {'  Device':<30} {device}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
