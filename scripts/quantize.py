#!/usr/bin/env python3
"""
nathwaniGPT — weight quantization

Converts a safetensors checkpoint to int8 using per-tensor absmax scaling.
Cuts weight file size roughly in half (~7 GB → ~3.5 GB for phi4-mini).

The quantized weights are saved to a new directory and are loaded
transparently by engine/loader.py (which detects quantization.json and
dequantizes at load time).

Engine:  engine/loader.py (reads quantization.json on load)
Source:  weights/phi4-mini/           (or any safetensors directory)
Output:  weights/phi4-mini-int8/      (default)

Usage:
    python scripts/quantize.py
    python scripts/quantize.py --src weights/mistral-7b --dst weights/mistral-7b-int8
    python scripts/quantize.py --dtype float16            # convert to fp16 instead
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SRC = ROOT / "weights" / "phi4-mini"
_2D_ONLY = True  # only quantize 2-D weight matrices (Linear layers); leave embeddings/norms in fp16


def _int8_quantize(src_dir: Path, dst_dir: Path):
    """Absmax int8 quantization — saves int8 tensors + quantization.json with scales."""
    import torch
    from safetensors.torch import load_file, save_file

    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    scales: dict[str, float] = {}
    src_bytes = 0
    dst_bytes = 0

    for shard in shards:
        print(f"  Quantizing {shard} …")
        raw = load_file(src_dir / shard)
        out: dict[str, torch.Tensor] = {}

        for name, tensor in raw.items():
            src_bytes += tensor.numel() * tensor.element_size()
            t = tensor.to(torch.float32)

            if _2D_ONLY and t.dim() == 2:
                # Per-tensor absmax scale
                scale = t.abs().max().item()
                if scale == 0:
                    scale = 1.0
                scale = scale / 127.0
                q = (t / scale).round().clamp(-128, 127).to(torch.int8)
                out[name] = q
                scales[name] = scale
                dst_bytes += q.numel() * 1  # int8 = 1 byte
            else:
                # Keep non-matrix tensors (norms, biases, embeddings) in fp16
                out[name] = t.to(torch.float16)
                dst_bytes += out[name].numel() * 2

        save_file(out, dst_dir / shard)

    # Save scale factors — loader.py reads this to dequantize
    with open(dst_dir / "quantization.json", "w") as f:
        json.dump(scales, f, indent=2)

    # Copy tokenizer and config files
    for fname in dst_dir.parent.glob("*"):
        pass  # dst_dir is already set
    for fname in src_dir.iterdir():
        if fname.suffix not in (".safetensors",) and fname.name != "quantization.json":
            shutil.copy2(fname, dst_dir / fname.name)

    # Update shard index if it exists
    if index_path.exists():
        shutil.copy2(index_path, dst_dir / "model.safetensors.index.json")

    reduction = (1 - dst_bytes / src_bytes) * 100 if src_bytes else 0
    print(f"\n  Original : {src_bytes/1e9:.2f} GB")
    print(f"  Quantized: {dst_bytes/1e9:.2f} GB")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Scales   : {len(scales)} tensors → {dst_dir}/quantization.json")


def _fp16_convert(src_dir: Path, dst_dir: Path):
    """Convert all weights to fp16 (useful when source is fp32)."""
    import torch
    from safetensors.torch import load_file, save_file

    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    shards = (
        sorted(set(json.load(open(index_path))["weight_map"].values()))
        if index_path.exists()
        else ["model.safetensors"]
    )

    src_bytes = 0
    dst_bytes = 0

    for shard in shards:
        print(f"  Converting {shard} …")
        raw = load_file(src_dir / shard)
        out = {k: v.to(torch.float16) for k, v in raw.items()}
        for v in raw.values():
            src_bytes += v.numel() * v.element_size()
        for v in out.values():
            dst_bytes += v.numel() * v.element_size()
        save_file(out, dst_dir / shard)

    for fname in src_dir.iterdir():
        if fname.suffix not in (".safetensors",):
            shutil.copy2(fname, dst_dir / fname.name)

    reduction = (1 - dst_bytes / src_bytes) * 100 if src_bytes else 0
    print(f"\n  Original : {src_bytes/1e9:.2f} GB")
    print(f"  Converted: {dst_bytes/1e9:.2f} GB")
    print(f"  Reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="nathwaniGPT weight quantization")
    parser.add_argument("--src",   default=str(DEFAULT_SRC),        help="Source weights directory")
    parser.add_argument("--dst",   default=None,                    help="Destination directory (default: <src>-int8 or <src>-fp16)")
    parser.add_argument("--dtype", default="int8", choices=["int8", "float16"],
                        help="Target dtype (default: int8)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    suffix = "-int8" if args.dtype == "int8" else "-fp16"
    dst = Path(args.dst) if args.dst else src.parent / (src.name + suffix)

    if dst.exists():
        print(f"Destination already exists: {dst}")
        print("Delete it first or specify a different --dst.")
        sys.exit(1)

    print(f"\nnathwaniGPT quantize: {src.name} → {dst.name}  (dtype={args.dtype})")
    print(f"{'─'*52}")
    t0 = time.perf_counter()

    if args.dtype == "int8":
        _int8_quantize(src, dst)
    else:
        _fp16_convert(src, dst)

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Quantized weights: {dst}")
    print(f"\n  To use:  python serve/cli.py --weights {dst}")
    print(f"           python serve/api.py  --weights {dst}\n")


if __name__ == "__main__":
    main()
