#!/usr/bin/env python3
"""
Download model weights from HuggingFace Hub.

All listed models use Apache 2.0 or MIT licensing — no restrictions
on use, modification, or distribution.

Usage:
    python scripts/download.py phi4-mini
    python scripts/download.py mistral-7b
    python scripts/download.py --list
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry — only permissively licensed models
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "phi4-mini": {
        "repo": "microsoft/Phi-4-mini-instruct",
        "license": "MIT",
        "size": "~7GB (bf16 safetensors)",
        "notes": "Best for Mac. Fast on MPS. Strong reasoning.",
    },
    "mistral-7b": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "license": "Apache 2.0",
        "size": "~14GB (bf16)",
        "notes": "Excellent general capability. Needs 16GB+ RAM.",
    },
    "olmo-7b": {
        "repo": "allenai/OLMo-2-1124-7B-Instruct",
        "license": "Apache 2.0",
        "size": "~14GB (bf16)",
        "notes": "Most open model — training data and code are public too.",
    },
    "smollm2": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "license": "Apache 2.0",
        "size": "~3GB (bf16)",
        "notes": "Tiny but capable. Instant on CPU. Good for testing.",
    },
}


def list_models():
    print("\nAvailable models:\n")
    for name, info in MODELS.items():
        print(f"  {name:<14}  {info['license']:<12}  {info['size']}")
        print(f"               {info['notes']}")
        print()


def download(name: str, token: str | None = None, dest_override: str | None = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: pip install huggingface_hub")
        sys.exit(1)

    if name not in MODELS:
        print(f"Unknown model '{name}'. Run with --list to see options.")
        sys.exit(1)

    info = MODELS[name]
    repo = info["repo"]
    dest = Path(dest_override) if dest_override else Path(__file__).parent.parent / "weights" / name

    print(f"\nDownloading {name} ({info['license']}) from {repo}")
    print(f"Destination: {dest}")
    print(f"Estimated size: {info['size']}\n")

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        # Skip deprecated/redundant formats — we want safetensors only
        ignore_patterns=["*.bin", "*.pt", "*.msgpack", "original/", "flax_model*"],
        token=token,
    )

    print(f"\nDone. Weights saved to: {dest}")
    print(f"\nTo chat:  python serve/cli.py --weights weights/{name}")
    print(f"To serve: python serve/api.py  --weights weights/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download nathwaniGPT base weights")
    parser.add_argument("model", nargs="?", help="Model name (see --list)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--token", help="HuggingFace token (for private/gated models)")
    parser.add_argument("--dest", help="Override destination directory")
    args = parser.parse_args()

    if args.list or not args.model:
        list_models()
        sys.exit(0)

    download(args.model, token=args.token, dest_override=args.dest)
