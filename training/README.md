# Training

This directory contains everything needed to fine-tune nathwaniGPT for coding using LoRA on a free Google Colab GPU.

## Quick Start

1. Open [finetune_colab.ipynb](finetune_colab.ipynb) in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Run cells top to bottom — you'll be prompted to upload your dataset at Step 4
4. Download the exported GGUF and load it into Ollama (instructions in the last cell)

## Dataset Format

Your dataset file should be `.jsonl` or `.csv` with these fields:

| Field | Required | Description |
|---|---|---|
| `instruction` or `prompt` | Yes | The coding task or question |
| `output` or `completion` | Yes | The expected code / answer |
| `input` | No | Extra context (appended to instruction) |

Example row:
```json
{"instruction": "Write a Python function that checks if a string is a palindrome.", "output": "def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"}
```

## What the Training Does

- Loads `qwen2.5:14b` at 4-bit quantization via [Unsloth](https://github.com/unslothai/unsloth)
- Attaches LoRA adapters (r=16) to attention and MLP layers
- Fine-tunes on your dataset for 3 epochs
- Exports a merged Q4_K_M GGUF ready to drop into Ollama

## Estimated Time (Colab T4)

| Dataset size | Time |
|---|---|
| 500 examples | ~30–45 min |
| 2,000 examples | ~2–3 hrs |
| 10,000 examples | ~8–12 hrs |

Colab free tier disconnects after ~12 hours of inactivity. For larger datasets, use Colab Pro or a paid GPU (RunPod/Vast.ai).

## After Training

Move the downloaded GGUF into `models/v1.8-beta/` and follow the instructions in the last notebook cell to load it into Ollama.
