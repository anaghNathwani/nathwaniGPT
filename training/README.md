# Training

Fine-tuning nathwaniGPT on your own data.

In the v1.x era, fine-tuning meant exporting a GGUF and loading it into Ollama — we were still inside someone else's runtime. With v2.0 and our own engine, fine-tuned adapters stay in safetensors format and load directly into the harness. No conversion step, no runtime swap.

---

## LoRA fine-tuning (Colab)

[`finetune_colab.ipynb`](finetune_colab.ipynb) — runs on a free T4 GPU in Google Colab.

**Quick start:**
1. Open in Colab, set runtime to **T4 GPU**
2. Run cells top to bottom — you'll be prompted to upload your dataset at Step 4
3. Download the exported safetensors adapter
4. Drop it into `weights/phi4-mini-finetuned/` and run normally

```bash
python models/v2.0/chat.py --weights weights/phi4-mini-finetuned
```

---

## Dataset format

One JSON object per line (`.jsonl`):

| Field | Required | Description |
|---|---|---|
| `instruction` or `prompt` | Yes | The task or question |
| `output` or `completion` | Yes | The expected answer |
| `input` | No | Extra context appended to the instruction |

```json
{"instruction": "Write a Python function that checks if a string is a palindrome.", "output": "def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"}
```

---

## What the notebook does

- Loads `phi4-mini` weights (or any model in `weights/`) at 4-bit via [Unsloth](https://github.com/unslothai/unsloth)
- Attaches LoRA adapters (r=16) to attention and MLP layers
- Fine-tunes for 3 epochs
- Exports merged weights in safetensors format

---

## Estimated time (Colab T4)

| Dataset size | Time |
|---|---|
| 500 examples | ~30–45 min |
| 2,000 examples | ~2–3 hrs |
| 10,000 examples | ~8–12 hrs |

Colab free tier disconnects after ~12 hours. For larger datasets use Colab Pro, RunPod, or Vast.ai.

---

## Evaluating after fine-tuning

Use `scripts/eval.py` to measure quality before and after:

```bash
# Baseline
python scripts/eval.py --dataset evals/test.jsonl --weights weights/phi4-mini

# After fine-tuning
python scripts/eval.py --dataset evals/test.jsonl --weights weights/phi4-mini-finetuned
```

See [`scripts/eval.py`](../scripts/eval.py) for dataset format and metric details.
