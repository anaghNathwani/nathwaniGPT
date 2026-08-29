# nathwaniGPT

We started where most people start: wrapping an existing model with a carefully crafted system prompt, tweaking parameters, calling it ours. That was v1.

Then we decided to actually build the thing.

v2.0 is a full inference harness written from scratch — our own transformer, our own sampler, our own context manager, our own server. We load open weights directly from safetensors, run every token through our code, and own the entire stack. No Ollama. No HuggingFace pipelines. No black boxes.

---

## What we built

```
engine/
  model.py        Decoder-only transformer — RMSNorm, RoPE, GQA, SwiGLU
  loader.py       Loads safetensors weights directly; handles fused QKV,
                  sharded checkpoints, and int8 dequantization
  sampler.py      Temperature · top-k · nucleus · repetition penalty
  tokenizer.py    HF tokenizer wrapper (tokenizer only, not their model)
  context.py      Conversation context manager — tracks token counts,
                  auto-trims history before the window fills up

serve/
  cli.py          Interactive chat terminal with live token counter
  api.py          OpenAI-compatible REST API (streaming + non-streaming)

scripts/
  download.py     Fetch weights from HuggingFace Hub
  quantize.py     Int8 weight quantization — cuts ~7 GB to ~3.5 GB
  eval.py         Quality evaluation against a JSONL test set

models/
  v1/             Era 1: Ollama Modelfile + system prompt (Qwen 2.5 14B)
  v1.2/           Era 1: Refined sampling params, improved system prompt
  v2.0/           Era 2: Our engine, our weights, our full harness

training/
  finetune_colab.ipynb    LoRA fine-tuning on Colab (free T4 GPU)

configs/
  phi4-mini.json    Architecture reference + inference defaults (MIT)
  mistral-7b.json   Architecture reference for Mistral 7B (Apache 2.0)

weights/
  phi4-mini/        microsoft/Phi-4-mini-instruct — 3.8B, MIT license
```

---

## Quick start

```bash
pip install -r requirements.txt

# Download base weights
python scripts/download.py phi4-mini

# Chat
python models/v2.0/chat.py

# Or run the API server (OpenAI-compatible on :8080)
python models/v2.0/serve.py

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

---

## All the tools

| Script | What it does |
|---|---|
| `models/v2.0/chat.py` | Interactive chat CLI |
| `models/v2.0/serve.py` | API server on :8080 |
| `models/v2.0/generate.py` | Single-shot generation (scriptable, stdin support) |
| `models/v2.0/benchmark.py` | Prefill latency + decode tokens/sec |
| `models/v2.0/inspect.py` | Architecture, param counts, memory estimates — no full load |
| `scripts/download.py` | Pull weights from HuggingFace Hub |
| `scripts/quantize.py` | Int8 quantization — halves file size |
| `scripts/eval.py` | Run quality evaluation against a JSONL test set |
| `training/finetune_colab.ipynb` | LoRA fine-tune on your own data |

---

## Base weights (all permissive)

| Name | License | Size | Notes |
|---|---|---|---|
| `phi4-mini` | MIT | ~7 GB | Default. Fast on Apple MPS. Strong reasoning. |
| `mistral-7b` | Apache 2.0 | ~14 GB | Strong general capability. |
| `olmo-7b` | Apache 2.0 | ~14 GB | Most open — training data is public too. |
| `smollm2` | Apache 2.0 | ~3 GB | Tiny; instant on CPU. Good for testing. |

```bash
python scripts/download.py --list
python scripts/download.py mistral-7b
python models/v2.0/chat.py --weights weights/mistral-7b
```

---

## Version history

| Version | What it was | Base |
|---|---|---|
| v1.0 | System prompt wrapper via Ollama | Qwen 2.5 14B |
| v1.2 | Refined sampling, improved prompt, doubled context | Qwen 2.5 14B |
| **v2.0** | **Full harness — our engine, our stack, our weights** | Phi-4 Mini (MIT) |

---

## Fine-tuning

See [`training/`](training/) — LoRA fine-tuning notebook for Colab.
Adapters are exported as safetensors and load directly into the v2.0 engine.
