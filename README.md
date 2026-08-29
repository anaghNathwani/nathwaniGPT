# nathwaniGPT

A custom AI assistant built from scratch — our own inference engine, our own sampler, our own server — running on permissively-licensed open weights.

No Ollama wrapping. No HuggingFace pipelines. Just PyTorch and our code.

---

## Architecture (v2.0)

```
engine/
  model.py       — Decoder-only transformer (RMSNorm · RoPE · GQA · SwiGLU)
  loader.py      — Load safetensors weights directly into our model
  sampler.py     — Temperature / top-k / nucleus / repetition penalty
  tokenizer.py   — Tokenizer wrapper (tokenizer only, not the upstream model)

serve/
  cli.py         — Interactive chat terminal
  api.py         — OpenAI-compatible REST API (streaming + non-streaming)

scripts/
  download.py    — Fetch weights from HuggingFace Hub

configs/
  phi4-mini.json    — Architecture reference for Phi-4 Mini (MIT)
  mistral-7b.json   — Architecture reference for Mistral 7B (Apache 2.0)
```

---

## Quick start

```bash
pip install -r requirements.txt

# Download base weights (MIT license — use however you want)
python scripts/download.py phi4-mini

# Chat
python serve/cli.py

# Or run the API server
python serve/api.py

# Hit it like any OpenAI-compatible endpoint
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

---

## Base models (all permissive)

| Name | License | Size | Notes |
|---|---|---|---|
| `phi4-mini` | MIT | ~7 GB | Default. Fast on Apple MPS. |
| `mistral-7b` | Apache 2.0 | ~14 GB | Strong general capability. |
| `olmo-7b` | Apache 2.0 | ~14 GB | Most open — training data is also public. |
| `smollm2` | Apache 2.0 | ~3 GB | Tiny; great for testing on CPU. |

```bash
python scripts/download.py --list
python scripts/download.py mistral-7b
python serve/cli.py --weights weights/mistral-7b
```

---

## Version history

| Version | Approach | Base |
|---|---|---|
| [v1.0–v1.2](models/) | Ollama Modelfile (system prompt wrapper) | Qwen 2.5 14B |
| v2.0-alpha | Own inference engine + permissive weights | Phi-4 Mini (MIT) |

---

## Fine-tuning

See [`training/`](training/) for the LoRA fine-tuning notebook (Colab).
Fine-tuned weights can be exported as GGUF (Ollama path, v1.x) or kept in
safetensors format for the v2 engine.
