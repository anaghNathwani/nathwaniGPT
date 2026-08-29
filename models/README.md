# Models

This directory tracks every version of nathwaniGPT. It tells the story of how we went from wrapping someone else's model with a clever system prompt to building the entire inference stack ourselves.

---

## Era 1 — System prompt engineering (v1.x)

We loaded Qwen 2.5 14B through Ollama and shaped its behaviour entirely through the system prompt and sampling parameters. Each version in this era is an Ollama `Modelfile` — a configuration file, not an engine.

| Version | Date | What changed |
|---|---|---|
| [v1](v1/) | 2026-08-22 | First release — base system prompt, Qwen 2.5 14B |
| [v1.2-beta](v1.2-beta/) | 2026-08-22 | Doubled context, tighter sampling, stronger reasoning prompt |
| [v1.2](v1.2/) | 2026-08-23 | Promoted from beta — stable |

To run any v1.x version:
```bash
ollama create nathwaniGPT -f models/v1.2/Modelfile
ollama run nathwaniGPT
```

---

## Era 2 — Our own engine (v2.x)

We stopped configuring other people's runtimes and built our own. v2.0 is a full inference harness: custom transformer in PyTorch, our own sampler, context manager, and server. The only thing we don't own is the base weights — and we picked ones with MIT/Apache 2.0 licenses so there are no restrictions.

| Version | Date | What it is |
|---|---|---|
| [v2.0](v2.0/) | 2026-08-29 | Full harness — own engine, phi4-mini weights (MIT) |

To run v2.0:
```bash
python models/v2.0/chat.py          # interactive CLI
python models/v2.0/serve.py         # API server on :8080
python models/v2.0/generate.py "…"  # single-shot
python models/v2.0/benchmark.py     # speed test
python models/v2.0/inspect.py       # architecture info
```

---

## Adding a new version

- **v2.x onwards**: Create a `models/vX.Y/` directory. Add runnable Python files (`chat.py`, `serve.py`, etc.) that reference the shared `engine/`, `serve/`, `weights/`, and `configs/` directories. No Modelfiles.
- **v1.x (legacy)**: Modelfile + README. This convention is retired.
