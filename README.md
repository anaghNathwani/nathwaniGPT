# nathwaniGPT

A personal AI assistant built on [Ollama](https://ollama.com), tuned for sharp, concise responses that adapt to the conversation — analytical when depth is needed, casual when it isn't.

## Quick Start

```bash
# Install Ollama if you haven't
brew install ollama

# Pull the base model
ollama pull qwen2.5:14b

# Build nathwaniGPT
ollama create nathwaniGPT -f Modelfile

# Run
ollama run nathwaniGPT
```

## Model

| Property | Value |
|---|---|
| Base | `qwen2.5:14b` (Q4_K_M, 14.8B params) |
| Context window | 8192 tokens |
| Temperature | 0.5 |
| Top-p | 0.85 |
| Top-k | 40 |
| Repeat penalty | 1.05 |
| Max tokens | unlimited |

## Versions

Versioned Modelfiles live in [`models/`](models/). The root `Modelfile` always reflects the latest version.

| Version | Notes |
|---|---|
| [v1.2](models/v1.2/) | Larger context, tighter sampling, improved system prompt for reasoning |
| [v1.2-beta](models/v1.2-beta/) | Superseded by v1.2 |
| [v1](models/v1/) | Initial release — qwen2.5:14b base |

## Adding a New Version

1. Create `models/vN/Modelfile` with your changes.
2. Create `models/vN/README.md` describing what changed and why.
3. Update the root `Modelfile` to match.
4. Update the versions table above and in [`models/README.md`](models/README.md).
