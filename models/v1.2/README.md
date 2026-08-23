# nathwaniGPT v1.2

**Released:** 2026-08-23  
**Status:** Stable

## What Changed from v1

### Parameters

| Parameter | v1 | v1.2 | Reason |
|---|---|---|---|
| `num_ctx` | 4096 | **8192** | Doubles context window; better long-form reasoning and memory within a conversation |
| `temperature` | 0.6 | **0.5** | More precise and consistent outputs; less random drift on analytical tasks |
| `top_p` | 0.9 | **0.85** | Tightens the sampling distribution; reduces filler and hedging |
| `top_k` | _(unset)_ | **40** | Caps vocabulary candidates per token; improves coherence |
| `repeat_penalty` | 1.1 | **1.05** | Slightly relaxed — was suppressing useful repetition in structured reasoning |
| `num_predict` | _(unset)_ | **-1** | Removes generation cap; allows full chain-of-thought without truncation |

### System Prompt

Added explicit instructions to:
- Reason step by step on complex problems before answering
- Disagree with faulty premises rather than going along with them
- Acknowledge uncertainty plainly instead of hedging vaguely

## Base Model

`qwen2.5:14b` — unchanged from v1

## Recreate

```bash
ollama create nathwaniGPT -f models/v1.2/Modelfile
ollama run nathwaniGPT
```
