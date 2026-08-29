"""
Token sampling strategies: greedy, temperature, top-k, nucleus (top-p),
and repetition penalty.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated: Optional[list[int]] = None,
) -> int:
    """
    Sample the next token from a logits vector.

    Args:
        logits:            Raw logits for the full vocabulary, shape [vocab_size].
        temperature:       Softmax temperature. 0 = greedy argmax.
        top_k:             Keep only the top-k candidates (0 = disabled).
        top_p:             Nucleus probability mass (1.0 = disabled).
        repetition_penalty: >1.0 discourages repeating tokens already generated.
        generated:         Token IDs already generated (for repetition penalty).

    Returns:
        Sampled token ID (int).
    """
    logits = logits.clone().float()

    # Repetition penalty — applied before temperature so scaling is independent
    if repetition_penalty != 1.0 and generated:
        for tid in set(generated):
            score = logits[tid]
            logits[tid] = score / repetition_penalty if score > 0 else score * repetition_penalty

    # Greedy
    if temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature

    # Top-K filter
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    # Nucleus (top-p) filter
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Zero out tokens past the nucleus
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

    return int(torch.multinomial(probs, num_samples=1).item())
