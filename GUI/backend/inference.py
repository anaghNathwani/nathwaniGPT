"""
Model backend: loads weights once and streams token strings during inference.
All PyTorch work stays in this module so the rest of the TUI stays framework-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Optional

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine.loader import load_model
from engine.sampler import sample_token
from engine.tokenizer import Tokenizer
from engine.context import ConversationContext


class GenerationParams:
    __slots__ = ("max_tokens", "temperature", "top_k", "top_p", "repetition_penalty")

    def __init__(
        self,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty


class ModelBackend:
    def __init__(self, weights_path: Path, context_limit: int = 16_384) -> None:
        self.weights_path = weights_path
        self.context_limit = context_limit
        self._model = None
        self._tokenizer: Optional[Tokenizer] = None
        self._device = "cpu"

    def load(self) -> None:
        self._model, _, self._device = load_model(self.weights_path)
        self._tokenizer = Tokenizer(self.weights_path)

    def make_context(self, system_prompt: str) -> ConversationContext:
        return ConversationContext(
            self._tokenizer,
            max_tokens=self.context_limit,
            system_prompt=system_prompt,
        )

    def stream(
        self, messages: list[dict], params: GenerationParams
    ) -> Generator[str, None, None]:
        """Yield decoded token strings one at a time, synchronously."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = self._tokenizer.encode(prompt)
        generated_ids: list[int] = []
        stop_ids = self._tokenizer.stop_ids

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        logits, kv_caches = self._model(input_tensor, start_pos=0)
        start_pos = len(input_ids)

        for _ in range(params.max_tokens):
            next_token = sample_token(
                logits[0, -1],
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                repetition_penalty=params.repetition_penalty,
                generated=generated_ids,
            )
            generated_ids.append(next_token)

            if next_token in stop_ids:
                break

            yield self._tokenizer.decode_token(next_token)

            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self._device)
            logits, kv_caches = self._model(
                input_tensor, kv_caches=kv_caches, start_pos=start_pos
            )
            start_pos += 1
