"""
Thin wrapper around HuggingFace tokenizers.

We only use the tokenizer half of transformers — not the model —
so this is just a convenience shim that keeps the rest of our code
from importing transformers directly.
"""

from pathlib import Path
from typing import Union

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, weights_dir: Union[str, Path]):
        self._tok = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token without skipping specials (for streaming)."""
        return self._tok.decode([token_id], skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eos_id(self) -> int:
        return self._tok.eos_token_id or 2

    @property
    def stop_ids(self) -> set[int]:
        """All token IDs that should terminate generation.

        Includes the canonical EOS token plus any model-specific end-of-turn
        tokens (e.g. Phi-4-mini uses <|end|> to close assistant turns, which
        is distinct from <|endoftext|>).
        """
        ids: set[int] = set()
        if self._tok.eos_token_id is not None:
            ids.add(self._tok.eos_token_id)
        # Collect any <|end*|>-style tokens and role boundary tokens
        for token in ("<|end|>", "<|endoftext|>", "<|user|>", "<|eot_id|>", "</s>"):
            tid = self._tok.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid != self._tok.unk_token_id:
                ids.add(tid)
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self._tok)
