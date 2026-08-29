"""
Conversation context manager.

Tracks message history with per-message token counts and automatically
trims the oldest non-system messages when the total approaches the model's
context window, so generation never gets silently truncated.

Used by: serve/cli.py, serve/api.py, models/v2.0/chat.py
"""

from typing import Optional


class ConversationContext:
    # Reserve this fraction of the context window for generation headroom
    _HEADROOM = 0.15

    def __init__(
        self,
        tokenizer,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ):
        self._tok = tokenizer
        self._max = max_tokens
        self._limit = int(max_tokens * (1 - self._HEADROOM))

        self._messages: list[dict] = []
        self._counts:   list[int]  = []

        if system_prompt:
            self._push("system", system_prompt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def _push(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        self._counts.append(self._count(content))

    def _trim(self):
        """Drop the oldest non-system messages until we fit within the limit."""
        has_system = self._messages and self._messages[0]["role"] == "system"
        floor = 1 if has_system else 0

        while self.token_count > self._limit and len(self._messages) > floor + 1:
            self._messages.pop(floor)
            self._counts.pop(floor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str):
        """Append a message and trim history if needed."""
        self._push(role, content)
        self._trim()

    def reset(self, keep_system: bool = True):
        """Clear conversation history, optionally preserving the system prompt."""
        if keep_system and self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
            self._counts   = [self._counts[0]]
        else:
            self._messages = []
            self._counts   = []

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    @property
    def token_count(self) -> int:
        return sum(self._counts)

    @property
    def tokens_remaining(self) -> int:
        """Tokens available for the next generation (after headroom)."""
        return max(0, self._limit - self.token_count)

    @property
    def utilisation(self) -> float:
        """Fraction of the usable context window currently occupied (0–1)."""
        return self.token_count / self._limit if self._limit else 0.0

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationContext("
            f"{len(self._messages)} msgs, "
            f"{self.token_count}/{self._max} tokens, "
            f"{self.utilisation:.0%} full)"
        )
