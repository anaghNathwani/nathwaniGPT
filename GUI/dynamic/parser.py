"""
Real-time stream parser for embedded tool calls.

The model emits tags like:
    <TOOL_CALL>{"action": "set_background", "color": "#1e1e2e"}</TOOL_CALL>

Feed tokens in one character at a time via feed(). The parser yields:
    ("text",   str)  — characters safe to display
    ("action", dict) — parsed JSON payload ready for the dispatcher

The lookahead buffer holds up to len(OPEN) chars so regular text is
only delayed by at most one tag-length, then flushed immediately.
"""
import json

OPEN = "<TOOL_CALL>"
CLOSE = "</TOOL_CALL>"


class ActionStreamParser:
    def __init__(self) -> None:
        self._buf = ""      # potential OPEN-tag prefix accumulator
        self._in_tag = False
        self._tag_buf = ""  # content inside an open tag

    def feed(self, chunk: str) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        for c in chunk:
            if self._in_tag:
                self._tag_buf += c
                if self._tag_buf.endswith(CLOSE):
                    payload = self._tag_buf[: -len(CLOSE)]
                    try:
                        events.append(("action", json.loads(payload)))
                    except json.JSONDecodeError:
                        events.append(("text", OPEN + payload + CLOSE))
                    self._tag_buf = ""
                    self._in_tag = False
            else:
                self._buf += c
                if self._buf.endswith(OPEN):
                    pre = self._buf[: -len(OPEN)]
                    if pre:
                        events.append(("text", pre))
                    self._buf = ""
                    self._in_tag = True
                elif len(self._buf) > len(OPEN):
                    # First char cannot be the start of OPEN — safe to emit.
                    events.append(("text", self._buf[0]))
                    self._buf = self._buf[1:]
        return events

    def flush(self) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        if self._buf:
            events.append(("text", self._buf))
        if self._in_tag and self._tag_buf:
            events.append(("text", OPEN + self._tag_buf))
        self._buf = ""
        self._tag_buf = ""
        self._in_tag = False
        return events
