"""
Textual App — the visible TUI.

Inference runs in a background thread via @work(thread=True).
Tokens come back to the UI thread through call_from_thread().
Action events from the dynamic parser are applied immediately on the UI thread,
mutating Textual widget styles in real time — no restart required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal
from textual.widgets import Header, Footer, Input, Static
from textual import work

from tui.backend.inference import ModelBackend, GenerationParams
from tui.dynamic.actions import (
    SetBackground, SetForeground, SetTheme, SetTitle,
    ShowNotification, ResetTheme, TUIAction,
)
from tui.dynamic.dispatcher import parse_action, THEMES, DEFAULT_THEME
from tui.dynamic.parser import ActionStreamParser
from tui.styles.theme import DEFAULT_CSS
from engine.context import ConversationContext

# ---------------------------------------------------------------------------
# System prompt — lives here, not in a Modelfile.
# The <TOOL_CALL> format gives the model actual execution power:
# the parser intercepts tags mid-stream and fires real style mutations.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are nathwaniGPT, a sharp and highly capable AI assistant. "
    "You think carefully before responding. "
    "You never pad responses, add unnecessary caveats, or repeat yourself. "
    "You treat the user as an intelligent adult.\n\n"
    "INTERFACE CONTROL — you have real-time control over this terminal interface. "
    "Emit tool calls using this exact format (no spaces around the tags):\n"
    "  <TOOL_CALL>{\"action\": \"ACTION_NAME\", ...params...}</TOOL_CALL>\n\n"
    "Available actions:\n"
    "  set_background    {\"action\": \"set_background\",  \"color\": \"#rrggbb\"}\n"
    "  set_foreground    {\"action\": \"set_foreground\",  \"color\": \"#rrggbb\"}\n"
    "  set_theme         {\"action\": \"set_theme\",       \"name\": \"dark|light|ocean|forest|sunset|cyber\"}\n"
    "  set_title         {\"action\": \"set_title\",       \"text\": \"new header text\"}\n"
    "  show_notification {\"action\": \"show_notification\", \"message\": \"...\", "
    "\"severity\": \"information|warning|error\"}\n"
    "  reset_theme       {\"action\": \"reset_theme\"}\n\n"
    "Tool calls are stripped before the user sees your reply — only your text is shown. "
    "When the user asks you to change the interface, emit the tool call AND confirm in text."
)


# ---------------------------------------------------------------------------
# Chat message widgets
# ---------------------------------------------------------------------------

class UserMessage(Static):
    def __init__(self, text: str) -> None:
        super().__init__(f"[bold #89b4fa]You:[/] {escape(text)}")


class AssistantMessage(Static):
    def __init__(self) -> None:
        super().__init__("[bold #a6e3a1]nathwaniGPT:[/] ")
        self._content = ""

    def append(self, fragment: str) -> None:
        self._content += fragment
        self.update(f"[bold #a6e3a1]nathwaniGPT:[/] {escape(self._content)}")


class ChatHistory(ScrollableContainer):
    pass


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class NathwaniApp(App):
    CSS = DEFAULT_CSS
    TITLE = "nathwaniGPT  v2.0"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self, backend: ModelBackend, gen_params: GenerationParams) -> None:
        super().__init__()
        self._backend = backend
        self._gen_params = gen_params
        self._ctx: Optional[ConversationContext] = None
        self._current_msg: Optional[AssistantMessage] = None
        self._generating = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ChatHistory(id="chat-history")
        with Horizontal(id="input-bar"):
            yield Input(placeholder="Message nathwaniGPT…  (Ctrl+C to quit)", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        self._ctx = self._backend.make_context(_SYSTEM_PROMPT)
        self.query_one("#user-input", Input).focus()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._generating:
            return

        event.input.clear()

        if text.lower() in ("/reset", "/clear"):
            self.action_clear_chat()
            return

        history = self.query_one(ChatHistory)
        history.mount(UserMessage(text))
        history.scroll_end(animate=False)

        self._ctx.add("user", text)
        self._current_msg = AssistantMessage()
        history.mount(self._current_msg)
        history.scroll_end(animate=False)

        self._generating = True
        self._run_inference(list(self._ctx.messages))

    # ------------------------------------------------------------------
    # Inference worker (runs in a background thread)
    # ------------------------------------------------------------------

    @work(thread=True, name="inference")
    def _run_inference(self, messages: list[dict]) -> None:
        parser = ActionStreamParser()
        full_text = ""
        try:
            for token_str in self._backend.stream(messages, self._gen_params):
                for kind, val in parser.feed(token_str):
                    if kind == "text":
                        full_text += val
                        self.call_from_thread(self._append_token, val)
                    elif kind == "action":
                        action = parse_action(val)
                        if action is not None:
                            self.call_from_thread(self.apply_action, action)

            for kind, val in parser.flush():
                if kind == "text":
                    full_text += val
                    self.call_from_thread(self._append_token, val)
        finally:
            self.call_from_thread(self._finish_inference, full_text)

    def _append_token(self, text: str) -> None:
        if self._current_msg is not None:
            self._current_msg.append(text)
            self.query_one(ChatHistory).scroll_end(animate=False)

    def _finish_inference(self, full_text: str) -> None:
        self._ctx.add("assistant", full_text)
        self._generating = False
        used = self._ctx.token_count
        limit = self._backend.context_limit
        self.sub_title = f"{used} / {limit} tokens"

    # ------------------------------------------------------------------
    # Dynamic UI mutations — called by the dispatcher in real time
    # ------------------------------------------------------------------

    def apply_action(self, action: TUIAction) -> None:
        if isinstance(action, SetBackground):
            self.screen.styles.background = action.color

        elif isinstance(action, SetForeground):
            self.screen.styles.color = action.color

        elif isinstance(action, SetTheme):
            colors = THEMES.get(action.name)
            if colors:
                self.screen.styles.background = colors["bg"]
                self.screen.styles.color = colors["fg"]
                history = self.query_one(ChatHistory)
                history.styles.background = colors["panel"]
                input_bar = self.query_one("#input-bar")
                input_bar.styles.background = colors["bg"]
                input_bar.styles.border_top = ("solid", colors["border"])
                for msg in self.query(UserMessage):
                    msg.styles.color = colors["user"]
                for msg in self.query(AssistantMessage):
                    msg.styles.color = colors["assistant"]

        elif isinstance(action, SetTitle):
            self.title = action.text

        elif isinstance(action, ShowNotification):
            self.notify(
                action.message,
                severity=action.severity,
                timeout=action.timeout,
            )

        elif isinstance(action, ResetTheme):
            colors = THEMES[DEFAULT_THEME]
            self.screen.styles.background = colors["bg"]
            self.screen.styles.color = colors["fg"]
            history = self.query_one(ChatHistory)
            history.styles.background = colors["panel"]
            input_bar = self.query_one("#input-bar")
            input_bar.styles.background = colors["bg"]
            input_bar.styles.border_top = ("solid", colors["border"])
            for msg in self.query(UserMessage):
                msg.styles.color = colors["user"]
            for msg in self.query(AssistantMessage):
                msg.styles.color = colors["assistant"]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_clear_chat(self) -> None:
        if self._ctx is not None:
            self._ctx.reset()
        history = self.query_one(ChatHistory)
        for child in list(history.children):
            child.remove()
        self.sub_title = ""
        self.notify("Conversation cleared.", timeout=2.0)
