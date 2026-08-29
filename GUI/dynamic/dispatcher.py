"""
Maps raw action dicts (parsed from TOOL_CALL JSON) to TUIAction dataclasses.
"""
from __future__ import annotations
from tui.dynamic.actions import (
    SetBackground, SetForeground, SetTheme, SetTitle,
    ShowNotification, ResetTheme, TUIAction,
)

THEMES: dict[str, dict[str, str]] = {
    "dark":   {"bg": "#1e1e2e", "fg": "#cdd6f4", "panel": "#181825",
               "border": "#6c7086", "user": "#89b4fa", "assistant": "#a6e3a1"},
    "light":  {"bg": "#eff1f5", "fg": "#4c4f69", "panel": "#dce0e8",
               "border": "#8c8fa1", "user": "#1e66f5", "assistant": "#40a02b"},
    "ocean":  {"bg": "#0d1117", "fg": "#79c0ff", "panel": "#161b22",
               "border": "#30363d", "user": "#58a6ff", "assistant": "#3fb950"},
    "forest": {"bg": "#1a2e1a", "fg": "#a3d9a5", "panel": "#122012",
               "border": "#2d4a2d", "user": "#57d364", "assistant": "#f0e070"},
    "sunset": {"bg": "#2d1b33", "fg": "#ffb347", "panel": "#1e0f22",
               "border": "#5a2d6b", "user": "#ff79c6", "assistant": "#ffb86c"},
    "cyber":  {"bg": "#0a0a0f", "fg": "#00ff9f", "panel": "#050508",
               "border": "#00ff9f", "user": "#00eaff", "assistant": "#ff00ff"},
}

DEFAULT_THEME = "dark"


def parse_action(data: dict) -> TUIAction | None:
    action = data.get("action", "")
    try:
        if action == "set_background":
            return SetBackground(color=str(data["color"]))
        if action == "set_foreground":
            return SetForeground(color=str(data["color"]))
        if action == "set_theme":
            return SetTheme(name=str(data["name"]).lower())
        if action == "set_title":
            return SetTitle(text=str(data["text"]))
        if action == "show_notification":
            return ShowNotification(
                message=str(data["message"]),
                severity=data.get("severity", "information"),
                timeout=float(data.get("timeout", 3.0)),
            )
        if action == "reset_theme":
            return ResetTheme()
    except (KeyError, ValueError):
        pass
    return None
