from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class SetBackground:
    color: str


@dataclass
class SetForeground:
    color: str


@dataclass
class SetTheme:
    name: str  # dark | light | ocean | forest | sunset | cyber


@dataclass
class SetTitle:
    text: str


@dataclass
class ShowNotification:
    message: str
    severity: Literal["information", "warning", "error"] = "information"
    timeout: float = 3.0


@dataclass
class ResetTheme:
    pass


TUIAction = SetBackground | SetForeground | SetTheme | SetTitle | ShowNotification | ResetTheme
