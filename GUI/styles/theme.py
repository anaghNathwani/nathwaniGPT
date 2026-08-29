"""
Default Textual CSS and theme colour registry.
"""

DEFAULT_CSS = """
Screen {
    background: #1e1e2e;
    color: #cdd6f4;
}

#chat-history {
    height: 1fr;
    overflow-y: auto;
    padding: 1 2;
    background: #181825;
}

UserMessage {
    color: #89b4fa;
    margin-bottom: 1;
    padding: 0 1;
}

AssistantMessage {
    color: #a6e3a1;
    margin-bottom: 1;
    padding: 0 1;
}

#input-bar {
    dock: bottom;
    height: auto;
    padding: 0 1 1 1;
    background: #1e1e2e;
    border-top: solid #6c7086;
}

#user-input {
    background: #181825;
    border: solid #6c7086;
    color: #cdd6f4;
    width: 1fr;
}
"""
