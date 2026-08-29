import { useState, useRef, useEffect, useCallback } from 'react';
import { ActionStreamParser } from './dynamic/parser';
import { parseAction, dispatch, THEMES, DEFAULT_THEME } from './dynamic/dispatcher';
import './App.css';

// ── Types ────────────────────────────────────────────────────────────────

interface Message {
  id:      string;
  role:    'user' | 'assistant';
  content: string;
}

interface Toast {
  id:       string;
  message:  string;
  severity: string;
}

// ── Constants ────────────────────────────────────────────────────────────

const API_BASE = '';  // same origin (proxied via Vite in dev, same port in prod)

const SYSTEM_PROMPT =
  'You are nathwaniGPT, a sharp and highly capable AI assistant. ' +
  'You think carefully before responding. ' +
  'You never pad responses, add unnecessary caveats, or repeat yourself. ' +
  'You treat the user as an intelligent adult.\n\n' +
  'INTERFACE CONTROL — you have real-time control over this interface. ' +
  'Emit tool calls using this exact format (no spaces around the tags):\n' +
  '  <TOOL_CALL>{"action": "ACTION_NAME", ...params}</TOOL_CALL>\n\n' +
  'Available actions:\n' +
  '  set_background    {"action": "set_background",    "color": "#rrggbb"}\n' +
  '  set_foreground    {"action": "set_foreground",    "color": "#rrggbb"}\n' +
  '  set_theme         {"action": "set_theme",         "name": "dark|light|ocean|forest|sunset|cyber"}\n' +
  '  set_title         {"action": "set_title",         "text": "new header title"}\n' +
  '  show_notification {"action": "show_notification", "message": "...", "severity": "information|warning|error"}\n' +
  '  reset_theme       {"action": "reset_theme"}\n\n' +
  'Tool calls are stripped before the user sees your reply — only your text is shown. ' +
  'When asked to change the interface, emit the tool call AND briefly confirm in text.';

let _uid = 0;
const uid = () => `m${++_uid}`;

// ── Component ────────────────────────────────────────────────────────────

export default function App() {
  const [messages,   setMessages]   = useState<Message[]>([]);
  const [input,      setInput]      = useState('');
  const [generating, setGenerating] = useState(false);
  const [title,      setTitle]      = useState('nathwaniGPT  v2.0');
  const [toasts,     setToasts]     = useState<Toast[]>([]);
  const [tokenInfo,  setTokenInfo]  = useState<{ used: number; limit: number } | null>(null);

  const historyRef = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLInputElement>(null);
  const abortRef   = useRef<AbortController | null>(null);

  // Apply default theme on first render
  useEffect(() => {
    for (const [k, v] of Object.entries(THEMES[DEFAULT_THEME])) {
      document.documentElement.style.setProperty(`--${k}`, v);
    }
  }, []);

  // Keep scroll pinned to bottom while generating
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [messages]);

  const notify = useCallback((message: string, severity: string, timeout: number) => {
    const id = uid();
    setToasts(prev => [...prev, { id, message, severity }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), timeout * 1000);
  }, []);

  const submit = useCallback(async () => {
    const text = input.trim();
    if (!text || generating) return;
    setInput('');

    const userMsg: Message    = { id: uid(), role: 'user',      content: text };
    const assistantId         = uid();
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '' };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setGenerating(true);

    // Build history including system prompt
    const history = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...[...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
    ];

    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ messages: history, stream: true }),
        signal:  abortRef.current.signal,
      });

      if (!res.ok) {
        notify(`API error: ${res.status}`, 'error', 4);
        return;
      }

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      const parser  = new ActionStreamParser();
      let fullText  = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const raw = decoder.decode(value, { stream: true });
        for (const line of raw.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data: '))          continue;

          let chunk: { choices?: Array<{ delta?: { content?: string } }> };
          try { chunk = JSON.parse(trimmed.slice(6)) as typeof chunk; }
          catch { continue; }

          const token = chunk.choices?.[0]?.delta?.content ?? '';
          if (!token) continue;

          for (const event of parser.feed(token)) {
            if (event.type === 'text') {
              fullText += event.value;
              setMessages(prev =>
                prev.map(m => m.id === assistantId ? { ...m, content: fullText } : m)
              );
            } else {
              const action = parseAction(event.value);
              if (action) dispatch(action, { setTitle, notify });
            }
          }
        }
      }

      // Flush any buffered partial tag
      for (const event of parser.flush()) {
        if (event.type === 'text') {
          fullText += event.value;
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, content: fullText } : m)
          );
        }
      }

      // Rough token estimate for display
      const totalChars = history.reduce((s, m) => s + m.content.length, 0) + fullText.length;
      setTokenInfo({ used: Math.round(totalChars / 4), limit: 16384 });

    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        notify('Connection error — is the API server running?', 'error', 5);
      }
    } finally {
      setGenerating(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [input, generating, messages, notify]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  };

  const lastId = messages.at(-1)?.id;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <span className="header-title">{title}</span>
        <div className="header-meta">
          {tokenInfo && (
            <span className="token-count">{tokenInfo.used} / {tokenInfo.limit} tokens</span>
          )}
        </div>
      </header>

      {/* Chat history */}
      <div className="chat-history" ref={historyRef}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">nathwaniGPT</span>
            <span>Try: "switch to ocean theme" or "make the background deep purple"</span>
          </div>
        ) : (
          messages.map(m => (
            <div key={m.id} className={`message message-${m.role}`}>
              <span className="message-label">{m.role === 'user' ? 'You' : 'nathwaniGPT'}</span>
              <span className="message-content">
                {m.content}
                {generating && m.id === lastId && m.role === 'assistant' && (
                  <span className="cursor" />
                )}
              </span>
            </div>
          ))
        )}
      </div>

      {/* Input bar */}
      <div className="input-bar">
        <input
          ref={inputRef}
          className="input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message nathwaniGPT…"
          disabled={generating}
          autoFocus
        />
        <button
          className="send-btn"
          onClick={() => void submit()}
          disabled={generating || !input.trim()}
          aria-label="Send"
        >
          {generating ? '…' : '↑'}
        </button>
      </div>

      {/* Toast notifications */}
      <div className="notifications" aria-live="polite">
        {toasts.map(t => (
          <div key={t.id} className={`notification notification-${t.severity}`}>
            {t.message}
          </div>
        ))}
      </div>
    </div>
  );
}
