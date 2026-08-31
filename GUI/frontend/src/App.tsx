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

// Keep this short — Phi-4-mini degrades with long system prompts.
const SYSTEM_PROMPT =
  'You are nathwaniGPT, a concise AI assistant. Answer directly, no padding.\n\n' +
  'You can control the UI by emitting a tool call anywhere in your reply:\n' +
  '<TOOL_CALL>{"action":"NAME",...params}</TOOL_CALL>\n' +
  'Actions: set_theme(name: dark|light|ocean|forest|sunset|cyber), ' +
  'set_background(color), set_foreground(color), set_title(text), ' +
  'show_notification(message, severity: information|warning|error), reset_theme.\n' +
  'Tool calls are invisible to the user. Emit one, then confirm in plain text.';

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
        const body = await res.text().catch(() => '');
        let detail = `HTTP ${res.status}`;
        try { detail = (JSON.parse(body) as { detail?: string }).detail ?? detail; } catch { /* raw text */ }
        notify(`API error: ${detail}`, 'error', 6);
        setGenerating(false);
        return;
      }

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      const parser  = new ActionStreamParser();
      let fullText  = '';
      let lineBuf   = '';  // accumulates partial SSE lines across read() chunks

      const processLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') return;
        if (!trimmed.startsWith('data: ')) return;
        let chunk: { choices?: Array<{ delta?: { content?: string } }>; error?: string };
        try { chunk = JSON.parse(trimmed.slice(6)) as typeof chunk; }
        catch { return; }
        if (chunk.error) { notify(`Model error: ${chunk.error}`, 'error', 8); return; }
        const token = chunk.choices?.[0]?.delta?.content ?? '';
        if (!token) return;
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
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append decoded bytes to line buffer, then process all complete lines.
        // This prevents JSON parse failures when a read() boundary falls mid-line.
        lineBuf += decoder.decode(value, { stream: true });
        const lines = lineBuf.split('\n');
        lineBuf = lines.pop() ?? '';  // last element may be an incomplete line
        for (const line of lines) processLine(line);
      }
      // Flush any remaining partial line
      if (lineBuf) processLine(lineBuf);

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
