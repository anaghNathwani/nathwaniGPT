/**
 * Maps raw JSON payloads → typed TUIActions, then executes them
 * by mutating CSS custom properties on :root. This is the real power:
 * the AI updating its own interface via CSS variables at runtime.
 */
import type { TUIAction } from './actions';

export const THEMES: Record<string, Record<string, string>> = {
  dark:   { bg: '#1e1e2e', fg: '#cdd6f4', panel: '#181825', border: '#6c7086', user: '#89b4fa', assistant: '#a6e3a1', input: '#313244' },
  light:  { bg: '#eff1f5', fg: '#4c4f69', panel: '#dce0e8', border: '#8c8fa1', user: '#1e66f5', assistant: '#40a02b', input: '#ccd0da' },
  ocean:  { bg: '#0d1117', fg: '#79c0ff', panel: '#161b22', border: '#30363d', user: '#58a6ff', assistant: '#3fb950', input: '#21262d' },
  forest: { bg: '#1a2e1a', fg: '#a3d9a5', panel: '#122012', border: '#2d4a2d', user: '#57d364', assistant: '#f0e070', input: '#1e3a1e' },
  sunset: { bg: '#2d1b33', fg: '#ffb347', panel: '#1e0f22', border: '#5a2d6b', user: '#ff79c6', assistant: '#ffb86c', input: '#3d1f45' },
  cyber:  { bg: '#0a0a0f', fg: '#00ff9f', panel: '#050508', border: '#00ff9f', user: '#00eaff', assistant: '#ff00ff', input: '#0f0f18' },
};

export const DEFAULT_THEME = 'dark';

function cssVar(name: string, value: string) {
  document.documentElement.style.setProperty(`--${name}`, value);
}

function applyColors(colors: Record<string, string>) {
  for (const [k, v] of Object.entries(colors)) cssVar(k, v);
}

export interface DispatchCallbacks {
  setTitle: (t: string) => void;
  notify:   (msg: string, severity: string, timeout: number) => void;
}

export function dispatch(action: TUIAction, cb: DispatchCallbacks): void {
  switch (action.action) {
    case 'set_background':
      cssVar('bg', action.color);
      break;
    case 'set_foreground':
      cssVar('fg', action.color);
      break;
    case 'set_theme': {
      const colors = THEMES[action.name];
      if (colors) applyColors(colors);
      break;
    }
    case 'set_title':
      document.title = action.text;
      cb.setTitle(action.text);
      break;
    case 'show_notification':
      cb.notify(action.message, action.severity ?? 'information', action.timeout ?? 3);
      break;
    case 'reset_theme':
      applyColors(THEMES[DEFAULT_THEME]);
      break;
  }
}

export function parseAction(data: Record<string, unknown>): TUIAction | null {
  const a = data.action as string;
  try {
    if (a === 'set_background')    return { action: a, color: data.color as string };
    if (a === 'set_foreground')    return { action: a, color: data.color as string };
    if (a === 'set_theme')         return { action: a, name: (data.name as string).toLowerCase() };
    if (a === 'set_title')         return { action: a, text: data.text as string };
    if (a === 'show_notification') return { action: a, message: data.message as string, severity: data.severity as string, timeout: data.timeout as number };
    if (a === 'reset_theme')       return { action: a };
  } catch { /* malformed payload */ }
  return null;
}
