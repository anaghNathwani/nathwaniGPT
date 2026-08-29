export type SetBackground    = { action: 'set_background'; color: string };
export type SetForeground    = { action: 'set_foreground'; color: string };
export type SetTheme         = { action: 'set_theme';      name: string };
export type SetTitle         = { action: 'set_title';      text: string };
export type ShowNotification = { action: 'show_notification'; message: string; severity?: string; timeout?: number };
export type ResetTheme       = { action: 'reset_theme' };

export type TUIAction =
  | SetBackground
  | SetForeground
  | SetTheme
  | SetTitle
  | ShowNotification
  | ResetTheme;
