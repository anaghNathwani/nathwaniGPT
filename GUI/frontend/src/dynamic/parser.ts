/**
 * Real-time stream parser for embedded tool calls.
 *
 * The model emits:
 *   <TOOL_CALL>{"action": "set_theme", "name": "ocean"}</TOOL_CALL>
 *
 * feed() processes tokens one chunk at a time and yields:
 *   { type: 'text',   value: string }
 *   { type: 'action', value: Record<string, unknown> }
 *
 * Tags are stripped from the text stream before reaching the display.
 */

const OPEN  = '<TOOL_CALL>';
const CLOSE = '</TOOL_CALL>';

export type StreamEvent =
  | { type: 'text';   value: string }
  | { type: 'action'; value: Record<string, unknown> };

export class ActionStreamParser {
  private buf    = '';   // lookahead for potential OPEN prefix
  private inTag  = false;
  private tagBuf = '';   // content inside an open tag

  feed(chunk: string): StreamEvent[] {
    const events: StreamEvent[] = [];
    for (const c of chunk) {
      if (this.inTag) {
        this.tagBuf += c;
        if (this.tagBuf.endsWith(CLOSE)) {
          const payload = this.tagBuf.slice(0, -CLOSE.length);
          try {
            events.push({ type: 'action', value: JSON.parse(payload) as Record<string, unknown> });
          } catch {
            events.push({ type: 'text', value: OPEN + payload + CLOSE });
          }
          this.tagBuf = '';
          this.inTag  = false;
        }
      } else {
        this.buf += c;
        if (this.buf.endsWith(OPEN)) {
          const pre = this.buf.slice(0, -OPEN.length);
          if (pre) events.push({ type: 'text', value: pre });
          this.buf   = '';
          this.inTag = true;
        }
        // Don't emit char-by-char — bulk-flush after the loop instead.
      }
    }

    // After consuming the whole chunk, emit everything that provably can't be
    // the start of an OPEN tag. We only need to keep OPEN.length-1 chars as
    // lookahead (a complete OPEN match would already have been caught above).
    if (!this.inTag && this.buf.length > OPEN.length - 1) {
      const safe = this.buf.length - (OPEN.length - 1);
      events.push({ type: 'text', value: this.buf.slice(0, safe) });
      this.buf = this.buf.slice(safe);
    }

    return events;
  }

  flush(): StreamEvent[] {
    const events: StreamEvent[] = [];
    if (this.buf)                        events.push({ type: 'text', value: this.buf });
    if (this.inTag && this.tagBuf)       events.push({ type: 'text', value: OPEN + this.tagBuf });
    this.buf    = '';
    this.tagBuf = '';
    this.inTag  = false;
    return events;
  }
}
