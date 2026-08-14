import { warn } from "./debug";

export interface RunEvent {
  event: string;
  run_id: string;
  timestamp: number;
  delta?: string;
  tool?: string;
  preview?: string;
  text?: string;
  duration?: number;
  error?: boolean | string;
  output?: string;
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
  command?: string;
  description?: string;
  choices?: string[];
  // Structured path-like args forwarded from tool.started (B8). Keys are a
  // whitelist of arg names (path, output_path, file, work_dir, …).
  paths?: Record<string, string>;
  // Optional event id supplied by the server when SSE replay is supported.
  // Used as `Last-Event-ID` on reconnect for idempotent resume.
  id?: string;
  // stream.gap event: emitted on reconnect when buffered events between the
  // client's cursor and the retained buffer were evicted (P1-1). Signals the
  // client to reconcile against the authoritative transcript.
  from_seq?: number;
  to_seq?: number;
  // Free-form metadata bag — used by `progress` events to carry the
  // StageProgress payload (`metadata.progress`). Other event types may
  // attach arbitrary structured data here in the future.
  metadata?: Record<string, unknown>;
}

export type StreamStatus =
  | { kind: "connecting" }
  | { kind: "connected" }
  | { kind: "reconnecting"; attempt: number; max?: number }
  | { kind: "failed"; reason: string };

export interface SubscribeOptions {
  /** Optional status callback for UI feedback (banner, dot). */
  onStatus?: (status: StreamStatus) => void;
  /** Last seen event id — used to ask the server to resume from there. */
  lastEventId?: string | null;
}

const GATEWAY_BASE = "/v1";
// Backoff climbs 2^(attempt-1)*1s until this cap, then stays flat. We retry
// indefinitely while the run has not delivered a terminal event — a dropped
// SSE connection must never be treated as a clean end (that was the frequent
// "disconnect" bug). Only a server-sent run.failed/cancelled ends the stream.
const BACKOFF_CAP_ATTEMPTS = 5; // 2^4 * 1000ms = 16s ceiling
const MAX_BACKOFF_MS = 20_000;

function getAuthHeaders(): Record<string, string> {
  const key = import.meta.env?.VITE_API_SERVER_KEY as string | undefined;
  if (key) return { Authorization: `Bearer ${key}` };
  return {};
}

export async function startRun(
  input: string,
  sessionId?: string,
  conversationHistory?: Array<{ role: string; content: string }>,
): Promise<string> {
  const body: Record<string, unknown> = { input };
  if (sessionId) body.session_id = sessionId;
  if (conversationHistory?.length) body.conversation_history = conversationHistory;

  const res = await fetch(`${GATEWAY_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Failed to start run: ${res.status} ${text}`);
  }
  const data = await res.json();
  return data.run_id;
}

const TERMINAL_EVENTS = new Set(["run.completed", "run.failed", "run.cancelled"]);

export function subscribeToRun(
  runId: string,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
  options: SubscribeOptions = {},
): () => void {
  const url = `${GATEWAY_BASE}/runs/${runId}/events`;
  const authHeaders = getAuthHeaders();

  if (Object.keys(authHeaders).length > 0) {
    return subscribeWithFetchRetry(url, authHeaders, onEvent, onDone, onError, options);
  }

  return subscribeWithEventSourceRetry(url, onEvent, onDone, onError, options);
}

function subscribeWithEventSourceRetry(
  url: string,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  // Retained for signature symmetry with subscribeToRun; unused because we now
  // retry indefinitely and never surface a hard "failed" error.
  _onError: (err: Error) => void,
  options: SubscribeOptions,
): () => void {
  let attempt = 0;
  let source: EventSource | null = null;
  let cancelled = false;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let lastEventId: string | null = options.lastEventId ?? null;
  const status = options.onStatus;

  function connect() {
    if (cancelled) return;
    status?.(attempt === 0 ? { kind: "connecting" } : { kind: "reconnecting", attempt });
    // EventSource has no standard way to carry Last-Event-ID on initial connect;
    // pass via query string for backends that support it (no-op otherwise).
    const fullUrl = lastEventId ? `${url}?last_event_id=${encodeURIComponent(lastEventId)}` : url;
    source = new EventSource(fullUrl);

    source.onopen = () => {
      attempt = 0;
      status?.({ kind: "connected" });
    };

    source.onmessage = (e) => {
      attempt = 0;
      try {
        const data = JSON.parse(e.data) as RunEvent;
        if (e.lastEventId) lastEventId = e.lastEventId;
        else if (data.id) lastEventId = data.id;
        onEvent(data);
        if (TERMINAL_EVENTS.has(data.event)) {
          source?.close();
          onDone();
        }
      } catch (err) { warn("sse", "dropped malformed event frame", err); }
    };

    source.onerror = () => {
      source?.close();
      if (cancelled) return;

      // Do NOT treat a closed connection as a clean completion. Only a
      // server-sent terminal event (handled in onmessage) calls onDone().
      // Any error/close here means the transport dropped — reconnect and let
      // Last-Event-ID replay fill the gap.
      attempt++;
      status?.({ kind: "reconnecting", attempt });
      const climb = Math.min(attempt, BACKOFF_CAP_ATTEMPTS);
      const delay = Math.min(Math.pow(2, climb - 1) * 1000, MAX_BACKOFF_MS);
      retryTimer = setTimeout(connect, delay);
    };
  }

  connect();

  return () => {
    cancelled = true;
    source?.close();
    if (retryTimer) clearTimeout(retryTimer);
  };
}

function subscribeWithFetchRetry(
  url: string,
  headers: Record<string, string>,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  // Retained for signature symmetry with subscribeToRun; unused because we now
  // retry indefinitely and never surface a hard "failed" error.
  _onError: (err: Error) => void,
  options: SubscribeOptions,
): () => void {
  let attempt = 0;
  let cancelled = false;
  let controller: AbortController | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let lastEventId: string | null = options.lastEventId ?? null;
  const status = options.onStatus;

  async function connect() {
    if (cancelled) return;
    status?.(attempt === 0 ? { kind: "connecting" } : { kind: "reconnecting", attempt });
    controller = new AbortController();

    try {
      const reqHeaders: Record<string, string> = {
        Accept: "text/event-stream",
        ...headers,
      };
      if (lastEventId) reqHeaders["Last-Event-ID"] = lastEventId;
      const res = await fetch(url, { headers: reqHeaders, signal: controller.signal });
      if (!res.ok || !res.body) {
        throw new Error(`SSE fetch failed: ${res.status}`);
      }

      attempt = 0;
      status?.({ kind: "connected" });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let pendingId: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("id: ")) {
            pendingId = line.slice(4).trim();
            continue;
          }
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6)) as RunEvent;
              if (pendingId) {
                lastEventId = pendingId;
                pendingId = null;
              } else if (data.id) {
                lastEventId = data.id;
              }
              onEvent(data);
              if (TERMINAL_EVENTS.has(data.event)) {
                onDone();
                return;
              }
            } catch (err) { warn("sse", "dropped malformed event frame (fetch)", err); }
          }
        }
      }
      onDone();
    } catch (err) {
      if ((err as Error).name === "AbortError" || cancelled) return;

      // Same policy as the EventSource path: retry indefinitely until a
      // server terminal event ends the stream. Never surface a hard "failed".
      attempt++;
      status?.({ kind: "reconnecting", attempt });
      const climb = Math.min(attempt, BACKOFF_CAP_ATTEMPTS);
      const delay = Math.min(Math.pow(2, climb - 1) * 1000, MAX_BACKOFF_MS);
      retryTimer = setTimeout(connect, delay);
    }
  }

  connect();

  return () => {
    cancelled = true;
    controller?.abort();
    if (retryTimer) clearTimeout(retryTimer);
  };
}

export async function stopRun(runId: string): Promise<{ stopped: boolean }> {
  try {
    const res = await fetch(`${GATEWAY_BASE}/runs/${runId}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    });
    if (!res.ok) return { stopped: false };
    const data = await res.json().catch(() => ({}));
    // Older gateways don't return `stopped`; a 2xx there means it was found
    // and interrupted, so default to true for back-compat.
    return { stopped: data?.stopped ?? true };
  } catch (err) {
    // Network error — we can't confirm a stop; report false so callers know
    // the run may still be live.
    warn("stopRun", `failed to confirm stop for ${runId}`, err);
    return { stopped: false };
  }
}
