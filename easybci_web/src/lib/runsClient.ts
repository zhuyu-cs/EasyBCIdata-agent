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
  // tool.started / tool.completed: pairs a result with its call.
  tool_call_id?: string;
  // tool.completed: head of the result and its original length.
  result_preview?: string;
  result_chars?: number;
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

// No conversation_history is sent: the gateway restores the full stored
// transcript (tool calls and results included) from the session id.
export async function startRun(input: string, sessionId?: string): Promise<string> {
  const body: Record<string, unknown> = { input };
  if (sessionId) body.session_id = sessionId;

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

export interface RunStatusInfo {
  run_id: string;
  status: string;
  session_id?: string;
  error?: string;
  output?: string;
}

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

/** Pollable run status. Resolves null when the gateway no longer knows the
 *  run (404), throws on transport errors so callers can keep retrying. */
export async function getRunStatus(runId: string): Promise<RunStatusInfo | null> {
  const res = await fetch(`${GATEWAY_BASE}/runs/${runId}`, { headers: getAuthHeaders() });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`run status ${res.status}`);
  return (await res.json()) as RunStatusInfo;
}

export function isLiveRunStatus(status: string | undefined): boolean {
  return !!status && !TERMINAL_STATUSES.has(status);
}

// After a few failed reconnects, ask the gateway whether the run still exists.
// A vanished run is a hard failure; a run that finished while we were away is
// closed with a synthetic gap + terminal event so the caller reconciles
// against the stored transcript instead of retrying forever.
const PROBE_AFTER_ATTEMPTS = 3;

type ProbeOutcome = "retry" | "handled";

async function probeDroppedRun(
  runId: string,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): Promise<ProbeOutcome> {
  let info: RunStatusInfo | null;
  try {
    info = await getRunStatus(runId);
  } catch {
    return "retry";
  }
  if (info === null) {
    onError(new Error("The run is no longer available on the gateway."));
    return "handled";
  }
  if (TERMINAL_STATUSES.has(info.status)) {
    const ts = Date.now() / 1000;
    onEvent({ event: "stream.gap", run_id: runId, timestamp: ts });
    onEvent({
      event: `run.${info.status}`,
      run_id: runId,
      timestamp: ts,
      ...(info.error ? { error: info.error } : {}),
    });
    onDone();
    return "handled";
  }
  return "retry";
}

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
    return subscribeWithFetchRetry(runId, url, authHeaders, onEvent, onDone, onError, options);
  }

  return subscribeWithEventSourceRetry(runId, url, onEvent, onDone, onError, options);
}

function subscribeWithEventSourceRetry(
  runId: string,
  url: string,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
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
      scheduleRetry();
    };
  }

  async function scheduleRetry() {
    if (attempt >= PROBE_AFTER_ATTEMPTS) {
      const outcome = await probeDroppedRun(runId, onEvent, onDone, (err) => {
        status?.({ kind: "failed", reason: err.message });
        onError(err);
      });
      if (cancelled || outcome === "handled") return;
    }
    const climb = Math.min(attempt, BACKOFF_CAP_ATTEMPTS);
    const delay = Math.min(Math.pow(2, climb - 1) * 1000, MAX_BACKOFF_MS);
    retryTimer = setTimeout(connect, delay);
  }

  connect();

  return () => {
    cancelled = true;
    source?.close();
    if (retryTimer) clearTimeout(retryTimer);
  };
}

function subscribeWithFetchRetry(
  runId: string,
  url: string,
  headers: Record<string, string>,
  onEvent: (ev: RunEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
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
      // A clean EOF without a terminal event is a dropped stream, not a
      // completion — fall through to the reconnect path.
      throw new Error("SSE stream ended without a terminal event");
    } catch (err) {
      if ((err as Error).name === "AbortError" || cancelled) return;

      // Same policy as the EventSource path: keep reconnecting while the run
      // is alive; the probe ends the stream once it is gone or finished.
      attempt++;
      status?.({ kind: "reconnecting", attempt });
      if (attempt >= PROBE_AFTER_ATTEMPTS) {
        const outcome = await probeDroppedRun(runId, onEvent, onDone, (e) => {
          status?.({ kind: "failed", reason: e.message });
          onError(e);
        });
        if (cancelled || outcome === "handled") return;
      }
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
