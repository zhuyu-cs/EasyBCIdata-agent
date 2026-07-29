import { useEffect, useState, useCallback, useRef } from "react";
import { api, type SessionMessage } from "@/lib/api";
import { startRun, subscribeToRun, stopRun, type RunEvent, type StreamStatus } from "@/lib/runsClient";
import type { StageProgress } from "@/components/StageProgressBar";
import { applyEwma, type TurnEta } from "@/lib/turnEta";
import { useSessionStore } from "@/stores/sessionStore";
import {
  enqueuePendingMessage,
  listPendingMessages,
  removePendingMessage,
  bumpPendingAttempt,
  clearAllPendingMessages,
  type PendingMessage,
} from "@/lib/offlineCache";
import { useToastStore } from "@/stores/toastStore";
import { warn } from "@/lib/debug";

export interface ToolCall {
  tool: string;
  preview?: string;
  status: "running" | "done" | "error";
  duration?: number;
  reasoning?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  thinking?: string;
  toolCalls?: ToolCall[];
  pendingReasoning?: string;
  isNew?: boolean;
}

function findToolResult(
  all: SessionMessage[],
  assistantIdx: number,
  toolCallId: string,
): SessionMessage | undefined {
  for (let i = assistantIdx + 1; i < all.length; i++) {
    const m = all[i];
    // The next assistant message bounds the lookup window — any tool result
    // belonging to this assistant turn must appear before it.
    if (m.role === "assistant") return undefined;
    if (m.role === "tool" && m.tool_call_id === toolCallId) return m;
  }
  return undefined;
}

function transformApiMessage(
  msg: SessionMessage,
  idx: number,
  allMessages: SessionMessage[],
): Message | null {
  if (msg.role !== "user" && msg.role !== "assistant") return null;
  const result: Message = {
    id: `hist-${idx}`,
    role: msg.role,
    content: msg.content ?? "",
    timestamp: msg.timestamp ?? Date.now() / 1000,
  };
  if (msg.role === "assistant") {
    const reasoning = msg.reasoning ?? msg.reasoning_content ?? undefined;
    if (reasoning) result.thinking = reasoning;
  }
  if (msg.tool_calls?.length) {
    result.toolCalls = msg.tool_calls.map((tc) => {
      const toolMsg = findToolResult(allMessages, idx, tc.id);
      // Legacy rows (pre-Phase-1) have no tool_status — fall back to "done".
      const status = (toolMsg?.tool_status as ToolCall["status"] | undefined) ?? "done";
      return {
        tool: tc.function.name,
        preview: tryParsePreview(tc.function.arguments),
        status,
        duration: toolMsg?.tool_duration ?? undefined,
        reasoning: toolMsg?.content ?? undefined,
      };
    });
  }
  return result;
}

function tryParsePreview(args: string): string | undefined {
  try {
    const parsed = JSON.parse(args);
    return parsed.preview ?? parsed.path ?? undefined;
  } catch {
    return undefined;
  }
}

export type OnRunCompleteCallback = () => void;

export interface ApprovalRequest {
  runId: string;
  command?: string;
  description?: string;
  choices: string[];
}

export function useConversation() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approvalRequest, setApprovalRequest] = useState<ApprovalRequest | null>(null);
  // Live status of the SSE stream — null when no run is active.
  const [streamStatus, setStreamStatus] = useState<StreamStatus | null>(null);
  // Coarse run lifecycle status used by StepCard's false-fail filtering (Phase
  // 4). "running" while a run streams; "failed" only when the gateway emits
  // run.failed (or the SSE errors out); "completed" on run.completed/done;
  // "unknown" before any run / after reloading history we can't classify.
  // classifyCalls treats anything but "failed" as "never show terminal red".
  const [runStatus, setRunStatus] = useState<"running" | "completed" | "failed" | "unknown">("unknown");
  // Latest StageProgress from the gateway (T1.6). null when no run is active
  // or the run hasn't emitted a progress event yet. Reset to null on every
  // run lifecycle transition so the bar disappears between turns.
  const [progress, setProgress] = useState<StageProgress | null>(null);
  // Turn-scope ETA state. Distinct from `progress` which is
  // stage-scope. Updated by `progress` SSE events with scope === "turn";
  // cleared on every run lifecycle transition.
  const [latestTurnEta, setLatestTurnEta] = useState<TurnEta | null>(null);
  // ms since epoch when the active turn began — set when `send()` posts the
  // user message, cleared on every lifecycle transition.
  const [turnStartedAtMs, setTurnStartedAtMs] = useState<number | null>(null);
  // True when the backend has a newer version than what we last loaded.
  const [externalUpdateAvailable, setExternalUpdateAvailable] = useState(false);
  const activeRunRef = useRef<string | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);
  const workspaceCallbackRef = useRef<((ev: RunEvent) => void) | null>(null);
  const onRunCompleteRef = useRef<OnRunCompleteCallback | null>(null);
  // Latest messages mirror — avoids stale-closure issues in `send`/`interrupt`
  // and lets effects observe current state without resubscribing.
  const messagesRef = useRef<Message[]>([]);
  // Mirror of activeSessionId so callbacks don't need it as a dep.
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  // Set of seen event ids per active run — used to drop replays after reconnect.
  const seenEventIdsRef = useRef<Set<string>>(new Set());
  // Set true when the server reports a stream.gap (evicted events on reconnect,
  // P1-1). On run completion we reconcile against the authoritative transcript
  // instead of trusting our locally-accumulated (now hole-y) message.
  const gapDetectedRef = useRef(false);
  // Last seen version stamp from the backend — used by external-update polling.
  const sessionVersionRef = useRef<string | null>(null);
  // Mirror of isStreaming so the polling effect can early-exit without resubscribing.
  const isStreamingRef = useRef(false);
  // Coalesce high-frequency message.delta / reasoning.available events into one
  // setState per animation frame, so a long token stream doesn't trigger a React
  // reconcile per token (U2). Buffers accumulate between flushes.
  const deltaBufferRef = useRef("");
  const reasoningBufferRef = useRef("");
  const rafRef = useRef<number | null>(null);
  // Pending message queue surfaced to the UI (for offline indicator + manual flush).
  const [pendingMessages, setPendingMessages] = useState<PendingMessage[]>(() =>
    typeof window !== "undefined" ? listPendingMessages() : [],
  );

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Reset per-session UI state the moment activeSessionId changes — done during
  // render via the prev-value pattern so the fetch effect below only performs
  // the async load (no synchronous setState inside an effect).
  const [prevSessionId, setPrevSessionId] = useState(activeSessionId);
  if (activeSessionId !== prevSessionId) {
    setPrevSessionId(activeSessionId);
    setError(null);
    setExternalUpdateAvailable(false);
    // Reloaded history can't be reliably classified as failed (the messages
    // endpoint carries no run-level status), so default to "unknown" →
    // classifyCalls treats every error as grey, never terminal-red
    // (conservative default).
    setRunStatus("unknown");
    if (!activeSessionId) setMessages([]);
  }

  useEffect(() => {
    if (!activeSessionId) {
      sessionVersionRef.current = null;
      return;
    }
    let cancelled = false;
    sessionVersionRef.current = null;
    api
      .getSessionMessages(activeSessionId)
      .then((data) => {
        if (cancelled) return;
        const transformed = data.messages
          .map((m, i) => transformApiMessage(m, i, data.messages))
          .filter((m): m is Message => m !== null);
        setMessages(transformed);
        setError(null);
        sessionVersionRef.current = data.version ?? null;
      })
      .catch((e) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        // A freshly-created session has no backing DB row until the first
        // assistant turn lands — `/api/sessions/{id}/messages` returns 404.
        // That's not an error; just silence it. Crucially we MUST NOT clear
        // the in-memory message list here: the caller may have just pushed
        // a user message via `send()` whose `setActiveSessionId(newId)` is
        // what triggered this effect — clearing would lose the user's bubble.
        if (/^404\b/.test(msg) || /session not found/i.test(msg)) {
          setError(null);
          sessionVersionRef.current = null;
          return;
        }
        setError(msg);
      });
    return () => { cancelled = true; };
  }, [activeSessionId]);

  // Reload the active session's messages from the backend — used by the
  // "Session updated externally" banner and as a post-stream correction.
  // `reloadCount` increments on every successful reload so consumers (e.g.
  // App.tsx's artifacts effect) can re-fetch dependent data.
  const [reloadCount, setReloadCount] = useState(0);
  const reloadSession = useCallback(async () => {
    const sid = activeSessionIdRef.current;
    if (!sid) return;
    try {
      const data = await api.getSessionMessages(sid);
      const transformed = data.messages
        .map((m, i) => transformApiMessage(m, i, data.messages))
        .filter((m): m is Message => m !== null);
      setMessages(transformed);
      sessionVersionRef.current = data.version ?? null;
      setExternalUpdateAvailable(false);
      setReloadCount((c) => c + 1);
    } catch (err) {
      warn("reloadSession", "failed to reload messages; banner stays for retry", err);
    }
  }, []);

  // Poll backend version while idle (not streaming). Detects CLI-side or
  // external mutations to the active session. Uses a conditional request
  // (If-None-Match → 304) so an unchanged session costs almost nothing, and
  // backs the interval off while idle (10s → 30s → 60s) so a quiet tab isn't
  // hammering the dashboard (B7).
  useEffect(() => {
    if (!activeSessionId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const INTERVALS = [10_000, 30_000, 60_000];
    let idleStreak = 0;

    const schedule = () => {
      if (cancelled) return;
      const delay = INTERVALS[Math.min(idleStreak, INTERVALS.length - 1)];
      timer = setTimeout(tick, delay);
    };

    const tick = async () => {
      if (cancelled) return;
      // Skip while streaming — the run itself is the authoritative writer.
      // Reset the backoff so we poll promptly once streaming ends.
      if (isStreamingRef.current) {
        idleStreak = 0;
        schedule();
        return;
      }
      try {
        const res = await api.getSessionVersionConditional(
          activeSessionId,
          sessionVersionRef.current,
        );
        if (cancelled) return;
        if (res.notModified) {
          // No change — lengthen the interval.
          idleStreak += 1;
        } else {
          const known = sessionVersionRef.current;
          if (known && res.data.version !== known) {
            setExternalUpdateAvailable(true);
          }
          // A response (even unchanged content with no prior known version)
          // resets the cadence so we react quickly to the next edit.
          idleStreak = 0;
        }
      } catch (err) {
        warn("versionPoll", "version probe failed; keeping current cadence", err);
      }
      schedule();
    };

    schedule();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeSessionId]);

  // Drain the delta/reasoning buffers into the last assistant message. Called on
  // an animation frame so bursts of tokens coalesce into a single setState.
  const flushBuffers = useCallback(() => {
    rafRef.current = null;
    const deltaChunk = deltaBufferRef.current;
    const reasoningChunk = reasoningBufferRef.current;
    if (!deltaChunk && !reasoningChunk) return;
    deltaBufferRef.current = "";
    reasoningBufferRef.current = "";
    setMessages((prev) => {
      const msgs = [...prev];
      const lastIdx = msgs.length - 1;
      if (lastIdx < 0 || msgs[lastIdx].role !== "assistant") return prev;
      const last = { ...msgs[lastIdx] };
      if (deltaChunk && !last.content.endsWith(deltaChunk)) {
        last.content += deltaChunk;
      }
      if (reasoningChunk && (!last.thinking || !last.thinking.endsWith(reasoningChunk))) {
        last.thinking = (last.thinking ?? "") + reasoningChunk;
        last.pendingReasoning = (last.pendingReasoning ?? "") + reasoningChunk;
      }
      msgs[lastIdx] = last;
      return msgs;
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafRef.current != null) return;
    if (typeof requestAnimationFrame === "function") {
      rafRef.current = requestAnimationFrame(() => flushBuffers());
    } else {
      // Fallback for non-DOM environments (tests): flush synchronously-ish.
      rafRef.current = setTimeout(() => flushBuffers(), 16) as unknown as number;
    }
  }, [flushBuffers]);

  // Cancel any scheduled flush and DISCARD buffered tokens without draining
  // them. Used when tearing down a stream whose remaining text must not land in
  // the (possibly different) currently-rendered message — e.g. on session
  // switch, where flushing would append the old run's trailing tokens onto the
  // newly-loaded session's last assistant message (P0-3).
  const cancelPendingFlush = useCallback(() => {
    if (rafRef.current != null) {
      if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(rafRef.current);
      else clearTimeout(rafRef.current as unknown as ReturnType<typeof setTimeout>);
      rafRef.current = null;
    }
    deltaBufferRef.current = "";
    reasoningBufferRef.current = "";
  }, []);

  const send = useCallback(
    async (text: string, opts?: { rethrowOnFailure?: boolean }) => {
      // Auto-interrupt any prior in-flight run before starting a new one (C10).
      if (activeRunRef.current) {
        try { await stopRun(activeRunRef.current); } catch { /* best-effort */ }
        unsubRef.current?.();
        unsubRef.current = null;
        activeRunRef.current = null;
      }
      setError(null);
      setStreamStatus({ kind: "connecting" });
      seenEventIdsRef.current = new Set();
      gapDetectedRef.current = false;
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: Date.now() / 1000,
        isNew: true,
      };
      // Build multi-turn history off a SYNCHRONOUSLY-updated mirror, not the
      // effect-synced messagesRef which lags state by a render. Two rapid sends
      // could otherwise build history from a snapshot missing the first send's
      // messages and ship truncated context to the LLM (P2-4). We advance the
      // mirror here so the second send sees the first's appends immediately.
      const hasSession = !!useSessionStore.getState().activeSessionId;
      const base = hasSession ? messagesRef.current : [];
      const withUser = [...base, userMsg];
      messagesRef.current = withUser;
      const history = withUser.map((m) => ({ role: m.role, content: m.content }));
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setRunStatus("running");
      // Anchor the turn-elapsed counter for TurnEtaIndicator.
      setTurnStartedAtMs(Date.now());
      setLatestTurnEta(null);

      try {
        // Determine session ID — use existing or generate new
        let sessionId = activeSessionIdRef.current;
        if (!sessionId) {
          sessionId = crypto.randomUUID();
          useSessionStore.getState().setActiveSessionId(sessionId);
        }

        const runId = await startRun(text, sessionId, history);
        activeRunRef.current = runId;
        // Backend writes a heuristic title (truncated first user message) at
        // the start of the run; refresh the sidebar shortly after so the
        // session row picks it up instead of staying on "Untitled Session"
        // until the full pipeline completes. The post-run handleDone refresh
        // (below) still catches the LLM-refined title at end-of-run.
        try {
          const _bump = useSessionStore.getState().triggerRefresh;
          setTimeout(_bump, 400);
        } catch { /* best-effort */ }

        const assistantId = crypto.randomUUID();
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: "assistant", content: "", timestamp: Date.now() / 1000, toolCalls: [], isNew: true },
        ]);

        const handleDone = () => {
          // Flush any buffered trailing tokens before tearing down.
          if (rafRef.current != null) {
            if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(rafRef.current);
            else clearTimeout(rafRef.current as unknown as ReturnType<typeof setTimeout>);
            rafRef.current = null;
          }
          flushBuffers();
          setIsStreaming(false);
          setStreamStatus(null);
          activeRunRef.current = null;
          // The run is over — any approval prompt for it is now moot. Leaving it
          // up would block the UI with a dialog that POSTs to a dead run (P0-2).
          setApprovalRequest(null);
          onRunCompleteRef.current?.();
          // Refresh the sidebar session list so an auto-generated title shows
          // up. Title generation is fire-and-forget on the backend (extra LLM
          // call after the main reply), so a short delay reduces a no-op fetch
          // before the title row updates. Two-stage refresh (250ms + 2.5s)
          // covers both cached and slow title-gen paths.
          try {
            const _bump = useSessionStore.getState().triggerRefresh;
            setTimeout(_bump, 250);
            setTimeout(_bump, 2500);
          } catch { /* best-effort */ }
          // A reconnect dropped events the server couldn't replay (P1-1): our
          // in-memory message has a hole. Pull the authoritative transcript so
          // the rendered conversation matches what the backend actually stored.
          if (gapDetectedRef.current) {
            gapDetectedRef.current = false;
            reloadSession();
          }
          // Refresh the version stamp so the polling effect doesn't fire a
          // false "external update" banner from our own write.
          const sid = activeSessionIdRef.current;
          if (sid) {
            api
              .getSessionVersion(sid)
              .then((v) => {
                sessionVersionRef.current = v.version;
              })
              .catch((err) => { warn("versionRefresh", "post-run version refresh failed", err); });
          }
        };

        unsubRef.current = subscribeToRun(
          runId,
          (ev: RunEvent) => {
            // Drop replays after reconnect — same event id seen twice.
            if (ev.id) {
              if (seenEventIdsRef.current.has(ev.id)) return;
              seenEventIdsRef.current.add(ev.id);
            }
            workspaceCallbackRef.current?.(ev);

            // Reconnect left a hole the server couldn't replay (P1-1). Mark it;
            // handleDone will reload the authoritative transcript so the user
            // never sees a silently truncated message.
            if (ev.event === "stream.gap") {
              gapDetectedRef.current = true;
              return;
            }

            // StageProgress payload lives on a distinct
            // `progress` SSE event so it doesn't pollute message.delta /
            // tool.* dispatch.  Always replace; the gateway only emits an
            // event when something actually changed.
            if (ev.event === "progress") {
              const payload = ev.metadata?.progress as
                | (StageProgress & { source?: string; next_intent_kind?: string })
                | undefined;
              if (!payload) return;
              const scope = payload.scope ?? "stage";
              if (scope === "turn") {
                // Turn-scope ETA: EWMA-smooth and store, do NOT
                // overwrite `progress` (which carries stage-scope data).
                const newEta = (payload as { eta_seconds?: number | null }).eta_seconds;
                const conf = (payload as { confidence?: StageProgress["confidence"] }).confidence ?? "unknown";
                if (typeof newEta === "number" && newEta > 0) {
                  setLatestTurnEta((prev) => ({
                    smoothedSeconds: Math.max(240, applyEwma(newEta, prev?.smoothedSeconds ?? null)),
                    emittedAtMs: Date.now(),
                    confidence: conf,
                  }));
                }
                return;
              }
              // scope === "stage" — legacy path
              setProgress(payload as StageProgress);
              return;
            }

            if (ev.event === "approval.request") {
              setApprovalRequest({
                runId: ev.run_id,
                command: ev.command,
                description: ev.description,
                choices: ev.choices ?? ["once", "session", "always", "deny"],
              });
              return;
            }

            // High-frequency text events: buffer and flush on a frame (U2).
            if (ev.event === "message.delta" && ev.delta) {
              deltaBufferRef.current += ev.delta;
              scheduleFlush();
              return;
            }
            if (ev.event === "reasoning.available" && ev.text) {
              reasoningBufferRef.current += ev.text;
              scheduleFlush();
              return;
            }

            // Structural events (tool start/complete, run output): flush any
            // buffered text first so ordering is preserved, then apply now.
            if (deltaBufferRef.current || reasoningBufferRef.current) {
              if (rafRef.current != null) {
                if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(rafRef.current);
                else clearTimeout(rafRef.current as unknown as ReturnType<typeof setTimeout>);
                rafRef.current = null;
              }
              flushBuffers();
            }

            setMessages((prev) => {
              const msgs = [...prev];
              const lastIdx = msgs.length - 1;
              if (lastIdx < 0 || msgs[lastIdx].role !== "assistant") return msgs;
              const last = { ...msgs[lastIdx] };

              if (ev.event === "tool.started") {
                last.toolCalls = [
                  ...(last.toolCalls ?? []),
                  {
                    tool: ev.tool ?? "unknown",
                    preview: ev.preview,
                    status: "running",
                    reasoning: last.pendingReasoning || undefined,
                  },
                ];
                last.pendingReasoning = "";
              } else if (ev.event === "tool.completed") {
                const calls = [...(last.toolCalls ?? [])];
                const idx = calls.findLastIndex((c) => c.tool === ev.tool && c.status === "running");
                if (idx >= 0) {
                  calls[idx] = { ...calls[idx], status: ev.error ? "error" : "done", duration: ev.duration };
                }
                last.toolCalls = calls;
              } else if (ev.event === "run.completed" && ev.output) {
                if (!last.content.includes(ev.output.slice(0, 50))) {
                  last.content += ev.output;
                }
              }

              msgs[lastIdx] = last;
              return msgs;
            });

            // Track coarse run lifecycle for false-fail filtering.
            // run.failed is the ONLY signal that lets StepCard show terminal
            // red; run.completed / run.cancelled resolve to non-failed.
            if (ev.event === "run.failed") {
              setRunStatus("failed");
              setProgress(null);
              setLatestTurnEta(null);
              setTurnStartedAtMs(null);
            } else if (ev.event === "run.completed") {
              setRunStatus("completed");
              setProgress(null);
              setLatestTurnEta(null);
              setTurnStartedAtMs(null);
            } else if (ev.event === "run.cancelled") {
              setRunStatus("completed");
              setProgress(null);
              setLatestTurnEta(null);
              setTurnStartedAtMs(null);
            }
          },
          handleDone,
          (err) => {
            setIsStreaming(false);
            setRunStatus("failed");
            setProgress(null);
            setLatestTurnEta(null);
            setTurnStartedAtMs(null);
            activeRunRef.current = null;
            setApprovalRequest(null);
            setError(err.message);
            // Leave streamStatus on the failed kind so the banner persists
            // until the user retries or starts a new run.
          },
          {
            onStatus: (s) => setStreamStatus(s),
          },
        );
      } catch (e) {
        setIsStreaming(false);
        setStreamStatus(null);
        const msg = e instanceof Error ? e.message : "Failed to send message";
        // When replaying from the pending queue, let the caller own retry
        // bookkeeping (attempt count / drop) instead of enqueuing a duplicate.
        if (opts?.rethrowOnFailure) {
          setError(msg);
          throw e instanceof Error ? e : new Error(msg);
        }
        // Gateway unreachable — queue the message so it can be replayed later.
        const queued = enqueuePendingMessage({
          text,
          sessionId: activeSessionIdRef.current,
        });
        setPendingMessages((prev) => [...prev, queued]);
        setError(`${msg} (queued for retry when connection returns)`);
      }
    },
    [flushBuffers, scheduleFlush, reloadSession],
  );

  // Replay pending queue when the network/gateway comes back online.
  const flushPending = useCallback(async () => {
    const items = listPendingMessages();
    if (items.length === 0) return;
    // Replay sequentially so multi-turn ordering is preserved.
    for (const item of items) {
      // Strict sessionId targeting: switch to the message's own session before
      // replaying so a queued message never lands in whatever session happens
      // to be active now (B4). Messages with no recorded session fall back to
      // the current one (they were composed before a session existed).
      const sid = item.sessionId ?? activeSessionIdRef.current;
      if (item.sessionId && item.sessionId !== activeSessionIdRef.current) {
        useSessionStore.getState().setActiveSessionId(item.sessionId);
      } else if (!activeSessionIdRef.current && sid) {
        useSessionStore.getState().setActiveSessionId(sid);
      }
      try {
        // Issue a fresh send (auto-interrupts current run if any). Rethrow on
        // failure so we record the attempt rather than silently re-enqueueing.
        await send(item.text, { rethrowOnFailure: true });
        removePendingMessage(item.id);
        setPendingMessages((prev) => prev.filter((m) => m.id !== item.id));
        // Wait for run to finish before queuing next — basic gate.
        await new Promise<void>((resolve) => {
          const start = Date.now();
          const t = setInterval(() => {
            if (!isStreamingRef.current || Date.now() - start > 60_000) {
              clearInterval(t);
              resolve();
            }
          }, 200);
        });
      } catch {
        // Failed again — record the attempt. Drop after the cap so a
        // permanently-failing message can't wedge the queue forever (B4).
        const dropped = bumpPendingAttempt(item.id);
        if (dropped) {
          setPendingMessages((prev) => prev.filter((m) => m.id !== item.id));
          useToastStore
            .getState()
            .add("A queued message could not be delivered and was discarded.", "error");
        }
        // Stop this pass; remaining items retry on the next online event.
        return;
      }
    }
  }, [send]);

  useEffect(() => {
    const onOnline = () => { flushPending(); };
    window.addEventListener("online", onOnline);
    return () => window.removeEventListener("online", onOnline);
  }, [flushPending]);

  const interrupt = useCallback(() => {
    if (activeRunRef.current) {
      stopRun(activeRunRef.current);
      unsubRef.current?.();
      unsubRef.current = null;
      // Flush any buffered tokens captured before the stop.
      if (rafRef.current != null) {
        if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(rafRef.current);
        else clearTimeout(rafRef.current as unknown as ReturnType<typeof setTimeout>);
        rafRef.current = null;
      }
      flushBuffers();
      setIsStreaming(false);
      setStreamStatus(null);
      setApprovalRequest(null);
      activeRunRef.current = null;
    }
  }, [flushBuffers]);

  // Auto-interrupt streaming when the user switches sessions (C10).
  useEffect(() => {
    if (activeRunRef.current) {
      stopRun(activeRunRef.current).catch(() => { /* best-effort */ });
      unsubRef.current?.();
      unsubRef.current = null;
      setIsStreaming(false);
      setStreamStatus(null);
      activeRunRef.current = null;
    }
    // Discard (do NOT flush) any in-flight buffers and pending frame: flushing
    // would graft the old run's trailing tokens onto the new session's last
    // assistant message. Also clear the approval prompt and the dedup set so
    // the old run's state can't bleed into the newly-selected session (P0-2/P0-3).
    cancelPendingFlush();
    seenEventIdsRef.current.clear();
    setApprovalRequest(null);
  }, [activeSessionId, cancelPendingFlush]);

  const setWorkspaceCallback = useCallback((cb: ((ev: RunEvent) => void) | null) => {
    workspaceCallbackRef.current = cb;
  }, []);

  const setOnRunComplete = useCallback((cb: OnRunCompleteCallback | null) => {
    onRunCompleteRef.current = cb;
  }, []);

  const dismissApproval = useCallback(() => {
    setApprovalRequest(null);
  }, []);

  const dismissStreamError = useCallback(() => {
    setStreamStatus(null);
    setError(null);
  }, []);

  // Copy whatever the assistant has streamed so far — used by the failure banner.
  const copyStreamedOutput = useCallback(() => {
    const last = messagesRef.current[messagesRef.current.length - 1];
    if (last?.role === "assistant" && last.content) {
      navigator.clipboard.writeText(last.content).catch((err) => {
        warn("copyOutput", "clipboard write failed", err);
      });
    }
  }, []);

  useEffect(() => {
    return () => {
      // Best-effort cleanup on unmount: stop any active run + close stream.
      if (activeRunRef.current) {
        stopRun(activeRunRef.current).catch(() => { /* unmount path */ });
      }
      unsubRef.current?.();
    };
  }, []);

  const clearPending = useCallback(() => {
    clearAllPendingMessages();
    setPendingMessages([]);
  }, []);

  return {
    messages,
    isStreaming,
    runStatus,
    progress,
    latestTurnEta,
    turnStartedAtMs,
    error,
    send,
    interrupt,
    setWorkspaceCallback,
    setOnRunComplete,
    approvalRequest,
    dismissApproval,
    streamStatus,
    dismissStreamError,
    copyStreamedOutput,
    externalUpdateAvailable,
    reloadSession,
    reloadCount,
    pendingMessages,
    flushPending,
    clearPending,
  };
}
