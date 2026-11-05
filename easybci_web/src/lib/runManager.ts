import { api } from "@/lib/api";
import {
  getRunStatus,
  isLiveRunStatus,
  startRun,
  stopRun,
  subscribeToRun,
  type RunEvent,
} from "@/lib/runsClient";
import { applyEwma } from "@/lib/turnEta";
import {
  appendText,
  closeTurn,
  completeStep,
  historyToTurns,
  mergeTurns,
  newLiveTurn,
  splitTurn,
  startStep,
  type Turn,
} from "@/lib/turns";
import type { StageProgress } from "@/lib/stageProgress";
import {
  bumpPendingAttempt,
  clearAllPendingMessages,
  enqueuePendingMessage,
  listPendingMessages,
  removePendingMessage,
} from "@/lib/offlineCache";
import { getSlot, useConversationStore, type SessionSlot } from "@/stores/conversationStore";
import { useSessionStore } from "@/stores/sessionStore";
import { refreshSessionList } from "@/stores/sessionListStore";
import { useToastStore } from "@/stores/toastStore";
import { warn } from "@/lib/debug";
import { t } from "@/lib/i18n";

/**
 * Conversation engine. Owns run subscriptions outside React so a run keeps
 * streaming into its session's slot while another session is on screen, and
 * survives component remounts. Components read slots from the store and call
 * the exported actions; nothing here depends on which session is visible
 * except the workspace event tap.
 */

const subs = new Map<string, () => void>();
const seenIds = new Map<string, Set<string>>();
const gapSessions = new Set<string>();
const stopTimers = new Map<string, ReturnType<typeof setTimeout>>();
const idleWaiters = new Map<string, Array<() => void>>();
const loadSeq = new Map<string, number>();

interface TextBuffer {
  delta: string;
  frame: number | null;
}
const buffers = new Map<string, TextBuffer>();

const STOP_CONFIRM_TIMEOUT_MS = 3000;

// ── store helpers ──────────────────────────────────────────────────────

function patch(sid: string, update: Partial<SessionSlot> | ((s: SessionSlot) => Partial<SessionSlot>)) {
  useConversationStore.getState().patch(sid, update);
}

function updateTurns(sid: string, fn: (turns: Turn[]) => Turn[]) {
  useConversationStore.getState().updateTurns(sid, fn);
}

function activeSessionId(): string | null {
  return useSessionStore.getState().activeSessionId;
}

/** Apply `fn` to the trailing turn — the one a run is writing into. */
function updateLastTurn(sid: string, fn: (t: Turn) => Turn) {
  updateTurns(sid, (turns) => {
    const lastIdx = turns.length - 1;
    if (lastIdx < 0) return turns;
    const updated = fn(turns[lastIdx]);
    if (updated === turns[lastIdx]) return turns;
    const next = turns.slice();
    next[lastIdx] = updated;
    return next;
  });
}

/** Close the trailing turn, stamping its wall-clock length. */
function closeLastTurn(
  sid: string,
  status: Parameters<typeof closeTurn>[1],
  opts: { error?: string; output?: string } = {},
) {
  const startedAt = getSlot(sid).turnStartedAtMs;
  const durationSec = startedAt != null ? Math.max(0, (Date.now() - startedAt) / 1000) : undefined;
  updateLastTurn(sid, (t) => closeTurn(t, status, { ...opts, durationSec }));
}

// ── session list refresh (deduplicated) ────────────────────────────────

let listRefreshTimer: ReturnType<typeof setTimeout> | null = null;

/** Refresh the sidebar once after `delayMs`; a later call replaces an
 *  earlier pending one, so back-to-back turns cost a single request. */
function scheduleListRefresh(delayMs: number) {
  if (listRefreshTimer) clearTimeout(listRefreshTimer);
  listRefreshTimer = setTimeout(() => {
    listRefreshTimer = null;
    void refreshSessionList();
  }, delayMs);
}

// ── live-run persistence (survives a page reload) ──────────────────────

const LIVE_RUNS_KEY = "easybci-live-runs";

interface PersistedRun {
  runId: string;
  lastEventId: string | null;
  /** The turn's input and start time: no stored row has them mid-run. */
  userText?: string;
  startedAtMs?: number;
}

function readPersisted(): Record<string, PersistedRun> {
  if (typeof window === "undefined") return {};
  try {
    const raw = sessionStorage.getItem(LIVE_RUNS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writePersisted(map: Record<string, PersistedRun>) {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(LIVE_RUNS_KEY, JSON.stringify(map));
  } catch {
    /* storage full / disabled — reload recovery is best-effort */
  }
}

function persistRun(sid: string, rec: PersistedRun) {
  const map = readPersisted();
  map[sid] = { ...map[sid], ...rec };
  writePersisted(map);
}

function forgetRun(sid: string) {
  const map = readPersisted();
  if (sid in map) {
    delete map[sid];
    writePersisted(map);
  }
}

// ── external taps (workspace panel, run-complete hooks) ────────────────

let eventTap: ((sid: string, ev: RunEvent) => void) | null = null;
let runCompleteTap: ((sid: string) => void) | null = null;

/** Receive run events for the session currently on screen. */
export function setRunEventTap(fn: ((sid: string, ev: RunEvent) => void) | null) {
  eventTap = fn;
}

export function setRunCompleteTap(fn: ((sid: string) => void) | null) {
  runCompleteTap = fn;
}

// ── history loading ────────────────────────────────────────────────────

function isNotFound(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return /^404\b/.test(msg) || /session not found/i.test(msg);
}

/**
 * Load a session's transcript into its slot. Stale responses (a newer load
 * started for the same session) are dropped, and a live run's slot is never
 * overwritten — its in-memory turn is newer than anything stored. Stored rows
 * merge over the slot's turns by id, so settled turns don't re-render.
 * `silent` revalidates without showing a loading state and skips the
 * re-render when the version stamp is unchanged.
 */
export async function loadSession(sid: string, opts: { silent?: boolean } = {}) {
  const seq = (loadSeq.get(sid) ?? 0) + 1;
  loadSeq.set(sid, seq);
  if (!opts.silent) patch(sid, (s) => (s.loadState === "loaded" ? {} : { loadState: "loading" }));
  try {
    const data = await api.getSessionMessages(sid);
    if (loadSeq.get(sid) !== seq) return;
    const slot = getSlot(sid);
    if (slot.run) {
      patch(sid, { loadState: "loaded" });
      return;
    }
    const version = data.version ?? null;
    if (opts.silent && version && version === slot.version && slot.loadState === "loaded") return;
    patch(sid, (s) => ({
      turns: mergeTurns(historyToTurns(sid, data.messages), s.turns),
      loadState: "loaded",
      error: null,
      version,
      externalUpdate: false,
    }));
  } catch (e) {
    if (loadSeq.get(sid) !== seq) return;
    // A brand-new session has no DB row until its first turn lands.
    if (isNotFound(e)) {
      patch(sid, { loadState: "loaded", error: null, version: null });
      return;
    }
    warn("loadSession", `failed to load ${sid}`, e);
    if (!opts.silent) {
      patch(sid, { loadState: "error", error: e instanceof Error ? e.message : String(e) });
    }
  }
}

/** Authoritative reload (external-update banner, post-gap reconcile). */
export async function reloadSession(sid: string) {
  await loadSession(sid, { silent: true });
  patch(sid, (s) => ({ reloadCount: s.reloadCount + 1, externalUpdate: false }));
}

/** Version probe for idle polling. Returns false when the call failed. */
export async function probeVersion(sid: string): Promise<"unchanged" | "changed" | "error"> {
  const known = getSlot(sid).version;
  try {
    const res = await api.getSessionVersionConditional(sid, known);
    if (res.notModified) return "unchanged";
    const slot = getSlot(sid);
    if (slot.run) return "unchanged";
    if (known && res.data.version !== known) {
      patch(sid, { externalUpdate: true });
      return "changed";
    }
    if (!known) patch(sid, { version: res.data.version });
    return "unchanged";
  } catch (err) {
    warn("versionPoll", "version probe failed; keeping current cadence", err);
    return "error";
  }
}

// ── text buffering (one store update per animation frame) ──────────────

function bufferFor(sid: string): TextBuffer {
  let b = buffers.get(sid);
  if (!b) {
    b = { delta: "", frame: null };
    buffers.set(sid, b);
  }
  return b;
}

function cancelFrame(b: TextBuffer) {
  if (b.frame == null) return;
  if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(b.frame);
  else clearTimeout(b.frame as unknown as ReturnType<typeof setTimeout>);
  b.frame = null;
}

function flush(sid: string) {
  const b = buffers.get(sid);
  if (!b) return;
  cancelFrame(b);
  const delta = b.delta;
  if (!delta) return;
  b.delta = "";
  updateLastTurn(sid, (t) => appendText(t, delta));
}

function scheduleFlush(sid: string) {
  const b = bufferFor(sid);
  if (b.frame != null) return;
  if (typeof requestAnimationFrame === "function") {
    b.frame = requestAnimationFrame(() => {
      b.frame = null;
      flush(sid);
    });
  } else {
    b.frame = setTimeout(() => {
      b.frame = null;
      flush(sid);
    }, 16) as unknown as number;
  }
}

// ── run event handling ─────────────────────────────────────────────────

function handleProgress(sid: string, ev: RunEvent) {
  const payload = ev.metadata?.progress as
    | (StageProgress & { eta_seconds?: number | null; reanchor?: boolean })
    | undefined;
  if (!payload) return;
  if ((payload.scope ?? "stage") === "turn") {
    const newEta = payload.eta_seconds;
    if (typeof newEta !== "number" || newEta <= 0) return;
    // Phase-boundary re-anchor: discard the previous smoothed value instead
    // of blending it with the ticked-down remainder.
    const reanchor = payload.reanchor === true;
    patch(sid, (s) => ({
      latestTurnEta: {
        smoothedSeconds: applyEwma(newEta, reanchor ? null : (s.latestTurnEta?.smoothedSeconds ?? null)),
        emittedAtMs: Date.now(),
        confidence: payload.confidence ?? "unknown",
      },
    }));
    return;
  }
  patch(sid, { progress: payload as StageProgress });
}

function onRunEvent(sid: string, ev: RunEvent) {
  if (ev.id) {
    let seen = seenIds.get(sid);
    if (!seen) {
      seen = new Set();
      seenIds.set(sid, seen);
    }
    if (seen.has(ev.id)) return;
    seen.add(ev.id);
  }
  if (sid === activeSessionId()) eventTap?.(sid, ev);

  switch (ev.event) {
    case "stream.gap":
      gapSessions.add(sid);
      return;
    case "progress":
      handleProgress(sid, ev);
      return;
    case "approval.request": {
      const slot = getSlot(sid);
      patch(sid, {
        approval: {
          runId: ev.run_id || slot.run?.runId || "",
          command: ev.command,
          description: ev.description,
          choices: ev.choices ?? ["once", "session", "always", "deny"],
        },
        approvalHidden: false,
      });
      if (sid !== activeSessionId()) {
        useToastStore.getState().add(t("A background session is waiting for your approval."), "info");
      }
      return;
    }
    case "approval.responded":
      patch(sid, { approval: null, approvalHidden: false });
      return;
    case "message.delta":
      if (ev.delta) {
        bufferFor(sid).delta += ev.delta;
        scheduleFlush(sid);
      }
      return;
    case "reasoning.available":
      // Despite the name this is a copy of the assistant message's text, which
      // already arrived as deltas. Real reasoning is not streamed; it comes
      // with the stored rows at the end of the turn.
      return;
  }

  // Structural events: drain buffered text first so ordering is preserved.
  flush(sid);
  const run = getSlot(sid).run;
  if (run?.runId && ev.id) persistRun(sid, { runId: run.runId, lastEventId: ev.id });

  if (ev.event === "tool.started") {
    const preview = ev.preview ?? ev.paths?.data_path ?? ev.paths?.work_dir ?? ev.paths?.output_dir;
    updateLastTurn(sid, (t) => startStep(t, { callId: ev.tool_call_id, tool: ev.tool ?? "unknown", preview }));
  } else if (ev.event === "tool.completed") {
    updateLastTurn(sid, (t) =>
      completeStep(t, {
        callId: ev.tool_call_id,
        tool: ev.tool,
        error: !!ev.error,
        durationSec: ev.duration,
        result: ev.result_preview,
        resultChars: ev.result_chars,
      }),
    );
  } else if (ev.event === "run.completed") {
    closeLastTurn(sid, "completed", { output: ev.output });
  } else if (ev.event === "run.failed") {
    closeLastTurn(sid, "failed", { error: typeof ev.error === "string" ? ev.error : undefined });
  } else if (ev.event === "run.cancelled") {
    closeLastTurn(sid, "cancelled");
  }
}

// ── run lifecycle ──────────────────────────────────────────────────────

function resolveIdle(sid: string) {
  const waiters = idleWaiters.get(sid);
  if (!waiters) return;
  idleWaiters.delete(sid);
  waiters.forEach((w) => w());
}

/** Resolves once the session has no active run. */
export function waitForIdle(sid: string): Promise<void> {
  if (!getSlot(sid).run) return Promise.resolve();
  return new Promise((resolve) => {
    const list = idleWaiters.get(sid) ?? [];
    list.push(resolve);
    idleWaiters.set(sid, list);
  });
}

function teardownStream(sid: string) {
  subs.get(sid)?.();
  subs.delete(sid);
  const t = stopTimers.get(sid);
  if (t) {
    clearTimeout(t);
    stopTimers.delete(sid);
  }
}

/** Close out a run: terminal event received, stream vanished, or stop timed out. */
function finishRun(sid: string, opts: { failedReason?: string } = {}) {
  flush(sid);
  teardownStream(sid);
  forgetRun(sid);
  const startedAt = getSlot(sid).turnStartedAtMs;
  // A terminal event already closed the turn; these cover a stream that
  // vanished without one.
  if (opts.failedReason) closeLastTurn(sid, "failed", { error: opts.failedReason });
  else closeLastTurn(sid, "completed");
  patch(sid, () => ({
    run: null,
    streamStatus: opts.failedReason ? { kind: "failed", reason: opts.failedReason } : null,
    progress: null,
    latestTurnEta: null,
    turnStartedAtMs: null,
    lastTurnTotalSec: startedAt != null ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000)) : null,
    approval: null,
    approvalHidden: false,
  }));
  scheduleListRefresh(300);

  // Pull the stored transcript so what's on screen is what the backend kept:
  // full step results, thinking, and its version stamp (so idle polling doesn't
  // flag our own write as an external update). Events lost on reconnect also
  // refresh what the workspace derived from them.
  if (gapSessions.has(sid)) {
    gapSessions.delete(sid);
    void reloadSession(sid);
  } else {
    void loadSession(sid, { silent: true });
  }
  runCompleteTap?.(sid);
  resolveIdle(sid);

  const next = getSlot(sid).queue[0];
  if (next) {
    patch(sid, (s) => ({ queue: s.queue.slice(1) }));
    void startTurn(sid, next.text);
  }
}

function subscribe(sid: string, runId: string, lastEventId: string | null) {
  teardownStream(sid);
  const unsub = subscribeToRun(
    runId,
    (ev) => onRunEvent(sid, ev),
    () => finishRun(sid),
    (err) => finishRun(sid, { failedReason: err.message }),
    {
      lastEventId,
      onStatus: (status) => {
        const prev = getSlot(sid).streamStatus;
        if (prev?.kind === status.kind && status.kind !== "reconnecting") return;
        patch(sid, { streamStatus: status });
      },
    },
  );
  subs.set(sid, unsub);
}

function newId(): string {
  return crypto.randomUUID();
}

async function startTurn(sid: string, text: string, opts: { rethrowOnFailure?: boolean } = {}) {
  const now = Date.now();
  const turn = newLiveTurn(sid, getSlot(sid).turns, text, now / 1000);
  seenIds.set(sid, new Set());
  gapSessions.delete(sid);
  patch(sid, (s) => ({
    turns: [...s.turns, turn],
    run: { runId: null, stopping: false },
    streamStatus: { kind: "connecting" },
    error: null,
    progress: null,
    latestTurnEta: null,
    turnStartedAtMs: now,
    lastTurnTotalSec: null,
  }));

  let runId: string;
  try {
    runId = await startRun(text, sid);
  } catch (e) {
    // Roll the optimistic turn back; the text either goes to the offline
    // queue (shown in the banner) or back to the caller for retry accounting.
    updateTurns(sid, (turns) => turns.filter((t) => t.id !== turn.id));
    patch(sid, { run: null, streamStatus: null, turnStartedAtMs: null });
    resolveIdle(sid);
    const msg = e instanceof Error ? e.message : t("Failed to send message");
    if (opts.rethrowOnFailure) {
      patch(sid, { error: msg });
      throw e instanceof Error ? e : new Error(msg);
    }
    enqueuePendingMessage({ text, sessionId: sid });
    useConversationStore.getState().setPending(listPendingMessages());
    patch(sid, { error: t("{m} (queued for retry when connection returns)", { m: msg }) });
    return;
  }

  const stopRequested = getSlot(sid).run?.stopping === true;
  patch(sid, { run: { runId, stopping: stopRequested } });
  persistRun(sid, { runId, lastEventId: null, userText: text, startedAtMs: now });
  // The gateway writes a heuristic title at run start; pick it up promptly.
  scheduleListRefresh(400);
  subscribe(sid, runId, null);
  if (stopRequested) requestStop(sid, runId);
}

/**
 * Send a message in `sid` (null = start a new session). While a run is active
 * the message is queued and sent automatically when the run ends — typing
 * never interrupts a running pipeline.
 */
export async function send(text: string, sid: string | null): Promise<{ queued: boolean }> {
  let target = sid;
  if (!target) {
    target = newId();
    patch(target, { loadState: "loaded" });
    useSessionStore.getState().setActiveSessionId(target);
  }
  if (getSlot(target).run) {
    patch(target, (s) => ({ queue: [...s.queue, { id: newId(), text }] }));
    return { queued: true };
  }
  await startTurn(target, text);
  return { queued: false };
}

export function removeQueued(sid: string, id: string) {
  patch(sid, (s) => ({ queue: s.queue.filter((q) => q.id !== id) }));
}

function requestStop(sid: string, runId: string) {
  void stopRun(runId);
  // Keep listening for run.cancelled; if the gateway never confirms, close
  // the turn locally so the UI never stays stuck in "running".
  const timer = setTimeout(() => {
    stopTimers.delete(sid);
    if (!getSlot(sid).run) return;
    closeLastTurn(sid, "cancelled");
    finishRun(sid);
  }, STOP_CONFIRM_TIMEOUT_MS);
  stopTimers.set(sid, timer);
}

/** Explicit stop (button / confirmed Escape). Queued messages are dropped. */
export function interrupt(sid: string) {
  const run = getSlot(sid).run;
  if (!run || run.stopping) return;
  patch(sid, { run: { ...run, stopping: true }, queue: [] });
  if (run.runId) requestStop(sid, run.runId);
}

// ── approvals ──────────────────────────────────────────────────────────

export function resolveApproval(sid: string) {
  patch(sid, { approval: null, approvalHidden: false });
}

export function setApprovalHidden(sid: string, hidden: boolean) {
  patch(sid, { approvalHidden: hidden });
}

export function dismissStreamError(sid: string) {
  patch(sid, { streamStatus: null, error: null });
}

// ── offline queue (gateway unreachable) ────────────────────────────────

let flushingPending = false;

/** Replay messages queued while the gateway was unreachable, in order. Each
 *  waits for the previous run to finish — no timeout can interrupt a run. */
export async function flushPending() {
  if (flushingPending) return;
  flushingPending = true;
  const store = useConversationStore.getState();
  try {
    for (const item of listPendingMessages()) {
      const sid = item.sessionId ?? activeSessionId() ?? newId();
      if (sid !== activeSessionId()) useSessionStore.getState().setActiveSessionId(sid);
      if (getSlot(sid).loadState === "idle") patch(sid, { loadState: "loaded" });
      await waitForIdle(sid);
      try {
        await startTurn(sid, item.text, { rethrowOnFailure: true });
        removePendingMessage(item.id);
        store.setPending(listPendingMessages());
        await waitForIdle(sid);
      } catch {
        const dropped = bumpPendingAttempt(item.id);
        store.setPending(listPendingMessages());
        if (dropped) {
          useToastStore.getState().add(t("A queued message could not be delivered and was discarded."), "error");
        }
        return;
      }
    }
  } finally {
    flushingPending = false;
  }
}

export function clearPending() {
  clearAllPendingMessages();
  useConversationStore.getState().setPending([]);
}

// ── reload recovery ────────────────────────────────────────────────────

/** The turn a re-attached run writes into. An agent flushes rows as it goes, so
 *  the stored rows may already hold part of it — then that turn restarts empty,
 *  since its events replay from the start. Without the input to compare, only a
 *  turn that visibly stopped short can be taken for it. */
function resumeTurns(sid: string, turns: Turn[], rec: PersistedRun): Turn[] {
  const last = turns[turns.length - 1];
  const isThisRun = rec.userText ? last?.user?.content === rec.userText : last?.status === "incomplete";
  if (last && isThisRun) {
    return [...turns.slice(0, -1), { ...last, items: [], status: "running", live: true }];
  }
  const ts = (rec.startedAtMs ?? Date.now()) / 1000;
  return [...turns, newLiveTurn(sid, turns, rec.userText ?? null, ts)];
}

/**
 * Re-attach to runs that were live when the page was reloaded. The stored
 * transcript is loaded first, then the gateway replays the run's events from
 * the beginning (its buffer outlives the connection) into a running turn. The
 * turn is marked for an authoritative reload at the end, which also covers any
 * events the bounded buffer has already dropped.
 */
export function restoreLiveRuns() {
  const persisted = readPersisted();
  for (const [sid, rec] of Object.entries(persisted)) {
    patch(sid, { loadState: "loading" });
    void (async () => {
      let live = false;
      try {
        const info = await getRunStatus(rec.runId);
        live = !!info && isLiveRunStatus(info.status);
      } catch {
        live = false;
      }
      if (!live) {
        forgetRun(sid);
        patch(sid, { loadState: "idle" });
        if (sid === activeSessionId()) void loadSession(sid);
        return;
      }
      patch(sid, { loadState: "idle" });
      await loadSession(sid);
      patch(sid, (s) => ({
        turns: resumeTurns(sid, s.turns, rec),
        run: { runId: rec.runId, stopping: false },
        streamStatus: { kind: "connecting" },
        turnStartedAtMs: rec.startedAtMs ?? null,
      }));
      seenIds.set(sid, new Set());
      gapSessions.add(sid);
      subscribe(sid, rec.runId, null);
    })();
  }
}

/** Copy whatever the assistant has streamed so far in `sid`. */
export function copyStreamedOutput(sid: string) {
  const turns = getSlot(sid).turns;
  const last = turns[turns.length - 1];
  const answer = last ? splitTurn(last).answer : "";
  if (answer) {
    navigator.clipboard.writeText(answer).catch((err) => warn("copyOutput", "clipboard write failed", err));
  }
}

/** Re-subscribe after a hard stream failure (Retry in the reconnect banner). */
export function retryStream(sid: string) {
  const persistedRun = readPersisted()[sid];
  const slot = getSlot(sid);
  patch(sid, { streamStatus: null, error: null });
  if (slot.run?.runId) {
    subscribe(sid, slot.run.runId, persistedRun?.lastEventId ?? null);
  } else {
    void reloadSession(sid);
  }
}
