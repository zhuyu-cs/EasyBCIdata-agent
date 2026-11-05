import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { getSlot, useConversationStore, useSlotField, type SessionSlot } from "@/stores/conversationStore";
import * as engine from "@/lib/runManager";

export type { ApprovalRequest } from "@/stores/conversationStore";

const VERSION_POLL_INTERVALS = [10_000, 30_000, 60_000];

// Actions bound to whichever session is on screen at call time. Module-level
// so their identity never changes and memoized children never re-render
// because of them.
function onActive(fn: (id: string) => void) {
  return () => {
    const id = useSessionStore.getState().activeSessionId;
    if (id) fn(id);
  };
}

export const conversationActions = {
  send: (text: string) => {
    void engine.send(text, useSessionStore.getState().activeSessionId);
  },
  interrupt: onActive(engine.interrupt),
  removeQueued: (qid: string) => {
    const id = useSessionStore.getState().activeSessionId;
    if (id) engine.removeQueued(id, qid);
  },
  resolveApproval: onActive(engine.resolveApproval),
  hideApproval: onActive((id) => engine.setApprovalHidden(id, true)),
  showApproval: onActive((id) => engine.setApprovalHidden(id, false)),
  dismissStreamError: onActive(engine.dismissStreamError),
  retryStream: onActive(engine.retryStream),
  copyStreamedOutput: onActive(engine.copyStreamedOutput),
  reloadSession: onActive((id) => void engine.reloadSession(id)),
  flushPending: engine.flushPending,
  clearPending: engine.clearPending,
};

/** One field of the slot for the session on screen. */
export function useActiveSlot<K extends keyof SessionSlot>(key: K): SessionSlot[K] {
  const sid = useSessionStore((s) => s.activeSessionId);
  return useSlotField(sid, key);
}

/** Whether the session on screen has any turns — a boolean selector, so
 *  callers don't re-render on every streamed token. */
export function useHasMessages(): boolean {
  const sid = useSessionStore((s) => s.activeSessionId);
  return useConversationStore((s) => ((sid ? s.slots[sid]?.turns.length : 0) ?? 0) > 0);
}

/** Pending approval for the session on screen, unless its dialog was hidden. */
export function useVisibleApproval() {
  const sid = useSessionStore((s) => s.activeSessionId);
  return useConversationStore((s) => {
    const slot = sid ? s.slots[sid] : undefined;
    return slot?.approval && !slot.approvalHidden ? slot.approval : null;
  });
}

/**
 * Keeps the session on screen in sync with the backend: loads it on first
 * visit, revalidates on return, polls its version while idle, and replays
 * offline messages when the browser comes back online. Mount once.
 *
 * State lives in the per-session conversation store and runs are driven by
 * `lib/runManager`, so switching sessions never stops a run — it keeps
 * streaming into its own slot.
 */
export function useConversationSync() {
  const sid = useSessionStore((s) => s.activeSessionId);

  // Load on first visit; revalidate silently on return (unless a run owns
  // the slot — its in-memory turn is newer than the stored transcript).
  useEffect(() => {
    if (!sid) return;
    const s = getSlot(sid);
    if (s.loadState === "idle" || s.loadState === "error") void engine.loadSession(sid);
    else if (s.loadState === "loaded" && !s.run) void engine.loadSession(sid, { silent: true });
  }, [sid]);

  // Idle version polling with backoff (10s → 30s → 60s); detects edits made
  // by the CLI or another tab. Cheap: conditional request, 304 when unchanged.
  useEffect(() => {
    if (!sid) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let idleStreak = 0;
    const schedule = () => {
      if (cancelled) return;
      timer = setTimeout(tick, VERSION_POLL_INTERVALS[Math.min(idleStreak, VERSION_POLL_INTERVALS.length - 1)]);
    };
    const tick = async () => {
      if (cancelled) return;
      if (getSlot(sid).run) {
        idleStreak = 0;
        schedule();
        return;
      }
      const res = await engine.probeVersion(sid);
      if (cancelled) return;
      idleStreak = res === "unchanged" ? idleStreak + 1 : 0;
      schedule();
    };
    schedule();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [sid]);

  useEffect(() => {
    const onOnline = () => void engine.flushPending();
    window.addEventListener("online", onOnline);
    return () => window.removeEventListener("online", onOnline);
  }, []);
}
