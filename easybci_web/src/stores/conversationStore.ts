import { create } from "zustand";
import type { Turn } from "@/lib/turns";
import type { StageProgress } from "@/lib/stageProgress";
import type { StreamStatus } from "@/lib/runsClient";
import type { TurnEta } from "@/lib/turnEta";
import { listPendingMessages, type PendingMessage } from "@/lib/offlineCache";

export interface ApprovalRequest {
  runId: string;
  command?: string;
  description?: string;
  choices: string[];
}

export interface QueuedMessage {
  id: string;
  text: string;
}

export interface ActiveRun {
  /** null while POST /v1/runs is in flight. */
  runId: string | null;
  stopping: boolean;
}

/**
 * Everything the conversation view needs for one session. Slots are kept per
 * session so a run keeps streaming into its own slot while the user looks at
 * another session, and switching back renders instantly from the slot.
 */
export interface SessionSlot {
  turns: Turn[];
  loadState: "idle" | "loading" | "loaded" | "error";
  error: string | null;
  version: string | null;
  externalUpdate: boolean;
  run: ActiveRun | null;
  streamStatus: StreamStatus | null;
  progress: StageProgress | null;
  latestTurnEta: TurnEta | null;
  turnStartedAtMs: number | null;
  lastTurnTotalSec: number | null;
  approval: ApprovalRequest | null;
  /** Approval dialog dismissed with Escape — request still pending. */
  approvalHidden: boolean;
  queue: QueuedMessage[];
  reloadCount: number;
}

export function emptySlot(): SessionSlot {
  return {
    turns: [],
    loadState: "idle",
    error: null,
    version: null,
    externalUpdate: false,
    run: null,
    streamStatus: null,
    progress: null,
    latestTurnEta: null,
    turnStartedAtMs: null,
    lastTurnTotalSec: null,
    approval: null,
    approvalHidden: false,
    queue: [],
    reloadCount: 0,
  };
}

const EMPTY_SLOT: SessionSlot = emptySlot();

interface ConversationStore {
  slots: Record<string, SessionSlot>;
  /** Messages that could not reach the gateway, persisted for replay. */
  pending: PendingMessage[];
  setPending: (pending: PendingMessage[]) => void;
  patch: (sid: string, update: Partial<SessionSlot> | ((slot: SessionSlot) => Partial<SessionSlot>)) => void;
  updateTurns: (sid: string, fn: (turns: Turn[]) => Turn[]) => void;
  dropSlot: (sid: string) => void;
}

export const useConversationStore = create<ConversationStore>((set) => ({
  slots: {},
  pending: typeof window !== "undefined" ? listPendingMessages() : [],
  setPending: (pending) => set({ pending }),
  patch: (sid, update) =>
    set((s) => {
      const prev = s.slots[sid] ?? emptySlot();
      const delta = typeof update === "function" ? update(prev) : update;
      return { slots: { ...s.slots, [sid]: { ...prev, ...delta } } };
    }),
  updateTurns: (sid, fn) =>
    set((s) => {
      const prev = s.slots[sid] ?? emptySlot();
      const next = fn(prev.turns);
      if (next === prev.turns) return s;
      return { slots: { ...s.slots, [sid]: { ...prev, turns: next } } };
    }),
  dropSlot: (sid) =>
    set((s) => {
      if (!(sid in s.slots)) return s;
      const slots = { ...s.slots };
      delete slots[sid];
      return { slots };
    }),
}));

export function getSlot(sid: string | null | undefined): SessionSlot {
  if (!sid) return EMPTY_SLOT;
  return useConversationStore.getState().slots[sid] ?? EMPTY_SLOT;
}

/** Subscribe to one field of a session's slot. */
export function useSlotField<K extends keyof SessionSlot>(
  sid: string | null | undefined,
  key: K,
): SessionSlot[K] {
  return useConversationStore((s) => (sid ? s.slots[sid]?.[key] : undefined) ?? EMPTY_SLOT[key]);
}

/** Per-session run indicator for the session list. */
export type SessionRunBadge = "running" | "approval" | null;

export function useSessionRunBadge(sid: string): SessionRunBadge {
  return useConversationStore((s) => {
    const slot = s.slots[sid];
    if (!slot?.run) return null;
    return slot.approval ? "approval" : "running";
  });
}
