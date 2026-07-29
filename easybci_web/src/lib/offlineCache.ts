/**
 * offlineCache — minimal localStorage cache for resilience when the
 * dashboard backend is unreachable.
 *
 * Scope:
 *   - session list: full list mirrored, surfaced as "stale (offline)".
 *   - pending message queue: outbound user messages that failed to start
 *     a run; replayed when the gateway returns.
 *
 * Non-goals (intentional):
 *   - Caching message bodies of every session (would balloon localStorage).
 *   - Conflict resolution: server is always authoritative when reachable.
 */

import type { SessionInfo } from "./api";
import { warn } from "./debug";

const SESSION_LIST_KEY = "easybci-offline-sessions";
const PENDING_QUEUE_KEY = "easybci-offline-pending";
const VERSION = 1;

// ── Session list cache ────────────────────────────────────────────────

interface SessionCacheEntry {
  v: number;
  cachedAt: number;
  sessions: SessionInfo[];
}

export function saveSessionList(sessions: SessionInfo[]): void {
  if (typeof window === "undefined") return;
  try {
    const entry: SessionCacheEntry = { v: VERSION, cachedAt: Date.now(), sessions };
    localStorage.setItem(SESSION_LIST_KEY, JSON.stringify(entry));
  } catch (err) {
    warn("offlineCache", "failed to persist session list (quota?)", err);
  }
}

export function loadSessionList(): { sessions: SessionInfo[]; cachedAt: number } | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(SESSION_LIST_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SessionCacheEntry;
    if (parsed.v !== VERSION || !Array.isArray(parsed.sessions)) return null;
    return { sessions: parsed.sessions, cachedAt: parsed.cachedAt };
  } catch (err) {
    warn("offlineCache", "failed to read cached session list", err);
    return null;
  }
}

export function clearSessionList(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(SESSION_LIST_KEY);
  } catch {
    /* ignore */
  }
}

// ── Pending message queue ────────────────────────────────────────────

/** Discard a pending message after this many failed flush attempts (B4). */
export const PENDING_MAX_ATTEMPTS = 3;
/** Discard a pending message older than this (24h) regardless of attempts (B4). */
export const PENDING_TTL_MS = 24 * 60 * 60 * 1000;

export interface PendingMessage {
  id: string;
  text: string;
  sessionId: string | null;
  createdAt: number;
  /** Number of flush attempts so far — capped by PENDING_MAX_ATTEMPTS. */
  attempts: number;
}

interface PendingQueueEntry {
  v: number;
  items: PendingMessage[];
}

/** Drop items past their TTL or attempt cap. Returns the survivors. */
function pruneQueue(items: PendingMessage[], now: number): PendingMessage[] {
  return items.filter(
    (m) =>
      (m.attempts ?? 0) < PENDING_MAX_ATTEMPTS &&
      now - (m.createdAt ?? 0) < PENDING_TTL_MS,
  );
}

function readQueue(): PendingMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(PENDING_QUEUE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as PendingQueueEntry;
    if (parsed.v !== VERSION || !Array.isArray(parsed.items)) return [];
    // Normalise legacy items that predate the `attempts` field.
    const normalised = parsed.items.map((m) => ({ ...m, attempts: m.attempts ?? 0 }));
    const pruned = pruneQueue(normalised, Date.now());
    if (pruned.length !== normalised.length) writeQueue(pruned);
    return pruned;
  } catch (err) {
    warn("offlineCache", "failed to read pending queue", err);
    return [];
  }
}

function writeQueue(items: PendingMessage[]): void {
  if (typeof window === "undefined") return;
  try {
    const entry: PendingQueueEntry = { v: VERSION, items };
    localStorage.setItem(PENDING_QUEUE_KEY, JSON.stringify(entry));
  } catch (err) {
    warn("offlineCache", "failed to persist pending queue (quota?)", err);
  }
}

export function enqueuePendingMessage(msg: Omit<PendingMessage, "id" | "createdAt" | "attempts">): PendingMessage {
  const item: PendingMessage = {
    id: `pending-${Math.random().toString(36).slice(2, 10)}`,
    createdAt: Date.now(),
    attempts: 0,
    ...msg,
  };
  const items = readQueue();
  items.push(item);
  writeQueue(items);
  return item;
}

export function listPendingMessages(): PendingMessage[] {
  return readQueue();
}

export function removePendingMessage(id: string): void {
  const items = readQueue().filter((m) => m.id !== id);
  writeQueue(items);
}

/** Drop every queued message (used by the "Dismiss" affordance on the offline banner). */
export function clearAllPendingMessages(): void {
  writeQueue([]);
}

/**
 * Record a failed flush attempt for a pending item. Returns true if the item
 * was dropped (hit the attempt cap), false if it remains queued for retry (B4).
 */
export function bumpPendingAttempt(id: string): boolean {
  const items = readQueue();
  const idx = items.findIndex((m) => m.id === id);
  if (idx < 0) return true; // already gone — treat as dropped
  items[idx] = { ...items[idx], attempts: (items[idx].attempts ?? 0) + 1 };
  const dropped = items[idx].attempts >= PENDING_MAX_ATTEMPTS;
  writeQueue(dropped ? items.filter((m) => m.id !== id) : items);
  return dropped;
}

export function clearPendingMessages(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(PENDING_QUEUE_KEY);
  } catch {
    /* ignore */
  }
}
