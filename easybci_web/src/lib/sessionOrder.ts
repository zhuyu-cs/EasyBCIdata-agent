import type { SessionInfo } from "@/lib/api";

export interface TimeBucket<T> {
  label: string;
  items: T[];
}

/**
 * Bucket by time into Today / Yesterday / This Week / Earlier, preserving the
 * incoming order within each bucket and dropping empty buckets. Items with no
 * time land in Earlier.
 */
export function groupByTime<T>(items: T[], timeOf: (item: T) => number): TimeBucket<T>[] {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const todayTs = start.getTime() / 1000;
  const yesterdayTs = todayTs - 86400;
  const weekTs = todayTs - 6 * 86400;
  const buckets: TimeBucket<T>[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "This Week", items: [] },
    { label: "Earlier", items: [] },
  ];
  for (const item of items) {
    const t = timeOf(item);
    const idx = t >= todayTs ? 0 : t >= yesterdayTs ? 1 : t >= weekTs ? 2 : 3;
    buckets[idx].items.push(item);
  }
  return buckets.filter((b) => b.items.length > 0);
}

export function sessionTime(s: SessionInfo): number {
  return s.last_active || s.started_at;
}

/** Sessions in the order the sidebar shows them: pinned first, then by bucket. */
export function visibleSessionOrder(sessions: SessionInfo[], pinnedIds: string[]): SessionInfo[] {
  const pinned = new Set(pinnedIds);
  return [
    ...sessions.filter((s) => pinned.has(s.id)),
    ...groupByTime(sessions.filter((s) => !pinned.has(s.id)), sessionTime).flatMap((b) => b.items),
  ];
}
