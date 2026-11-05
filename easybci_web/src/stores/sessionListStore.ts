import { create } from "zustand";
import { api, type SessionInfo } from "@/lib/api";
import { loadSessionList, saveSessionList } from "@/lib/offlineCache";
import { t } from "@/lib/i18n";

interface SessionListStore {
  sessions: SessionInfo[];
  /** All sessions on the backend; more than `sessions.length` until loaded. */
  total: number;
  /** A fetch is running with nothing to show yet — refreshes keep the list up. */
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  /** True when the current `sessions` array came from localStorage cache. */
  offline: boolean;
  /** When the cache was written (epoch ms). null when serving live data. */
  cachedAt: number | null;
}

const cached = typeof window !== "undefined" ? loadSessionList() : null;

// Cached sessions render immediately while the first live fetch runs, so a
// cold load never flashes an empty sidebar.
export const useSessionListStore = create<SessionListStore>(() => ({
  sessions: cached?.sessions ?? [],
  total: cached?.sessions.length ?? 0,
  loading: !cached?.sessions.length,
  loadingMore: false,
  error: null,
  offline: false,
  cachedAt: cached?.cachedAt ?? null,
}));

const PAGE_SIZE = 50;

let inflight: Promise<void> | null = null;
let again = false;

async function fetchOnce(): Promise<void> {
  const set = useSessionListStore.setState;
  set((prev) => (prev.sessions.length ? (prev.error ? { error: null } : prev) : { loading: true, error: null }));
  try {
    // Re-fetch everything already on screen, so a refresh doesn't drop pages
    // the user loaded with "Load more".
    const limit = Math.max(PAGE_SIZE, useSessionListStore.getState().sessions.length);
    const data = await api.getSessions(limit, 0);
    saveSessionList(data.sessions);
    set({ sessions: data.sessions, total: data.total, loading: false, error: null, offline: false, cachedAt: null });
  } catch (e) {
    const msg = e instanceof Error ? e.message : t("Failed to load sessions");
    const fallback = loadSessionList();
    if (fallback) {
      set({ sessions: fallback.sessions, loading: false, error: msg, offline: true, cachedAt: fallback.cachedAt });
    } else {
      set({ loading: false, error: msg, offline: false });
    }
  }
}

/**
 * Re-fetch the sidebar list. Calls made while a fetch is in flight collapse
 * into a single follow-up fetch, so bursts never fan out into N requests.
 */
export function refreshSessionList(): Promise<void> {
  if (inflight) {
    again = true;
    return inflight;
  }
  inflight = (async () => {
    try {
      do {
        again = false;
        await fetchOnce();
      } while (again);
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

/** Append the next page. A session created meanwhile only shifts the offset,
 *  so rows are de-duplicated by id. */
export async function loadMoreSessions(): Promise<void> {
  const { sessions, total, loadingMore } = useSessionListStore.getState();
  if (loadingMore || sessions.length >= total) return;
  useSessionListStore.setState({ loadingMore: true });
  try {
    const data = await api.getSessions(PAGE_SIZE, sessions.length);
    useSessionListStore.setState((prev) => {
      const known = new Set(prev.sessions.map((s) => s.id));
      return {
        sessions: [...prev.sessions, ...data.sessions.filter((s) => !known.has(s.id))],
        total: data.total,
        loadingMore: false,
      };
    });
  } catch {
    useSessionListStore.setState({ loadingMore: false });
  }
}

// One list for the whole app: fetched as soon as the module loads, then
// refreshed on demand via refreshSessionList().
if (typeof window !== "undefined") queueMicrotask(() => void refreshSessionList());

export function useSessionList() {
  const state = useSessionListStore();
  return { ...state, refresh: refreshSessionList };
}
