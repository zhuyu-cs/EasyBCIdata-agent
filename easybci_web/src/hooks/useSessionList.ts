import { useEffect, useState, useCallback } from "react";
import { api, type SessionInfo } from "@/lib/api";
import { useSessionStore } from "@/stores/sessionStore";
import { loadSessionList, saveSessionList } from "@/lib/offlineCache";

interface SessionListState {
  sessions: SessionInfo[];
  loading: boolean;
  error: string | null;
  /** True when the current `sessions` array came from localStorage cache. */
  offline: boolean;
  /** When the cache was written (epoch ms). null when serving live data. */
  cachedAt: number | null;
}

export function useSessionList() {
  const refreshTick = useSessionStore((s) => s.refreshTick);
  const [state, setState] = useState<SessionListState>(() => {
    const cached = loadSessionList();
    if (cached) {
      // Optimistically render cached sessions while the live fetch runs —
      // avoids the empty-list flash on slow networks / cold loads.
      return {
        sessions: cached.sessions,
        loading: true,
        error: null,
        offline: false,
        cachedAt: cached.cachedAt,
      };
    }
    return { sessions: [], loading: true, error: null, offline: false, cachedAt: null };
  });

  const refresh = useCallback(async () => {
    setState((prev) => (prev.loading && !prev.error ? prev : { ...prev, loading: true, error: null }));
    try {
      const data = await api.getSessions(50, 0);
      saveSessionList(data.sessions);
      setState({
        sessions: data.sessions,
        loading: false,
        error: null,
        offline: false,
        cachedAt: null,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load sessions";
      const cached = loadSessionList();
      if (cached) {
        // Backend unreachable — surface stale data + offline flag.
        setState({
          sessions: cached.sessions,
          loading: false,
          error: msg,
          offline: true,
          cachedAt: cached.cachedAt,
        });
      } else {
        setState((prev) => ({ ...prev, loading: false, error: msg, offline: false }));
      }
    }
  }, []);

  useEffect(() => {
    // Schedule in a microtask so no setState is reachable synchronously from the
    // effect body (refresh manages its own loading/error state).
    Promise.resolve().then(refresh);
  }, [refresh, refreshTick]);

  return { ...state, refresh };
}
