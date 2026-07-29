import { create } from "zustand";

interface SessionStore {
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  refreshTick: number;
  triggerRefresh: () => void;
  pinnedIds: string[];
  togglePin: (id: string) => void;
}

const PIN_KEY = "easybci-pinned-sessions";

function loadPinned(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(PIN_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function savePinned(ids: string[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(PIN_KEY, JSON.stringify(ids));
  } catch {
    /* ignore */
  }
}

function getSessionIdFromHash(): string | null {
  if (typeof window === "undefined") return null;
  const hash = window.location.hash;
  const match = hash.match(/^#session=(.+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

function syncHashToUrl(id: string | null) {
  if (typeof window === "undefined") return;
  const newHash = id ? `#session=${encodeURIComponent(id)}` : "";
  if (window.location.hash !== newHash) {
    history.pushState(null, "", newHash || window.location.pathname);
  }
}

export const useSessionStore = create<SessionStore>((set) => ({
  activeSessionId: getSessionIdFromHash(),
  setActiveSessionId: (id) => {
    syncHashToUrl(id);
    set({ activeSessionId: id });
  },
  refreshTick: 0,
  triggerRefresh: () => set((s) => ({ refreshTick: s.refreshTick + 1 })),
  pinnedIds: loadPinned(),
  togglePin: (id) =>
    set((s) => {
      const next = s.pinnedIds.includes(id)
        ? s.pinnedIds.filter((x) => x !== id)
        : [id, ...s.pinnedIds];
      savePinned(next);
      return { pinnedIds: next };
    }),
}));

if (typeof window !== "undefined") {
  window.addEventListener("hashchange", () => {
    const id = getSessionIdFromHash();
    const current = useSessionStore.getState().activeSessionId;
    if (id !== current) {
      useSessionStore.setState({ activeSessionId: id });
    }
  });

  window.addEventListener("popstate", () => {
    const id = getSessionIdFromHash();
    const current = useSessionStore.getState().activeSessionId;
    if (id !== current) {
      useSessionStore.setState({ activeSessionId: id });
    }
  });
}
