import { useCallback, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "easybci-input-history";
const MAX_ENTRIES = 50;

function loadHistory(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === "string");
  } catch {
    /* corrupt — ignore */
  }
  return [];
}

function saveHistory(entries: string[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    /* quota / disabled — non-fatal */
  }
}

/**
 * Shell-style input history backed by localStorage.
 *
 * `push` records a sent message (newest first, de-duplicated against the most
 * recent entry, capped at 50). `navigate` walks the history when the user
 * presses ↑/↓ on an empty (or history-navigated) input — returning the entry to
 * place in the textarea, or null to signal "stay at the live draft".
 */
export function useInputHistory() {
  const [history, setHistory] = useState<string[]>(loadHistory);
  // -1 means "not navigating" (showing the live draft). 0 is the newest entry.
  const cursorRef = useRef(-1);

  // Keep multiple tabs / instances loosely in sync.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setHistory(loadHistory());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const push = useCallback((entry: string) => {
    const trimmed = entry.trim();
    if (!trimmed) return;
    setHistory((prev) => {
      if (prev[0] === trimmed) return prev; // skip consecutive duplicate
      const next = [trimmed, ...prev.filter((e) => e !== trimmed)].slice(0, MAX_ENTRIES);
      saveHistory(next);
      return next;
    });
    cursorRef.current = -1;
  }, []);

  // Reset navigation cursor (call when the user types / sends).
  const resetCursor = useCallback(() => {
    cursorRef.current = -1;
  }, []);

  /**
   * Step through history. dir = -1 for "older" (↑), +1 for "newer" (↓).
   * Returns the string to show, or null to restore the live draft (when
   * stepping newer past the most recent entry).
   */
  const navigate = useCallback(
    (dir: -1 | 1): string | null => {
      if (history.length === 0) return null;
      let next = cursorRef.current;
      if (dir === -1) {
        next = Math.min(history.length - 1, cursorRef.current + 1);
      } else {
        next = cursorRef.current - 1;
      }
      cursorRef.current = next;
      if (next < 0) {
        cursorRef.current = -1;
        return null;
      }
      return history[next] ?? null;
    },
    [history],
  );

  const isNavigating = () => cursorRef.current !== -1;

  return { history, push, navigate, resetCursor, isNavigating };
}
