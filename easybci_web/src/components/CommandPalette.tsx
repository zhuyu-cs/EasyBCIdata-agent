import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api, type SessionSearchResult } from "@/lib/api";

export interface PaletteAction {
  id: string;
  label: string;
  /** Right-aligned hint — shortcut or category. */
  hint?: string;
  icon?: ReactNode;
  /** Extra terms folded into the fuzzy match. */
  keywords?: string;
  run: () => void;
}

interface Props {
  actions: PaletteAction[];
  onClose: () => void;
  onSelectSession: (id: string) => void;
}

const RECENT_KEY = "easybci-palette-recent";
const MAX_RECENT = 5;

function loadRecent(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function recordRecent(id: string) {
  try {
    const next = [id, ...loadRecent().filter((x) => x !== id)].slice(0, MAX_RECENT);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
}

// Lightweight relevance score: -1 = no match, higher is better.
function score(haystack: string, query: string): number {
  if (!query) return 0;
  const h = haystack.toLowerCase();
  const q = query.toLowerCase();
  const idx = h.indexOf(q);
  if (idx === -1) {
    // Subsequence fallback so "rica" still matches "Run ICA".
    let qi = 0;
    for (let i = 0; i < h.length && qi < q.length; i++) {
      if (h[i] === q[qi]) qi++;
    }
    return qi === q.length ? 1 : -1;
  }
  if (idx === 0) return 100;
  if (h[idx - 1] === " ") return 60; // word-boundary match
  return 30;
}

type Row =
  | { kind: "action"; action: PaletteAction }
  | { kind: "session"; result: SessionSearchResult };

export function CommandPalette({ actions, onClose, onSelectSession }: Props) {
  const [query, setQuery] = useState("");
  const [sessionResults, setSessionResults] = useState<SessionSearchResult[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Loaded once at mount and read during render for sort ordering — kept in
  // state (not a ref) so reading it during render is allowed.
  const [recent] = useState<string[]>(loadRecent);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Debounced session search (only when the query is non-trivial). When the
  // query is too short we simply skip the fetch; stale results are hidden by
  // the derived `visibleSessionResults` below rather than cleared in an effect.
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current);
    const q = query.trim();
    if (q.length < 2) return;
    searchTimer.current = setTimeout(async () => {
      try {
        const data = await api.searchSessions(q);
        setSessionResults(data.results.slice(0, 6));
      } catch {
        setSessionResults([]);
      }
    }, 200);
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, [query]);

  const filteredActions = useMemo(() => {
    const scored = actions
      .map((a) => ({ a, s: score(`${a.label} ${a.keywords ?? ""}`, query) }))
      .filter((x) => x.s >= 0);
    scored.sort((a, b) => {
      if (b.s !== a.s) return b.s - a.s;
      // Stable-ish secondary sort: recently used first when scores tie.
      const ra = recent.indexOf(a.a.id);
      const rb = recent.indexOf(b.a.id);
      const wa = ra === -1 ? Infinity : ra;
      const wb = rb === -1 ? Infinity : rb;
      return wa - wb;
    });
    return scored.map((x) => x.a);
  }, [actions, query, recent]);

  const rows: Row[] = useMemo(() => {
    // Hide stale session matches when the query is too short, rather than
    // clearing them in an effect.
    const visibleSessionResults = query.trim().length < 2 ? [] : sessionResults;
    const r: Row[] = filteredActions.map((action) => ({ kind: "action", action }));
    for (const result of visibleSessionResults) {
      r.push({ kind: "session", result });
    }
    return r;
  }, [filteredActions, sessionResults, query]);

  // Clamp the selection during render instead of storing a clamped value via an
  // effect — avoids a cascading re-render when the row set shrinks.
  const clampedIdx = Math.min(selectedIdx, Math.max(0, rows.length - 1));

  const choose = (row: Row) => {
    if (row.kind === "action") {
      recordRecent(row.action.id);
      row.action.run();
    } else {
      onSelectSession(row.result.session_id);
    }
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx(rows.length ? (clampedIdx + 1) % rows.length : 0);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx(rows.length ? (clampedIdx - 1 + rows.length) % rows.length : 0);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const row = rows[clampedIdx];
      if (row) choose(row);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  const actionCount = filteredActions.length;

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[12vh]" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40 animate-backdrop-in" />
      <div
        className="relative w-[90vw] max-w-[560px] rounded-xl overflow-hidden animate-dialog-in"
        style={{ background: "var(--bg-secondary)", boxShadow: "var(--shadow-lg)", border: "1px solid var(--border-primary)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search field */}
        <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: "var(--border-primary)" }}>
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none" style={{ color: "var(--text-muted)" }}>
            <circle cx="6.5" cy="6.5" r="4.5" stroke="currentColor" strokeWidth="1.3" />
            <path d="M10 10l3.5 3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search actions and sessions..."
            className="flex-1 bg-transparent text-[14px] outline-none placeholder:text-[var(--text-faint)]"
            style={{ color: "var(--text-primary)" }}
          />
          <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded border" style={{ color: "var(--text-faint)", borderColor: "var(--border-secondary)" }}>esc</kbd>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto py-1">
          {rows.length === 0 && (
            <div className="px-4 py-6 text-center text-[12px]" style={{ color: "var(--text-muted)" }}>
              No matches
            </div>
          )}

          {rows.map((row, i) => {
            const selected = i === clampedIdx;
            const isFirstSession = row.kind === "session" && (i === 0 || rows[i - 1].kind === "action");
            return (
              <div key={row.kind === "action" ? `a-${row.action.id}` : `s-${row.result.session_id}-${i}`}>
                {row.kind === "action" && i === 0 && actionCount > 0 && (
                  <div className="px-4 pt-2 pb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--text-faint)" }}>Actions</div>
                )}
                {isFirstSession && (
                  <div className="px-4 pt-2 pb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--text-faint)" }}>Sessions</div>
                )}
                <button
                  onClick={() => choose(row)}
                  onMouseEnter={() => setSelectedIdx(i)}
                  className="w-full text-left px-4 py-2 flex items-center gap-3 transition-colors"
                  style={{ background: selected ? "var(--bg-tertiary)" : undefined }}
                >
                  {row.kind === "action" ? (
                    <>
                      <span className="shrink-0 w-4 flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
                        {row.action.icon ?? (
                          <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                            <path d="M3 2l4 3.5L3 9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </span>
                      <span className="text-[13px] flex-1 truncate" style={{ color: "var(--text-primary)" }}>{row.action.label}</span>
                      {row.action.hint && (
                        <span className="text-[11px] font-mono shrink-0" style={{ color: "var(--text-faint)" }}>{row.action.hint}</span>
                      )}
                    </>
                  ) : (
                    <>
                      <span className="shrink-0 w-4 flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
                        <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                          <path d="M2 3.5h9M2 6.5h9M2 9.5h6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                        </svg>
                      </span>
                      <span className="text-[13px] flex-1 truncate" style={{ color: "var(--text-secondary)" }}>
                        {row.result.snippet || "Untitled session"}
                      </span>
                    </>
                  )}
                </button>
              </div>
            );
          })}
        </div>

        <div className="px-4 py-2 border-t flex items-center gap-3 text-[10px]" style={{ borderColor: "var(--border-primary)", color: "var(--text-faint)" }}>
          <span><kbd className="font-mono">↑↓</kbd> navigate</span>
          <span><kbd className="font-mono">↵</kbd> select</span>
        </div>
      </div>
    </div>
  );
}
