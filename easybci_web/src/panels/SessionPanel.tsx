import { useState, useCallback, useRef, useEffect } from "react";
import { useSessionList } from "@/hooks/useSessionList";
import { useSessionStore } from "@/stores/sessionStore";
import { useToast } from "@/stores/toastStore";
import { useThemeToggle } from "@/stores/themeStore";
import { Logo, LogoMark } from "@/components/Logo";
import { ContextMenu, type ContextMenuItem } from "@/components/ContextMenu";
import { api, type SessionInfo, type SessionSearchResult } from "@/lib/api";

function timeAgo(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hour ago`;
  if (diff < 172800) return "Yesterday";
  return `${Math.floor(diff / 86400)} days ago`;
}

// Heuristic modality detection from session title/preview. Order matters:
// sEEG/ECoG must be checked before the broader EEG match.
const MODALITY_KEYWORDS: { re: RegExp; label: string; color: string }[] = [
  { re: /\bseeg\b/i, label: "sEEG", color: "#7c3aed" },
  { re: /\becog\b/i, label: "ECoG", color: "#7c3aed" },
  { re: /\bmeg\b/i, label: "MEG", color: "var(--accent-blue)" },
  { re: /\bf?nirs\b/i, label: "fNIRS", color: "var(--accent-red)" },
  { re: /\b(spike|sorting|neuron)\b/i, label: "Spike", color: "var(--accent-yellow)" },
  { re: /\beeg\b/i, label: "EEG", color: "var(--accent-green)" },
];

function detectModality(s: SessionInfo): { label: string; color: string } | null {
  const text = `${s.title ?? ""} ${s.preview ?? ""}`;
  for (const m of MODALITY_KEYWORDS) {
    if (m.re.test(text)) return { label: m.label, color: m.color };
  }
  return null;
}

// Shorten a long model id ("anthropic/claude-opus-4-8" → "claude-opus-4-8").
function shortModel(model: string | null): string | null {
  if (!model) return null;
  const parts = model.split("/");
  return parts[parts.length - 1];
}

// Bucket sessions into Today / Yesterday / This Week / Earlier, preserving
// the incoming order within each bucket and dropping empty buckets.
function groupByTime(sessions: SessionInfo[]): { label: string; items: SessionInfo[] }[] {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const todayTs = start.getTime() / 1000;
  const yesterdayTs = todayTs - 86400;
  const weekTs = todayTs - 6 * 86400;
  const order = ["Today", "Yesterday", "This Week", "Earlier"];
  const buckets: Record<string, SessionInfo[]> = {
    Today: [],
    Yesterday: [],
    "This Week": [],
    Earlier: [],
  };
  for (const s of sessions) {
    const t = s.last_active || s.started_at;
    if (t >= todayTs) buckets.Today.push(s);
    else if (t >= yesterdayTs) buckets.Yesterday.push(s);
    else if (t >= weekTs) buckets["This Week"].push(s);
    else buckets.Earlier.push(s);
  }
  return order.filter((l) => buckets[l].length > 0).map((l) => ({ label: l, items: buckets[l] }));
}

function SessionSkeleton() {
  return (
    <div className="px-3 py-2.5 mx-1 animate-pulse" style={{ width: "calc(100% - 8px)" }}>
      <div className="h-3.5 bg-[var(--bg-active)] rounded w-3/4 mb-2" />
      <div className="h-3 bg-[var(--bg-tertiary)] rounded w-1/2" />
    </div>
  );
}

function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={i} className="text-inherit rounded-sm px-[1px]" style={{ background: "var(--bg-search-highlight)" }}>{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function PinIcon({ filled }: { filled?: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 14 14" fill={filled ? "currentColor" : "none"}>
      <path d="M5 1.5h4l-.5 3 2 2.5H3.5l2-2.5-.5-3z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" />
      <path d="M7 9.5V13" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
    </svg>
  );
}

interface SessionItemProps {
  session: SessionInfo;
  active: boolean;
  pinned: boolean;
  selected: boolean;
  selectionActive: boolean;
  onActivate: () => void;
  onToggleSelect: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (id: string) => void;
  searchQuery?: string;
}

function SessionItem({
  session,
  active,
  pinned,
  selected,
  selectionActive,
  onActivate,
  onToggleSelect,
  onDelete,
  onRename,
  onTogglePin,
  searchQuery,
}: SessionItemProps) {
  const [showMenu, setShowMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title ?? "");
  const editRef = useRef<HTMLInputElement>(null);
  const modality = detectModality(session);
  const model = shortModel(session.model);

  useEffect(() => {
    if (editing) {
      editRef.current?.focus();
      editRef.current?.select();
    }
  }, [editing]);

  const startRename = () => {
    setDraft(session.title ?? "");
    setEditing(true);
    setShowMenu(false);
  };

  const commitRename = () => {
    setEditing(false);
    const next = draft.trim();
    if (next !== (session.title ?? "")) {
      onRename(session.id, next);
    }
  };

  const handleClick = (e: React.MouseEvent) => {
    if (e.shiftKey) {
      e.preventDefault();
      onToggleSelect();
    } else if (selectionActive) {
      onToggleSelect();
    } else {
      onActivate();
    }
  };

  return (
    <div
      className={`group relative animate-fade-in ${showMenu ? "z-30" : ""}`}
      onMouseLeave={() => setShowMenu(false)}
    >
      {editing ? (
        <div className="px-3 py-2 mx-1" style={{ width: "calc(100% - 8px)" }}>
          <input
            ref={editRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.preventDefault(); commitRename(); }
              else if (e.key === "Escape") { e.preventDefault(); setEditing(false); }
            }}
            className="w-full px-2 py-1 text-[13px] rounded-md bg-[var(--bg-input)] border focus:outline-none"
            style={{ borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
          />
        </div>
      ) : (
        <div
          role="button"
          tabIndex={0}
          onClick={handleClick}
          onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); onToggleSelect(); }}
          onKeyDown={(e) => { if (e.key === "Enter") onActivate(); }}
          className={`w-full text-left px-3 py-2.5 mx-1 rounded-md transition-all duration-150 cursor-pointer relative ${
            active ? "shadow-sm" : "hover:translate-x-[2px] hover:shadow-sm"
          }`}
          style={{
            width: "calc(100% - 8px)",
            background: selected
              ? "var(--bg-active)"
              : active
                ? "linear-gradient(90deg, var(--bg-active), var(--bg-hover))"
                : undefined,
          }}
          onMouseEnter={(e) => { if (!active && !selected) (e.currentTarget as HTMLElement).style.background = "var(--bg-tertiary)"; }}
          onMouseLeave={(e) => { if (!active && !selected) (e.currentTarget as HTMLElement).style.background = ""; }}
        >
          {active && (
            <span className="absolute left-0 top-2 bottom-2 w-[3px] rounded-full" style={{ background: "var(--text-primary)" }} />
          )}
          <div className="flex items-start gap-2">
            {/* Selection checkbox (shown in selection mode) or modality dot */}
            {selectionActive ? (
              <span
                className="shrink-0 mt-[2px] w-3.5 h-3.5 rounded border flex items-center justify-center"
                style={{
                  borderColor: selected ? "var(--accent-green)" : "var(--border-secondary)",
                  background: selected ? "var(--accent-green)" : "transparent",
                }}
              >
                {selected && (
                  <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                    <path d="M1.5 4.5l2 2 4-4" stroke="var(--text-on-accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </span>
            ) : (
              <span
                className="shrink-0 mt-[3px]"
                style={{ color: modality ? modality.color : "var(--text-faint)" }}
                title={modality ? `${modality.label} session` : undefined}
              >
                <LogoMark size={13} />
              </span>
            )}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1 mb-0.5">
                {pinned && (
                  <span className="shrink-0" style={{ color: "var(--accent-yellow)" }} title="Pinned">
                    <PinIcon filled />
                  </span>
                )}
                <span
                  className="text-[13px] font-medium truncate pr-6 flex-1"
                  style={{ color: active ? "var(--text-primary)" : "var(--text-secondary)" }}
                  onDoubleClick={(e) => { e.stopPropagation(); startRename(); }}
                  title="Double-click to rename"
                >
                  {searchQuery ? (
                    <HighlightText text={session.title || "Untitled Session"} query={searchQuery} />
                  ) : (
                    session.title || "Untitled Session"
                  )}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-caption truncate max-w-[60%]" style={{ color: "var(--text-muted)" }}>
                  {searchQuery ? (
                    <HighlightText text={session.preview || "No messages yet"} query={searchQuery} />
                  ) : (
                    session.preview || "No messages yet"
                  )}
                </span>
                <span className="text-label shrink-0 ml-2" style={{ color: "var(--text-faint)" }}>
                  {timeAgo(session.last_active || session.started_at)}
                </span>
              </div>
              {/* Metadata row */}
              {(session.message_count > 0 || model) && (
                <div className="flex items-center gap-2 mt-1">
                  {session.message_count > 0 && (
                    <span className="text-micro tabular-nums" style={{ color: "var(--text-faint)" }}>
                      {session.message_count} msg
                    </span>
                  )}
                  {model && (
                    <span className="text-micro truncate max-w-[55%]" style={{ color: "var(--text-faint)" }} title={session.model ?? undefined}>
                      {model}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* More menu trigger */}
      {!editing && !selectionActive && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowMenu(!showMenu);
          }}
          className="absolute right-3 top-3 w-6 h-6 flex items-center justify-center rounded-md opacity-0 group-hover:opacity-100 hover:bg-[var(--bg-active)] transition-all"
          style={{ color: "var(--text-muted)" }}
          title="Session options"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
            <circle cx="7" cy="3" r="1.2" />
            <circle cx="7" cy="7" r="1.2" />
            <circle cx="7" cy="11" r="1.2" />
          </svg>
        </button>
      )}

      {/* Dropdown menu */}
      {showMenu && (
        <div className="absolute right-2 top-9 z-40 rounded-md border py-1 min-w-[150px] animate-fade-in" style={{ background: "var(--bg-secondary)", borderColor: "var(--border-primary)", boxShadow: "var(--shadow-lg)" }}>
          <button
            onClick={(e) => { e.stopPropagation(); startRename(); }}
            className="w-full text-left px-3 py-1.5 text-[12px] transition-colors hover:bg-[var(--bg-tertiary)]"
            style={{ color: "var(--text-primary)" }}
          >
            Rename
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setShowMenu(false); onTogglePin(session.id); }}
            className="w-full text-left px-3 py-1.5 text-[12px] transition-colors hover:bg-[var(--bg-tertiary)]"
            style={{ color: "var(--text-primary)" }}
          >
            {pinned ? "Unpin" : "Pin to top"}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setShowMenu(false); onToggleSelect(); }}
            className="w-full text-left px-3 py-1.5 text-[12px] transition-colors hover:bg-[var(--bg-tertiary)]"
            style={{ color: "var(--text-primary)" }}
          >
            Select
          </button>
          <div className="my-1 border-t" style={{ borderColor: "var(--border-primary)" }} />
          <button
            onClick={(e) => { e.stopPropagation(); setShowMenu(false); onDelete(session.id); }}
            className="w-full text-left px-3 py-1.5 text-[12px] transition-colors"
            style={{ color: "var(--text-delete)" }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--bg-delete-hover)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = ""; }}
          >
            Delete session
          </button>
        </div>
      )}
    </div>
  );
}

function DeleteConfirmDialog({ count, onConfirm, onCancel }: { count: number; onConfirm: () => void; onCancel: () => void }) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") onConfirm();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onConfirm, onCancel]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/30 animate-backdrop-in" onClick={onCancel} />
      <div className="relative rounded-lg p-5 w-[90vw] max-w-[320px] animate-dialog-in" style={{ background: "var(--bg-secondary)", boxShadow: "var(--shadow-lg)" }}>
        <h3 className="text-[14px] font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
          {count > 1 ? `Delete ${count} sessions?` : "Delete session?"}
        </h3>
        <p className="text-[13px] mb-4" style={{ color: "var(--text-secondary)" }}>This action cannot be undone.</p>
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-md text-[12px] font-medium hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--text-secondary)" }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-3 py-1.5 rounded-md text-[12px] font-medium transition-colors"
            style={{ color: "var(--text-on-accent)", background: "var(--bg-button-danger)" }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--bg-button-danger-hover)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--bg-button-danger)"; }}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

export function SessionPanel({ onOpenSettings }: { onOpenSettings?: () => void } = {}) {
  const { sessions, loading, error, refresh, offline, cachedAt } = useSessionList();
  const { activeSessionId, setActiveSessionId } = useSessionStore();
  const pinnedIds = useSessionStore((s) => s.pinnedIds);
  const togglePin = useSessionStore((s) => s.togglePin);
  const { toast } = useToast();
  const { isDark, toggle: toggleTheme } = useThemeToggle();

  const [deleteTargets, setDeleteTargets] = useState<string[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showSearch, setShowSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SessionSearchResult[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  // Right-click menu — used by empty-area refresh. SessionItem's own right-
  // click handler calls e.stopPropagation() so the toggle-select action keeps
  // working on individual rows without triggering this menu.
  const [menu, setMenu] = useState<{ x: number; y: number; items: ContextMenuItem[] } | null>(null);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const selectionActive = selectedIds.size > 0;

  const handleNewSession = () => {
    setActiveSessionId(null);
  };

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleRename = useCallback(async (id: string, title: string) => {
    try {
      await api.renameSession(id, title);
      toast(title ? "Session renamed" : "Title cleared", "success");
      refresh();
    } catch (e) {
      toast(e instanceof Error ? e.message.replace(/^\d+:\s*/, "") : "Rename failed", "error");
    }
  }, [refresh, toast]);

  const handleDeleteRequest = (id: string) => setDeleteTargets([id]);

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTargets || deleteTargets.length === 0) return;
    const targets = deleteTargets;
    const deletingActive = !!activeSessionId && targets.includes(activeSessionId);
    let failed = 0;
    // Backend-suggested navigation target after the active session is removed.
    let nextId: string | null | undefined;
    await Promise.all(
      targets.map(async (id) => {
        try {
          const res = await api.deleteSession(id);
          if (deletingActive && id === activeSessionId) {
            nextId = res.next_id;
          }
        } catch {
          failed++;
        }
      }),
    );
    if (failed === 0) {
      toast(targets.length > 1 ? `${targets.length} sessions deleted` : "Session deleted", "success");
    } else {
      toast(`${failed} of ${targets.length} failed to delete`, "error");
    }
    if (deletingActive) {
      // Prefer the backend's authoritative next_id; fall back to the most
      // recent still-loaded session so we never leave activeSessionId pointing
      // at a deleted id (B5).
      if (nextId !== undefined) {
        setActiveSessionId(nextId ?? null);
      } else {
        const remaining = sessions.filter((s) => !targets.includes(s.id));
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
      }
    }
    setSelectedIds(new Set());
    setDeleteTargets(null);
    refresh();
  }, [deleteTargets, activeSessionId, sessions, refresh, setActiveSessionId, toast]);

  const handleDeleteCancel = useCallback(() => setDeleteTargets(null), []);

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);

    if (!value.trim()) {
      setSearchResults(null);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    searchTimerRef.current = setTimeout(async () => {
      try {
        const data = await api.searchSessions(value.trim());
        setSearchResults(data.results);
      } catch {
        setSearchResults([]);
      }
      setIsSearching(false);
    }, 300);
  };

  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, []);

  // Reveal + focus the search box when "/" is pressed elsewhere in the app.
  useEffect(() => {
    const open = () => {
      setShowSearch(true);
      requestAnimationFrame(() => searchInputRef.current?.focus());
    };
    window.addEventListener("easybci:focus-search", open);
    return () => window.removeEventListener("easybci:focus-search", open);
  }, []);

  // Focus the input whenever the search bar is revealed.
  useEffect(() => {
    if (showSearch) searchInputRef.current?.focus();
  }, [showSearch]);

  const displaySessions: SessionInfo[] = searchResults
    ? searchResults.map((r) => ({
        id: r.session_id,
        title: r.snippet,
        preview: r.snippet,
        source: r.source,
        model: r.model,
        started_at: r.session_started ?? 0,
        last_active: r.session_started ?? 0,
        ended_at: null,
        is_active: false,
        message_count: 0,
        tool_call_count: 0,
        input_tokens: 0,
        output_tokens: 0,
      }))
    : sessions;

  const pinnedSet = new Set(pinnedIds);
  const pinnedSessions = displaySessions.filter((s) => pinnedSet.has(s.id));
  const unpinnedSessions = displaySessions.filter((s) => !pinnedSet.has(s.id));

  const renderItem = (session: SessionInfo) => (
    <SessionItem
      key={session.id}
      session={session}
      active={activeSessionId === session.id}
      pinned={pinnedSet.has(session.id)}
      selected={selectedIds.has(session.id)}
      selectionActive={selectionActive}
      onActivate={() => setActiveSessionId(session.id)}
      onToggleSelect={() => toggleSelect(session.id)}
      onDelete={handleDeleteRequest}
      onRename={handleRename}
      onTogglePin={togglePin}
      searchQuery={searchQuery}
    />
  );

  return (
    <>
      <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "var(--border-primary)" }}>
        <div className="flex items-center gap-2">
          <Logo size={22} className="text-[var(--text-primary)]" />
          <h1 className="text-heading-sm" style={{ color: "var(--text-primary)" }}>EasyBCI</h1>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              if (showSearch && searchQuery) handleSearchChange("");
              setShowSearch((v) => !v);
            }}
            className={`w-7 h-7 flex items-center justify-center rounded-md transition-colors ${showSearch ? "bg-[var(--bg-active)]" : "hover:bg-[var(--bg-hover)]"}`}
            style={{ color: showSearch ? "var(--text-primary)" : "var(--text-muted)" }}
            title="Search sessions (/)"
          >
            <svg width="15" height="15" viewBox="0 0 13 13" fill="none">
              <circle cx="5.5" cy="5.5" r="4" stroke="currentColor" strokeWidth="1.3" />
              <path d="M8.5 8.5L12 12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </button>
          <button
            onClick={handleNewSession}
            className="w-7 h-7 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--text-muted)" }}
            title="New session (Ctrl+N)"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {/* Search bar — hidden by default, revealed via search icon or "/" */}
      {showSearch && (
      <div className="px-3 pt-2 pb-1 animate-fade-in">
        <div className="relative">
          <svg
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]"
            width="13" height="13" viewBox="0 0 13 13" fill="none"
          >
            <circle cx="5.5" cy="5.5" r="4" stroke="currentColor" strokeWidth="1.3" />
            <path d="M8.5 8.5L12 12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search sessions..."
            className="w-full pl-8 pr-3 py-1.5 text-[12px] rounded-md bg-[var(--bg-input)] border border-transparent focus:border-[var(--border-secondary)] focus:bg-[var(--bg-secondary)] focus:outline-none placeholder:text-[var(--text-faint)] transition-colors"
            data-search-input
          />
          {searchQuery && (
            <button
              onClick={() => handleSearchChange("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
              </svg>
            </button>
          )}
        </div>
      </div>
      )}

      {/* Batch selection action bar */}
      {selectionActive && (
        <div className="mx-3 my-1 px-3 py-2 rounded-md flex items-center justify-between animate-fade-in" style={{ background: "var(--bg-tertiary)" }}>
          <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>
            {selectedIds.size} selected
          </span>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setSelectedIds(new Set())}
              className="text-[12px] px-2 py-1 rounded hover:bg-[var(--bg-hover)] transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              Cancel
            </button>
            <button
              onClick={() => setDeleteTargets([...selectedIds])}
              className="text-[12px] px-2 py-1 rounded transition-colors"
              style={{ color: "var(--text-on-accent)", background: "var(--bg-button-danger)" }}
            >
              Delete
            </button>
          </div>
        </div>
      )}

      <div
        className="flex-1 overflow-y-auto py-1"
        onContextMenu={(e) => {
          e.preventDefault();
          setMenu({
            x: e.clientX,
            y: e.clientY,
            items: [{ label: "Refresh", action: () => { void refresh(); } }],
          });
        }}
      >
        {offline && cachedAt && (
          <div
            className="mx-3 my-2 px-2.5 py-1.5 rounded text-[10.5px] flex items-center gap-2"
            style={{
              background: "var(--bg-warning-subtle)",
              color: "var(--text-warning)",
              border: "1px solid var(--border-warning)",
            }}
            title={`Cache from ${new Date(cachedAt).toLocaleString()}`}
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <circle cx="5" cy="5" r="4" stroke="currentColor" strokeWidth="1.2" />
              <path d="M5 3v2.5M5 7v.01" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            </svg>
            <span className="font-medium">Working offline</span>
            <span className="opacity-70">— showing cached sessions</span>
            <button
              onClick={refresh}
              className="ml-auto underline hover:no-underline"
              style={{ color: "var(--text-warning)" }}
            >
              Retry
            </button>
          </div>
        )}

        {loading && !searchResults && (
          <>
            <SessionSkeleton />
            <SessionSkeleton />
            <SessionSkeleton />
          </>
        )}

        {isSearching && (
          <div className="px-4 py-3 text-center">
            <span className="text-[11px] text-[var(--text-muted)]">Searching...</span>
          </div>
        )}

        {error && !offline && !searchResults && (
          <div className="px-4 py-6 text-center">
            <p className="text-[12px] text-[var(--text-muted)] mb-2">
              {error.includes("fetch") || error.includes("network")
                ? "Backend not connected"
                : error}
            </p>
            <p className="text-[11px] text-[var(--text-faint)] mb-3">
              Start with: <code className="bg-[var(--bg-tertiary)] px-1 rounded">easybci web</code>
            </p>
            <button
              onClick={refresh}
              className="text-[12px] text-[var(--text-primary)] underline hover:no-underline"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && !isSearching && displaySessions.length === 0 && (
          <div className="px-4 py-6 text-center">
            {searchQuery ? (
              <p className="text-[12px] text-[var(--text-muted)]">No results for &ldquo;{searchQuery}&rdquo;</p>
            ) : (
              <>
                <p className="text-[12px] text-[var(--text-muted)]">No sessions yet</p>
                <p className="text-[11px] text-[var(--text-faint)] mt-1">Start a conversation below</p>
              </>
            )}
          </div>
        )}

        {!loading && !error && !isSearching && (
          searchResults
            ? displaySessions.map(renderItem)
            : (
              <>
                {pinnedSessions.length > 0 && (
                  <div>
                    <div className="px-4 pt-3 pb-1 text-label uppercase tracking-wide flex items-center gap-1" style={{ color: "var(--text-faint)" }}>
                      <PinIcon /> Pinned
                    </div>
                    {pinnedSessions.map(renderItem)}
                  </div>
                )}
                {groupByTime(unpinnedSessions).map((group) => (
                  <div key={group.label}>
                    <div
                      className="px-4 pt-3 pb-1 text-label uppercase tracking-wide"
                      style={{ color: "var(--text-faint)" }}
                    >
                      {group.label}
                    </div>
                    {group.items.map(renderItem)}
                  </div>
                ))}
              </>
            )
        )}
      </div>

      {deleteTargets && (
        <DeleteConfirmDialog count={deleteTargets.length} onConfirm={handleDeleteConfirm} onCancel={handleDeleteCancel} />
      )}

      {menu && (
        <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} />
      )}

      {/* Footer: settings (primary, gear) + theme toggle (secondary) */}
      <div className="shrink-0 px-4 py-2 border-t border-[var(--border-primary)] flex items-center justify-between">
        {onOpenSettings ? (
          <button
            onClick={onOpenSettings}
            className="flex items-center gap-2 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            title="Settings"
            aria-label="Open settings"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            Settings
          </button>
        ) : <span />}
        <button
          onClick={toggleTheme}
          className="w-7 h-7 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
          style={{ color: "var(--text-muted)" }}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
          aria-label="Toggle theme"
        >
          {isDark ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="3" stroke="currentColor" strokeWidth="1.3" />
              <path d="M7 1.5v1M7 11.5v1M1.5 7h1M11.5 7h1M3.1 3.1l.7.7M10.2 10.2l.7.7M10.2 3.1l.7.7M3.1 10.2l.7.7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M12.5 8.5a5.5 5.5 0 01-7-7 5.5 5.5 0 107 7z" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
      </div>
    </>
  );
}
