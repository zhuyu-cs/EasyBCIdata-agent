import { useEffect } from "react";

interface Props {
  onClose: () => void;
}

const isMac = typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad|iPod/.test(navigator.userAgent);

const MOD = isMac ? "⌘" : "Ctrl";

interface Shortcut {
  keys: string[];
  label: string;
}

const GROUPS: { title: string; items: Shortcut[] }[] = [
  {
    title: "Navigation",
    items: [
      { keys: [MOD, "K"], label: "Open command palette" },
      { keys: [MOD, "/"], label: "Focus session search" },
      { keys: [MOD, "["], label: "Previous session" },
      { keys: [MOD, "]"], label: "Next session" },
    ],
  },
  {
    title: "Session",
    items: [
      { keys: [MOD, "N"], label: "New session" },
      { keys: [MOD, "⇧", "N"], label: "New session & focus composer" },
      { keys: [MOD, "E"], label: "Export conversation" },
    ],
  },
  {
    title: "Run",
    items: [
      { keys: [MOD, "."], label: "Interrupt current run" },
      { keys: ["Esc"], label: "Close dialog / overlay" },
      { keys: ["?"], label: "Show this help" },
    ],
  },
];

function Keys({ keys }: { keys: string[] }) {
  return (
    <span className="flex items-center gap-1 shrink-0">
      {keys.map((k, i) => (
        <kbd
          key={i}
          className="min-w-[20px] text-center text-[11px] font-mono px-1.5 py-0.5 rounded border"
          style={{ color: "var(--text-secondary)", borderColor: "var(--border-secondary)", background: "var(--bg-tertiary)" }}
        >
          {k}
        </kbd>
      ))}
    </span>
  );
}

export function ShortcutsOverlay({ onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "?") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40 animate-backdrop-in" />
      <div
        className="relative w-[90vw] max-w-[440px] rounded-lg overflow-hidden animate-dialog-in"
        style={{ background: "var(--bg-secondary)", boxShadow: "var(--shadow-lg)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b" style={{ borderColor: "var(--border-primary)" }}>
          <h3 className="text-heading-sm" style={{ color: "var(--text-primary)" }}>Keyboard shortcuts</h3>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--text-muted)" }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
          {GROUPS.map((group) => (
            <div key={group.title}>
              <div className="text-label uppercase tracking-wide mb-2" style={{ color: "var(--text-faint)" }}>
                {group.title}
              </div>
              <div className="space-y-1.5">
                {group.items.map((item) => (
                  <div key={item.label} className="flex items-center justify-between gap-3">
                    <span className="text-caption" style={{ color: "var(--text-secondary)" }}>{item.label}</span>
                    <Keys keys={item.keys} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
