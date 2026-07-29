import { useEffect, useRef, useState } from "react";

interface SlashCommand {
  command: string;
  description: string;
  /** Optional keyboard-shortcut hint shown on the right (display only). */
  shortcut?: string;
}

const SLASH_COMMANDS: SlashCommand[] = [
  { command: "/help", description: "Show available commands", shortcut: "?" },
  { command: "/clear", description: "Clear current conversation" },
  { command: "/export", description: "Export conversation to file", shortcut: "⌘E" },
  { command: "/new", description: "Start a new session", shortcut: "⌘⇧N" },
  { command: "/status", description: "Show agent status" },
  { command: "/tools", description: "List available tools" },
  { command: "/model", description: "Show or switch model" },
  { command: "/config", description: "Open settings" },
];

interface Props {
  query: string;
  onSelect: (command: string) => void;
  onClose: () => void;
  visible: boolean;
}

export function SlashMenu({ query, onSelect, onClose, visible }: Props) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [prevQuery, setPrevQuery] = useState(query);
  const menuRef = useRef<HTMLDivElement>(null);

  const filtered = SLASH_COMMANDS.filter((c) =>
    c.command.toLowerCase().startsWith(query.toLowerCase()),
  );

  // Reset the highlighted row when the query changes — done during render via
  // the prev-prop pattern rather than a state-setting effect.
  if (query !== prevQuery) {
    setPrevQuery(query);
    setSelectedIdx(0);
  }

  useEffect(() => {
    if (!visible) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIdx((i) => (i + 1) % filtered.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIdx((i) => (i - 1 + filtered.length) % filtered.length);
      } else if (e.key === "Tab" || e.key === "Enter") {
        if (filtered.length > 0) {
          e.preventDefault();
          onSelect(filtered[selectedIdx].command);
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", handler, true);
    return () => document.removeEventListener("keydown", handler, true);
  }, [visible, filtered, selectedIdx, onSelect, onClose]);

  if (!visible || filtered.length === 0) return null;

  return (
    <div
      ref={menuRef}
      className="absolute bottom-full left-0 mb-1 w-[300px] bg-[var(--bg-secondary)] rounded-lg shadow-lg border border-[var(--border-primary)] py-1 z-20 animate-fade-in"
    >
      {filtered.map((cmd, i) => (
        <button
          key={cmd.command}
          onClick={() => onSelect(cmd.command)}
          onMouseEnter={() => setSelectedIdx(i)}
          className={`w-full text-left px-3 py-2 flex items-center gap-2 transition-colors ${
            i === selectedIdx ? "bg-[var(--bg-tertiary)]" : "hover:bg-[var(--bg-input)]"
          }`}
        >
          <span className="text-[12px] font-mono text-[var(--text-primary)] shrink-0">{cmd.command}</span>
          <span className="text-[11px] text-[var(--text-muted)] truncate flex-1">{cmd.description}</span>
          {cmd.shortcut && (
            <kbd className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded border" style={{ color: "var(--text-faint)", borderColor: "var(--border-secondary)", background: "var(--bg-tertiary)" }}>
              {cmd.shortcut}
            </kbd>
          )}
        </button>
      ))}
      <div className="px-3 pt-1 mt-1 border-t flex items-center gap-2 text-[10px]" style={{ borderColor: "var(--border-primary)", color: "var(--text-faint)" }}>
        <span><kbd className="font-mono">↑↓</kbd> navigate</span>
        <span><kbd className="font-mono">↵</kbd> select</span>
        <span><kbd className="font-mono">esc</kbd> dismiss</span>
      </div>
    </div>
  );
}
