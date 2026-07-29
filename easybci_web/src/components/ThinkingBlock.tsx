import { useState } from "react";

interface Props {
  thinking: string;
}

export function ThinkingBlock({ thinking }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-caption text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
      >
        <svg
          width="12" height="12" viewBox="0 0 12 12"
          className={`transition-transform ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Thinking
      </button>
      {expanded && (
        <div className="mt-1.5 px-3 py-2 rounded-md bg-[var(--bg-input)] border border-[var(--border-primary)] text-caption text-[var(--text-secondary)] whitespace-pre-wrap overflow-y-auto" style={{ maxHeight: 280 }}>
          {thinking}
        </div>
      )}
    </div>
  );
}
