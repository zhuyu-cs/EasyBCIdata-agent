interface Props {
  onCopy: () => void;
  copied: boolean;
  onResend?: () => void;
  collapsible: boolean;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

function IconButton({
  onClick,
  title,
  active,
  children,
}: {
  onClick: () => void;
  title: string;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="w-6 h-6 flex items-center justify-center rounded transition-colors hover:bg-[var(--bg-hover)]"
      style={{ color: active ? "var(--accent-yellow)" : "var(--text-muted)" }}
    >
      {children}
    </button>
  );
}

/**
 * Compact actions surfaced on message hover. Lives in the message meta row so
 * it coexists with the A3 timestamp rather than overlapping the bubble.
 *
 * Bookmark was moved to the right-click ContextMenu (U10) since it's used
 * rarely — the hover toolbar keeps only Copy / Re-send / Collapse.
 */
export function MessageToolbar({
  onCopy,
  copied,
  onResend,
  collapsible,
  collapsed,
  onToggleCollapse,
}: Props) {
  return (
    <div className="flex items-center gap-0.5 opacity-0 group-hover/msg:opacity-100 focus-within:opacity-100 transition-opacity duration-150">
      <IconButton onClick={onCopy} title={copied ? "Copied" : "Copy text"}>
        {copied ? (
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M2.5 7l2.5 2.5L10.5 4" stroke="var(--accent-green)" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <rect x="4" y="4" width="6.5" height="6.5" rx="1" stroke="currentColor" strokeWidth="1.1" />
            <path d="M8.5 4V3a1 1 0 00-1-1H3a1 1 0 00-1 1v4.5a1 1 0 001 1h1" stroke="currentColor" strokeWidth="1.1" />
          </svg>
        )}
      </IconButton>

      {onResend && (
        <IconButton onClick={onResend} title="Re-send message">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M10.5 5.5A4 4 0 103 8.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
            <path d="M10.5 2.5v3h-3" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </IconButton>
      )}

      {collapsible && (
        <IconButton onClick={onToggleCollapse} title={collapsed ? "Expand" : "Collapse"}>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none" className={`transition-transform duration-200 ${collapsed ? "" : "rotate-180"}`}>
            <path d="M3 8l3.5-3.5L10 8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </IconButton>
      )}
    </div>
  );
}
