import { type ReactNode, useState, useEffect, useCallback } from "react";
import { useResizeHandle } from "@/hooks/useResizeHandle";
import { ErrorBoundary } from "@/components/ErrorBoundary";

interface WorkspaceLayoutProps {
  sessionPanel: ReactNode;
  conversationPanel: ReactNode;
  workspacePanel: ReactNode;
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [query]);
  return matches;
}

export function WorkspaceLayout({
  sessionPanel,
  conversationPanel,
  workspacePanel,
}: WorkspaceLayoutProps) {
  const { leftWidth, rightWidth, onResizeStart, onDoubleClick } = useResizeHandle();
  const isTablet = useMediaQuery("(max-width: 1024px)");
  const isMobile = useMediaQuery("(max-width: 768px)");

  const [sessionOpen, setSessionOpen] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);

  const closeOverlays = useCallback(() => {
    setSessionOpen(false);
    setWorkspaceOpen(false);
  }, []);

  // Close slide-over panels when the viewport grows past their breakpoint —
  // adjusted during render via the prev-value pattern (no state-setting effect).
  const [prevIsMobile, setPrevIsMobile] = useState(isMobile);
  const [prevIsTablet, setPrevIsTablet] = useState(isTablet);
  if (isMobile !== prevIsMobile) {
    setPrevIsMobile(isMobile);
    if (!isMobile) setSessionOpen(false);
  }
  if (isTablet !== prevIsTablet) {
    setPrevIsTablet(isTablet);
    if (!isTablet) setWorkspaceOpen(false);
  }

  if (isMobile) {
    return (
      <div className="h-full max-h-full overflow-hidden flex flex-col relative" style={{ background: "var(--bg-secondary)" }}>
        {/* Mobile header with hamburger + workspace toggle */}
        <div className="flex items-center px-3 py-2 border-b shrink-0" style={{ borderColor: "var(--border-primary)", background: "var(--bg-primary)" }}>
          <button
            onClick={() => setSessionOpen(true)}
            className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M3 5h12M3 9h12M3 13h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
          <span className="ml-2 text-[13px] font-medium" style={{ color: "var(--text-primary)" }}>EasyBCI</span>
          <button
            onClick={() => setWorkspaceOpen(true)}
            className="ml-auto w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M3 4h5l1 1h5a1 1 0 011 1v7a1 1 0 01-1 1H3a1 1 0 01-1-1V5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.3" fill="none" />
            </svg>
          </button>
        </div>

        {/* Conversation fills remaining space */}
        <div className="flex-1 min-h-0 flex flex-col">
          <ErrorBoundary panelName="Conversation">
            {conversationPanel}
          </ErrorBoundary>
        </div>

        {/* Session slide-over */}
        {sessionOpen && (
          <div className="absolute inset-0 z-40 flex">
            <div className="absolute inset-0 bg-black/40 animate-backdrop-in" onClick={closeOverlays} />
            <aside className="relative w-[280px] max-w-[80vw] h-full flex flex-col animate-dialog-in" style={{ background: "var(--bg-primary)" }}>
              <ErrorBoundary panelName="Sessions">
                {sessionPanel}
              </ErrorBoundary>
            </aside>
          </div>
        )}

        {/* Workspace slide-over from right */}
        {workspaceOpen && (
          <div className="absolute inset-0 z-40 flex justify-end">
            <div className="absolute inset-0 bg-black/40 animate-backdrop-in" onClick={closeOverlays} />
            <aside className="relative w-[300px] max-w-[85vw] h-full flex flex-col animate-dialog-in" style={{ background: "var(--bg-primary)" }}>
              <ErrorBoundary panelName="Workspace">
                {workspacePanel}
              </ErrorBoundary>
            </aside>
          </div>
        )}
      </div>
    );
  }

  if (isTablet) {
    return (
      <div className="h-full max-h-full overflow-hidden flex relative" style={{ background: "var(--bg-secondary)" }}>
        {/* Session panel fixed */}
        <aside className="w-[240px] shrink-0 flex flex-col overflow-hidden" style={{ borderRight: "1px solid var(--border-primary)", background: "var(--bg-primary)" }}>
          <ErrorBoundary panelName="Sessions">
            {sessionPanel}
          </ErrorBoundary>
        </aside>

        {/* Conversation fills remaining */}
        <main className="flex-1 min-w-0 flex flex-col overflow-hidden relative">
          <ErrorBoundary panelName="Conversation">
            {conversationPanel}
          </ErrorBoundary>
          {/* Workspace toggle button */}
          <button
            onClick={() => setWorkspaceOpen(true)}
            className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-md border transition-colors z-10"
            style={{ background: "var(--bg-primary)", borderColor: "var(--border-primary)" }}
            title="Show workspace"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 4h4l1 1h4.5a1 1 0 011 1v6a1 1 0 01-1 1H3a1 1 0 01-1-1V5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" fill="none" />
            </svg>
          </button>
        </main>

        {/* Workspace overlay from right */}
        {workspaceOpen && (
          <div className="absolute inset-0 z-40 flex justify-end">
            <div className="absolute inset-0 bg-black/30 animate-backdrop-in" onClick={closeOverlays} />
            <aside className="relative w-[300px] h-full flex flex-col animate-dialog-in shadow-xl" style={{ background: "var(--bg-primary)" }}>
              <button
                onClick={() => setWorkspaceOpen(false)}
                className="absolute top-3 right-3 w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--bg-hover)] z-10 transition-colors"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
              <ErrorBoundary panelName="Workspace">
                {workspacePanel}
              </ErrorBoundary>
            </aside>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className="workspace-grid h-full max-h-full overflow-hidden"
      style={{
        gridTemplateColumns: `${leftWidth}px 1fr ${rightWidth}px`,
        background: "var(--bg-primary)",
      }}
    >
      <aside className="session-panel flex flex-col overflow-hidden relative" style={{ borderRight: "1px solid var(--border-primary)", background: "var(--bg-primary)" }}>
        <ErrorBoundary panelName="Sessions">
          {sessionPanel}
        </ErrorBoundary>
        <div
          className="absolute top-0 right-0 w-[5px] h-full cursor-col-resize z-10 transition-colors"
          style={{ background: "transparent" }}
          onMouseDown={(e) => onResizeStart("left", e)}
          onDoubleClick={() => onDoubleClick("left")}
          onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "var(--resize-handle)"; }}
          onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
        />
      </aside>
      <main className="conversation-panel flex flex-col overflow-hidden" style={{ background: "var(--bg-secondary)" }}>
        <ErrorBoundary panelName="Conversation">
          {conversationPanel}
        </ErrorBoundary>
      </main>
      <aside className="workspace-panel flex flex-col overflow-hidden relative" style={{ borderLeft: "1px solid var(--border-primary)", background: "var(--bg-primary)" }}>
        <div
          className="absolute top-0 left-0 w-[5px] h-full cursor-col-resize z-10 transition-colors"
          style={{ background: "transparent" }}
          onMouseDown={(e) => onResizeStart("right", e)}
          onDoubleClick={() => onDoubleClick("right")}
          onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "var(--resize-handle)"; }}
          onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
        />
        <ErrorBoundary panelName="Workspace">
          {workspacePanel}
        </ErrorBoundary>
      </aside>
    </div>
  );
}
