import { useEffect, useState } from "react";
import type { StreamStatus } from "@/lib/runsClient";

interface Props {
  status: StreamStatus | null;
  onRetry?: () => void;
  /** Optional handler to copy whatever output has streamed so far. */
  onCopyOutput?: () => void;
}

/**
 * ReconnectBanner — surfaces SSE stream health for the active run.
 *
 * Distinct from `ConnectionBanner` (which reports backend reachability).
 * - "connecting" / "connected": no banner (silent happy path); we briefly
 *   show a green "Reconnected" pill when transitioning from reconnecting → connected.
 * - "reconnecting": warning banner with attempt counter.
 * - "failed": error banner with Retry + Copy output.
 */
export function ReconnectBanner({ status, onRetry, onCopyOutput }: Props) {
  const [justReconnected, setJustReconnected] = useState(false);
  // Track the previous status so we can detect the reconnecting→connected
  // transition during render (React's "adjust state on prop change" pattern)
  // instead of synchronously setting state inside an effect.
  const [prevStatus, setPrevStatus] = useState(status);
  if (status !== prevStatus) {
    setPrevStatus(status);
    if (status?.kind === "connected") setJustReconnected(true);
  }

  // Auto-clear the "Reconnected" pill after 2s. The setState here runs in the
  // timer callback (async), which is allowed.
  useEffect(() => {
    if (!justReconnected) return;
    const t = setTimeout(() => setJustReconnected(false), 2000);
    return () => clearTimeout(t);
  }, [justReconnected]);

  if (!status) return null;

  if (status.kind === "reconnecting") {
    return (
      <div
        className="flex items-center gap-2 px-4 py-2 border-b text-[12px] animate-fade-in"
        style={{
          background: "var(--bg-warning-subtle)",
          borderColor: "var(--border-warning)",
          color: "var(--text-warning)",
        }}
        role="status"
      >
        <span
          className="w-2 h-2 rounded-full animate-pulse"
          style={{ background: "var(--text-warning)" }}
        />
        <span className="font-medium">
          Reconnecting… {status.max ? `(attempt ${status.attempt}/${status.max})` : `(attempt ${status.attempt})`}
        </span>
      </div>
    );
  }

  if (status.kind === "failed") {
    return (
      <div
        className="flex items-center justify-between gap-3 px-4 py-2 border-b text-[12px] animate-fade-in"
        style={{
          background: "var(--bg-error-subtle)",
          borderColor: "var(--border-error)",
          color: "var(--text-error)",
        }}
        role="alert"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ background: "var(--text-error)" }}
          />
          <span className="font-medium truncate">Connection lost</span>
          <span className="opacity-80 truncate">— {status.reason}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {onCopyOutput && (
            <button
              onClick={onCopyOutput}
              className="underline hover:no-underline"
              style={{ color: "var(--text-error)" }}
            >
              Copy output
            </button>
          )}
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-2 py-0.5 rounded border font-medium hover:bg-[var(--bg-hover)]"
              style={{ borderColor: "var(--border-error)", color: "var(--text-error)" }}
            >
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  if (status.kind === "connected" && justReconnected) {
    return (
      <div
        className="flex items-center gap-2 px-4 py-1.5 border-b text-[11px] animate-fade-in"
        style={{
          background: "var(--bg-success-subtle, rgba(34, 197, 94, 0.08))",
          borderColor: "var(--border-success, rgba(34, 197, 94, 0.25))",
          color: "var(--accent-green, #16a34a)",
        }}
        role="status"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M2.5 6.5l2 2 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-medium">Reconnected</span>
      </div>
    );
  }

  return null;
}

/**
 * Compact connection-quality dot for the conversation header.
 * Mirrors `StreamStatus` so callers can drop it next to the title.
 */
export function StreamHealthDot({ status }: { status: StreamStatus | null }) {
  if (!status) return null;
  let color = "var(--accent-green, #16a34a)";
  let title = "Connected";
  if (status.kind === "connecting") {
    color = "var(--text-muted)";
    title = "Connecting…";
  } else if (status.kind === "reconnecting") {
    color = "var(--text-warning)";
    title = status.max ? `Reconnecting (${status.attempt}/${status.max})` : `Reconnecting (${status.attempt})`;
  } else if (status.kind === "failed") {
    color = "var(--text-error)";
    title = `Failed: ${status.reason}`;
  }
  return (
    <span
      title={title}
      aria-label={title}
      className={`inline-block w-2 h-2 rounded-full ${
        status.kind === "reconnecting" || status.kind === "connecting" ? "animate-pulse" : ""
      }`}
      style={{ background: color }}
    />
  );
}
