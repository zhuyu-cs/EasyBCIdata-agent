// Turn-scope ETA helpers for the WebUI.
//
// EWMA smoothing matches the CLI side (easybci_lib/cli_eta.py) to keep both
// UIs visually consistent. Output is countdown-only — when the remaining
// counter hits zero we just sit at `+0s` until the next ETA arrives; there
// is no "over by Xs" branch and no low-confidence `~?` qualifier.

export type TurnEta = {
  /** Smoothed ETA seconds — what we currently believe the wait is. */
  smoothedSeconds: number;
  /** ms since epoch when the latest emit landed. */
  emittedAtMs: number;
  /** Confidence band (kept on the type for upstream filtering, but no
   *  longer changes the rendered text). */
  confidence: "high" | "medium" | "low" | "unknown";
};

const DEFAULT_ALPHA = 0.4;

export function applyEwma(newSeconds: number, prev: number | null, alpha = DEFAULT_ALPHA): number {
  if (prev === null) return Math.round(newSeconds);
  return Math.round(alpha * newSeconds + (1 - alpha) * prev);
}

export function computeRemainingSeconds(eta: TurnEta | null, nowMs: number): number | null {
  if (!eta) return null;
  const elapsedMs = nowMs - eta.emittedAtMs;
  const remaining = eta.smoothedSeconds - Math.floor(elapsedMs / 1000);
  return Math.max(0, remaining);
}

export function formatEtaText(remaining: number | null): string {
  if (remaining === null) return "";
  const m = Math.floor(remaining / 60);
  const s = remaining % 60;
  if (m > 0) return `+${m}m ${s}s`;
  return `+${s}s`;
}

export function formatTotalElapsed(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const m = Math.floor(safe / 60);
  const s = safe % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
