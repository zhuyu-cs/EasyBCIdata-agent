/**
 * debug — lightweight diagnostic logging for swallowed errors (B13).
 *
 * Many error paths in the client are intentionally non-fatal (a failed health
 * probe, a malformed SSE frame, a localStorage quota hit). We don't want to
 * surface those as toasts — that would be noise — but swallowing them entirely
 * makes debugging painful when something genuinely breaks. This logs a single
 * tagged `console.warn` so the trace is visible in devtools without disrupting
 * the UI.
 *
 * Quiet by default in production builds; always on in dev. Can be forced on at
 * runtime with `localStorage.setItem("easybci-debug", "1")`.
 */

function isEnabled(): boolean {
  // Vite replaces import.meta.env.DEV with a literal at build time.
  if (import.meta.env?.DEV) return true;
  try {
    return typeof window !== "undefined" && window.localStorage?.getItem("easybci-debug") === "1";
  } catch {
    return false;
  }
}

/** Log a non-fatal warning with the shared `[easybci]` tag. */
export function warn(scope: string, message: string, err?: unknown): void {
  if (!isEnabled()) return;
  try {
    if (err !== undefined) {
      console.warn(`[easybci] ${scope}: ${message}`, err);
    } else {
      console.warn(`[easybci] ${scope}: ${message}`);
    }
  } catch {
    /* console unavailable — nothing else we can do */
  }
}
