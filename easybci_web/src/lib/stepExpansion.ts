// The user's open/closed choice for a turn's process area, kept for the browser
// session. Absent until they toggle it; turns then follow their default.

function key(turnId: string): string {
  return `steps-expanded:${turnId}`;
}

export function readExpandChoice(turnId: string): boolean | null {
  try {
    const raw = sessionStorage.getItem(key(turnId));
    return raw === null ? null : raw === "1";
  } catch {
    return null;
  }
}

export function saveExpandChoice(turnId: string, expanded: boolean) {
  try {
    sessionStorage.setItem(key(turnId), expanded ? "1" : "0");
  } catch {
    /* private mode */
  }
}
