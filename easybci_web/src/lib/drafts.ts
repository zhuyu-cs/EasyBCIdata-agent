/**
 * Unsent composer text, per session. In sessionStorage so a reload keeps it and
 * two tabs don't overwrite each other's.
 */
const PREFIX = "easybci-draft:";

export function readDraft(key: string): string {
  try {
    return sessionStorage.getItem(PREFIX + key) ?? "";
  } catch {
    return "";
  }
}

export function saveDraft(key: string, text: string) {
  try {
    if (text) sessionStorage.setItem(PREFIX + key, text);
    else sessionStorage.removeItem(PREFIX + key);
  } catch {
    // Storage full or blocked: the draft just doesn't outlive this page.
  }
}
