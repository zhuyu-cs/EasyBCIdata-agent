import { useEffect, useState } from "react";

type Highlighter = {
  codeToHtml: (code: string, options: { lang: string; theme: string }) => string;
};

let highlighterPromise: Promise<Highlighter> | null = null;
let loaded: Highlighter | null = null;

const SUPPORTED_LANGS = new Set(["python", "yaml", "json", "bash", "shell", "javascript", "typescript", "toml", "markdown"]);

// Highlighted HTML by `theme:lang:code`. A virtualized list remounts code
// blocks as they scroll in; the cache lets those remounts paint highlighted
// on the first frame instead of plain → highlighted.
const CACHE_MAX = 300;
const cache = new Map<string, string>();

function remember(key: string, html: string) {
  if (cache.size >= CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(key, html);
}

async function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = import("shiki").then(async (shiki) => {
      const hl = await shiki.createHighlighter({
        themes: ["github-light", "github-dark"],
        langs: ["python", "yaml", "json", "bash", "shell", "javascript", "typescript", "toml", "markdown"],
      });
      loaded = hl;
      return hl;
    });
  }
  return highlighterPromise;
}

function highlightNow(hl: Highlighter, key: string, code: string, lang: string, isDark: boolean): string | null {
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  try {
    const html = hl.codeToHtml(code, { lang, theme: isDark ? "github-dark" : "github-light" });
    remember(key, html);
    return html;
  } catch {
    return null;
  }
}

/**
 * Highlighted HTML for `code`, or null while unavailable. Pass `enabled=false`
 * for code that is still streaming: re-highlighting on every token is wasted
 * work and makes the block flicker between plain and highlighted.
 */
export function useHighlighter(code: string, lang: string, isDark: boolean, enabled = true): string | null {
  const supported = enabled && !!lang && SUPPORTED_LANGS.has(lang);
  const key = `${isDark ? "d" : "l"}:${lang}:${code}`;
  const [, setReady] = useState(0);

  useEffect(() => {
    if (!supported || loaded) return;
    let cancelled = false;
    getHighlighter()
      .then(() => { if (!cancelled) setReady((n) => n + 1); })
      .catch(() => { /* highlighter failed to load — render unhighlighted */ });
    return () => { cancelled = true; };
  }, [supported]);

  if (!supported || !loaded) return null;
  return highlightNow(loaded, key, code, lang, isDark);
}

export function isSupportedLang(lang: string): boolean {
  return SUPPORTED_LANGS.has(lang);
}

// Warm the shiki highlighter during browser idle time so the first code block
// renders without an await-induced flash. Safe to call repeatedly.
export function prewarmHighlighter(): void {
  if (highlighterPromise) return;
  const start = () => { getHighlighter().catch(() => { /* best-effort */ }); };
  if (typeof window === "undefined") return;
  const ric = (window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => void }).requestIdleCallback;
  if (ric) ric(start, { timeout: 3000 });
  else setTimeout(start, 1500);
}
