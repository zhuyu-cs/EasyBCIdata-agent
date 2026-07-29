import { useEffect, useState } from "react";

type Highlighter = {
  codeToHtml: (code: string, options: { lang: string; theme: string }) => string;
};

let highlighterPromise: Promise<Highlighter> | null = null;

const SUPPORTED_LANGS = new Set(["python", "yaml", "json", "bash", "shell", "javascript", "typescript", "toml", "markdown"]);

async function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = import("shiki").then(async (shiki) => {
      return shiki.createHighlighter({
        themes: ["github-light", "github-dark"],
        langs: ["python", "yaml", "json", "bash", "shell", "javascript", "typescript", "toml", "markdown"],
      });
    });
  }
  return highlighterPromise;
}

export function useHighlighter(code: string, lang: string, isDark: boolean): string | null {
  const supported = !!lang && SUPPORTED_LANGS.has(lang);
  // Store the highlighted HTML together with the key it was produced for, so a
  // stale result (from a previous code/lang/theme) is never returned — this lets
  // us derive the "no highlight yet" state during render instead of clearing it
  // with a synchronous setState inside the effect.
  const key = `${isDark ? "d" : "l"}:${lang}:${code}`;
  const [entry, setEntry] = useState<{ key: string; html: string } | null>(null);

  useEffect(() => {
    if (!supported) return;

    let cancelled = false;
    getHighlighter().then((hl) => {
      if (cancelled) return;
      try {
        const result = hl.codeToHtml(code, {
          lang,
          theme: isDark ? "github-dark" : "github-light",
        });
        setEntry({ key, html: result });
      } catch {
        /* leave entry as-is; unsupported tokens just render unhighlighted */
      }
    }).catch(() => { /* highlighter failed to load — render unhighlighted */ });

    return () => { cancelled = true; };
  }, [code, lang, isDark, supported, key]);

  if (!supported) return null;
  return entry && entry.key === key ? entry.html : null;
}

export function isSupportedLang(lang: string): boolean {
  return SUPPORTED_LANGS.has(lang);
}

// Warm the shiki highlighter during browser idle time so the first code block
// renders without an await-induced flash (U13). Safe to call repeatedly —
// getHighlighter() memoizes the promise. No-op if already loading/loaded.
export function prewarmHighlighter(): void {
  if (highlighterPromise) return;
  const start = () => { getHighlighter().catch(() => { /* best-effort */ }); };
  if (typeof window === "undefined") return;
  const ric = (window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => void }).requestIdleCallback;
  if (ric) ric(start, { timeout: 3000 });
  else setTimeout(start, 1500);
}
