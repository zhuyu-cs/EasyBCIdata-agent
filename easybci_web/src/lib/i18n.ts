import { create } from "zustand";
import { zh } from "@/locales/zh";

export type Lang = "en" | "zh";

const STORAGE_KEY = "easybci-lang";

/** Same config key the backend reads (display.language); its other languages
 *  have no web catalog and show English, Chinese variants the zh one. */
export function normalizeLang(value: unknown): Lang {
  return typeof value === "string" && /^(zh|chinese|mandarin|traditional)/i.test(value) ? "zh" : "en";
}

function initialLang(): Lang {
  if (typeof window === "undefined") return "en";
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return normalizeLang(saved);
  } catch {
    /* storage disabled */
  }
  return normalizeLang(navigator.language);
}

/** App remounts under a key on this value, so everything calling t() during
 *  render picks up a language change without subscribing individually. */
export const useLangStore = create<{ lang: Lang }>(() => ({ lang: initialLang() }));

function apply(lang: Lang) {
  useLangStore.setState({ lang });
  if (typeof document !== "undefined") document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    /* storage disabled */
  }
}

apply(useLangStore.getState().lang);

export const setLang = apply;

/** The English text is the key; `{name}` placeholders are filled from `vars`.
 *  Text with no translation shows in English. */
export function t(text: string, vars?: Record<string, string | number>): string {
  let out = text;
  if (useLangStore.getState().lang === "zh") {
    const hit = zh[text];
    if (hit) out = hit;
    else if (import.meta.env.DEV) console.warn(`[i18n] no zh for: ${text}`);
  }
  return vars ? out.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : out;
}
