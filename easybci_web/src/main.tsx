import { createRoot } from "react-dom/client";
import "./index.css";
import AppRoot from "./AppRoot";
import { api } from "@/lib/api";
import { normalizeLang, setLang } from "@/lib/i18n";
import { restoreLiveRuns } from "@/lib/runManager";

// A tab opened before a rebuild asks for chunk hashes that no longer exist; a
// reload picks up the new ones. Once only: a chunk that is still missing right
// after a reload is a real failure and goes to the error boundary.
const CHUNK_RELOAD_KEY = "easybci-chunk-reload";
const CHUNK_RELOAD_WINDOW_MS = 30_000;
window.addEventListener("vite:preloadError", (event) => {
  const last = Number(sessionStorage.getItem(CHUNK_RELOAD_KEY) ?? 0);
  if (Date.now() - last < CHUNK_RELOAD_WINDOW_MS) return;
  sessionStorage.setItem(CHUNK_RELOAD_KEY, String(Date.now()));
  event.preventDefault();
  location.reload();
});

// Re-attach to runs that were still streaming when the page was reloaded.
restoreLiveRuns();

// The configured language wins over the browser's once it is known. Failing
// to read it just leaves the language already on screen.
api.getConfig().then((config) => {
  const language = (config.display as { language?: unknown } | undefined)?.language;
  if (language) setLang(normalizeLang(language));
}).catch(() => {});

createRoot(document.getElementById("root")!).render(<AppRoot />);
