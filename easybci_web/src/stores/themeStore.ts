import { create } from "zustand";

type Theme = "light" | "dark" | "system";

interface ThemeStore {
  theme: Theme;
  /** Resolved appearance — follows the OS while `theme` is "system". */
  isDark: boolean;
  setTheme: (t: Theme) => void;
}

const darkQuery =
  typeof window !== "undefined" ? window.matchMedia("(prefers-color-scheme: dark)") : null;

function resolveDark(theme: Theme): boolean {
  return theme === "dark" || (theme === "system" && !!darkQuery?.matches);
}

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "system";
  return (localStorage.getItem("easybci-theme") as Theme) || "system";
}

function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", resolveDark(theme) ? "dark" : "light");
  localStorage.setItem("easybci-theme", theme);
}

let transitionTimer: ReturnType<typeof setTimeout> | null = null;

// Briefly enable color transitions on <html> so theme changes animate smoothly
// instead of snapping. Only used for user-initiated changes (not initial load).
function triggerThemeTransition() {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.setAttribute("data-theme-transitioning", "");
  if (transitionTimer) clearTimeout(transitionTimer);
  transitionTimer = setTimeout(() => {
    root.removeAttribute("data-theme-transitioning");
    transitionTimer = null;
  }, 200);
}

export const useThemeStore = create<ThemeStore>((set) => {
  const initial = getInitialTheme();
  if (typeof window !== "undefined") applyTheme(initial);
  return {
    theme: initial,
    isDark: resolveDark(initial),
    setTheme: (t) => {
      triggerThemeTransition();
      applyTheme(t);
      set({ theme: t, isDark: resolveDark(t) });
    },
  };
});

darkQuery?.addEventListener("change", () => {
  const { theme } = useThemeStore.getState();
  if (theme === "system") applyTheme(theme);
  useThemeStore.setState({ isDark: resolveDark(theme) });
});

export function useThemeToggle() {
  const theme = useThemeStore((s) => s.theme);
  const isDark = useThemeStore((s) => s.isDark);
  const setTheme = useThemeStore((s) => s.setTheme);
  const toggle = () => setTheme(isDark ? "light" : "dark");
  return { theme, isDark, toggle, setTheme };
}
