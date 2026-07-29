import { create } from "zustand";

type Theme = "light" | "dark" | "system";

interface ThemeStore {
  theme: Theme;
  setTheme: (t: Theme) => void;
}

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "system";
  return (localStorage.getItem("easybci-theme") as Theme) || "system";
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
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
    setTheme: (t) => {
      triggerThemeTransition();
      applyTheme(t);
      set({ theme: t });
    },
  };
});

export function useThemeToggle() {
  const { theme, setTheme } = useThemeStore();
  const toggle = () => {
    const isDark =
      theme === "dark" ||
      (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setTheme(isDark ? "light" : "dark");
  };
  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  return { theme, isDark, toggle, setTheme };
}
