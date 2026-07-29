import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";

const isMac = typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad|iPod/.test(navigator.userAgent);

function isModKey(e: KeyboardEvent): boolean {
  return isMac ? e.metaKey : e.ctrlKey;
}

// True when the event originates from a text-entry control, so global
// single-key shortcuts (like "?") don't fire while the user is typing.
function isTypingTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable;
}

export interface KeyboardActions {
  onNewSession?: () => void;
  onFocusSearch?: () => void;
  onCloseOverlay?: () => void;
  onCommandPalette?: () => void;
  onExport?: () => void;
  onInterrupt?: () => void;
  onPrevSession?: () => void;
  onNextSession?: () => void;
  onShowShortcuts?: () => void;
}

export function useKeyboard(actions: KeyboardActions) {
  const setActiveSessionId = useSessionStore((s) => s.setActiveSessionId);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        actions.onCloseOverlay?.();
        return;
      }

      // "?" toggles the shortcuts overlay, but only outside text inputs and
      // without a modifier (so it doesn't clash with browser shortcuts).
      if (e.key === "?" && !isModKey(e) && !isTypingTarget(e)) {
        e.preventDefault();
        actions.onShowShortcuts?.();
        return;
      }

      if (!isModKey(e)) return;

      const key = e.key.toLowerCase();

      if (key === "k") {
        e.preventDefault();
        actions.onCommandPalette?.();
        return;
      }

      if (key === "n") {
        e.preventDefault();
        setActiveSessionId(null);
        actions.onNewSession?.();
        // Shift+N additionally drops focus into the composer.
        if (e.shiftKey) {
          requestAnimationFrame(() => {
            document.querySelector<HTMLTextAreaElement>("[data-chat-input]")?.focus();
          });
        }
        return;
      }

      if (key === "e") {
        e.preventDefault();
        actions.onExport?.();
        return;
      }

      if (key === ".") {
        e.preventDefault();
        actions.onInterrupt?.();
        return;
      }

      if (e.key === "[") {
        e.preventDefault();
        actions.onPrevSession?.();
        return;
      }

      if (e.key === "]") {
        e.preventDefault();
        actions.onNextSession?.();
        return;
      }

      if (e.key === "/") {
        e.preventDefault();
        const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
        if (searchInput) {
          searchInput.focus();
        } else {
          // Search bar is collapsed — ask SessionPanel to reveal + focus it.
          window.dispatchEvent(new Event("easybci:focus-search"));
        }
        actions.onFocusSearch?.();
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [actions, setActiveSessionId]);
}
