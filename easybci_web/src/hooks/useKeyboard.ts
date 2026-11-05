import { useEffect, useRef } from "react";
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
  onCommandPalette?: () => void;
  onExport?: () => void;
  onInterrupt?: () => void;
  onPrevSession?: () => void;
  onNextSession?: () => void;
  onShowShortcuts?: () => void;
}

export function useKeyboard(actions: KeyboardActions) {
  const setActiveSessionId = useSessionStore((s) => s.setActiveSessionId);
  // Read the latest callbacks through a ref so the document listener is
  // attached once instead of on every render of the caller. Escape is not
  // handled here — overlays register themselves via useEscapeLayer.
  const actionsRef = useRef(actions);
  useEffect(() => {
    actionsRef.current = actions;
  });

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const actions = actionsRef.current;
      // "?" toggles the shortcuts overlay, but only outside text inputs and
      // without a modifier (so it doesn't clash with browser shortcuts).
      if (e.key === "?" && !isModKey(e) && !isTypingTarget(e)) {
        e.preventDefault();
        actions.onShowShortcuts?.();
        return;
      }

      // "/" jumps to session search, under the same conditions.
      if (e.key === "/" && !isModKey(e) && !e.altKey && !isTypingTarget(e)) {
        e.preventDefault();
        const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
        if (searchInput) {
          searchInput.focus();
        } else {
          // Search bar is collapsed — ask SessionPanel to reveal + focus it.
          window.dispatchEvent(new Event("easybci:focus-search"));
        }
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
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [setActiveSessionId]);
}
