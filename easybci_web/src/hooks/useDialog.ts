import { useEffect, useRef, useState } from "react";
import { useEscapeLayer } from "./useEscapeLayer";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), iframe, [tabindex]:not([tabindex="-1"])';

// Open dialogs, innermost last: only that one traps Tab.
const open: HTMLElement[] = [];

function focusableIn(el: HTMLElement): HTMLElement[] {
  return [...el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((n) => n.getClientRects().length > 0);
}

function onTab(e: KeyboardEvent) {
  const el = open[open.length - 1];
  if (e.key !== "Tab" || !el) return;
  const items = focusableIn(el);
  if (items.length === 0) {
    e.preventDefault();
    el.focus();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const current = document.activeElement;
  if (!el.contains(current)) {
    e.preventDefault();
    first.focus();
  } else if (e.shiftKey && (current === first || current === el)) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && current === last) {
    e.preventDefault();
    first.focus();
  }
}

/**
 * Behaviour shared by every modal: Escape closes it (through the shared layer
 * stack), Tab stays inside, focus moves in on open and returns on close — to
 * what had it before, or to the composer if that is gone. Attach the returned
 * ref to the dialog element, which also needs `tabIndex={-1}`.
 */
export function useDialog<T extends HTMLElement>(onClose: () => void, active = true) {
  const ref = useRef<T>(null);
  useEscapeLayer(onClose, active);

  // Read during first render: an autoFocus child takes focus before effects run.
  const [focusAtMount] = useState(() => (typeof document !== "undefined" ? document.activeElement : null));
  const mountedActive = useRef(active);

  useEffect(() => {
    const el = ref.current;
    if (!active || !el) return;
    const returnTo = mountedActive.current ? focusAtMount : document.activeElement;
    mountedActive.current = false;
    if (open.length === 0) document.addEventListener("keydown", onTab, true);
    open.push(el);
    if (!el.contains(document.activeElement)) (focusableIn(el)[0] ?? el).focus();
    return () => {
      open.splice(open.indexOf(el), 1);
      if (open.length === 0) document.removeEventListener("keydown", onTab, true);
      const target =
        returnTo instanceof HTMLElement && returnTo !== document.body && returnTo.isConnected
          ? returnTo
          : document.querySelector<HTMLElement>("[data-chat-input]");
      target?.focus();
    };
  }, [active, focusAtMount]);

  return ref;
}
