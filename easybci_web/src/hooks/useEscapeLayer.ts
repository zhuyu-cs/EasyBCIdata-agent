import { useEffect, useRef } from "react";

// One document-level Escape listener shared by every overlay. Each open layer
// pushes itself; Escape closes only the top-most one, so dismissing a file
// preview can never also dismiss an approval request or the settings drawer
// underneath it.
interface Layer {
  id: number;
  handler: () => void;
}

const stack: Layer[] = [];
let nextId = 1;
let installed = false;

function install() {
  if (installed || typeof document === "undefined") return;
  installed = true;
  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "Escape" || stack.length === 0) return;
      e.preventDefault();
      e.stopPropagation();
      stack[stack.length - 1].handler();
    },
    true,
  );
}

export function hasEscapeLayer(): boolean {
  return stack.length > 0;
}

/** Register an Escape handler while `active`; the most recently activated
 *  layer wins. The handler may change between renders without re-stacking. */
export function useEscapeLayer(handler: () => void, active = true) {
  const handlerRef = useRef(handler);
  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    if (!active) return;
    install();
    const layer: Layer = { id: nextId++, handler: () => handlerRef.current() };
    stack.push(layer);
    return () => {
      const idx = stack.findIndex((l) => l.id === layer.id);
      if (idx >= 0) stack.splice(idx, 1);
    };
  }, [active]);
}
