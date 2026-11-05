import { useCallback, useState } from "react";

const STORAGE_KEY = "easybci-panel-widths";
const DEFAULTS = { left: 260, right: 280 };
const MIN_WIDTH = 180;
const MAX_WIDTH = 400;
const KEY_STEP = 16;

function loadWidths(): { left: number; right: number } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        left: clamp(parsed.left ?? DEFAULTS.left),
        right: clamp(parsed.right ?? DEFAULTS.right),
      };
    }
  } catch { /* ignore */ }
  return { ...DEFAULTS };
}

function clamp(v: number): number {
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, v));
}

function saveWidths(widths: { left: number; right: number }) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(widths));
  } catch { /* storage disabled — widths just won't persist */ }
}

/**
 * Drag-to-resize for the side panels. Pointer capture keeps the drag alive
 * when the cursor leaves the thin handle (or the window); width updates are
 * batched to one per animation frame and persisted once, when the drag ends.
 */
export function useResizeHandle() {
  const [widths, setWidths] = useState(loadWidths);

  const handlePointerDown = useCallback((side: "left" | "right", e: React.PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const handle = e.currentTarget;
    const pointerId = e.pointerId;
    const startX = e.clientX;
    const startWidth = widths[side];
    let frame: number | null = null;
    let latestX = startX;
    let committed = startWidth;

    handle.setPointerCapture(pointerId);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.body.classList.add("panel-resizing");

    const apply = () => {
      frame = null;
      const delta = latestX - startX;
      committed = clamp(side === "left" ? startWidth + delta : startWidth - delta);
      setWidths((prev) => (prev[side] === committed ? prev : { ...prev, [side]: committed }));
    };
    const onMove = (ev: PointerEvent) => {
      latestX = ev.clientX;
      if (frame === null) frame = requestAnimationFrame(apply);
    };
    const onEnd = () => {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        apply();
      }
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onEnd);
      handle.removeEventListener("pointercancel", onEnd);
      if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.body.classList.remove("panel-resizing");
      saveWidths({ ...widths, [side]: committed });
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onEnd);
    handle.addEventListener("pointercancel", onEnd);
  }, [widths]);

  // Left/right move the handle itself, so the same key grows one panel and
  // shrinks the other.
  const handleKeyDown = useCallback((side: "left" | "right", e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const moved = e.key === "ArrowRight" ? KEY_STEP : -KEY_STEP;
    setWidths((prev) => {
      const next = { ...prev, [side]: clamp(prev[side] + (side === "left" ? moved : -moved)) };
      saveWidths(next);
      return next;
    });
  }, []);

  const handleDoubleClick = useCallback((side: "left" | "right") => {
    setWidths((prev) => {
      const next = { ...prev, [side]: DEFAULTS[side] };
      saveWidths(next);
      return next;
    });
  }, []);

  return {
    leftWidth: widths.left,
    rightWidth: widths.right,
    onResizeStart: handlePointerDown,
    onResizeKey: handleKeyDown,
    onDoubleClick: handleDoubleClick,
  };
}
