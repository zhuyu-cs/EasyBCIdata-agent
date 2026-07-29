import { useCallback, useRef, useState, useEffect } from "react";

const STORAGE_KEY = "easybci-panel-widths";
const DEFAULTS = { left: 260, right: 280 };
const MIN_WIDTH = 180;
const MAX_WIDTH = 400;

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
  localStorage.setItem(STORAGE_KEY, JSON.stringify(widths));
}

export function useResizeHandle() {
  const [widths, setWidths] = useState(loadWidths);
  const draggingRef = useRef<"left" | "right" | null>(null);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMouseDown = useCallback((side: "left" | "right", e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = side;
    startXRef.current = e.clientX;
    startWidthRef.current = side === "left" ? widths.left : widths.right;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [widths]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const delta = e.clientX - startXRef.current;
      const side = draggingRef.current;
      const newWidth = clamp(
        side === "left"
          ? startWidthRef.current + delta
          : startWidthRef.current - delta,
      );
      setWidths((prev) => {
        const next = { ...prev, [side]: newWidth };
        saveWidths(next);
        return next;
      });
    };

    const handleMouseUp = () => {
      if (draggingRef.current) {
        draggingRef.current = null;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
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
    onResizeStart: handleMouseDown,
    onDoubleClick: handleDoubleClick,
  };
}
