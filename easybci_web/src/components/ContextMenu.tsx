import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface ContextMenuItem {
  label: string;
  action: () => void;
  danger?: boolean;
}

interface Props {
  x: number;
  y: number;
  items: ContextMenuItem[];
  onClose: () => void;
}

export function ContextMenu({ x, y, items, onClose }: Props) {
  const menuRef = useRef<HTMLDivElement>(null);
  // Corrected coordinates after measuring the menu box. Seed with the raw
  // cursor position so the first paint is already close.
  const [pos, setPos] = useState({ left: x, top: y });

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [onClose]);

  // Measure AFTER layout (menu is portaled to body, no ancestor transform, no
  // translate animation) and clamp to the viewport on all four edges.
  useLayoutEffect(() => {
    const menu = menuRef.current;
    if (!menu) return;
    const rect = menu.getBoundingClientRect();
    const margin = 8;
    let left = x;
    let top = y;
    if (left + rect.width > window.innerWidth) left = x - rect.width;
    if (top + rect.height > window.innerHeight) top = y - rect.height;
    if (left < margin) left = margin;
    if (top < margin) top = margin;
    setPos({ left, top });
  }, [x, y]);

  return createPortal(
    <div
      ref={menuRef}
      className="fixed z-50 bg-[var(--bg-secondary)] rounded-lg shadow-lg border border-[var(--border-primary)] py-1 min-w-[160px] animate-fade-in-opacity"
      style={{ left: pos.left, top: pos.top }}
    >
      {items.map((item, i) => (
        <button
          key={i}
          onClick={() => {
            item.action();
            onClose();
          }}
          className={`w-full text-left px-3 py-1.5 text-[12px] transition-colors ${
            item.danger
              ? "text-[var(--accent-red)] hover:bg-[var(--bg-error-subtle)]"
              : "text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]"
          }`}
        >
          {item.label}
        </button>
      ))}
    </div>,
    document.body,
  );
}
