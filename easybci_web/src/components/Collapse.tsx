import { useState, type ReactNode } from "react";

/**
 * Animates open and closed at any content height. Only a toggle animates: what
 * mounts already open (a reopened session, a row the virtualizer brings back)
 * appears as it is. Closed content is unmounted, so long tool output costs
 * nothing until it's asked for.
 */
export function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  const [mounted, setMounted] = useState(open);
  const [prevOpen, setPrevOpen] = useState(open);
  const [animate, setAnimate] = useState(false);
  if (open !== prevOpen) {
    setPrevOpen(open);
    setAnimate(true);
    if (open) setMounted(true);
  }
  if (!mounted) return null;
  return (
    <div
      className={`reveal ${open ? "reveal-open" : ""} ${animate ? "reveal-animate" : ""}`}
      inert={!open}
      onTransitionEnd={(e) => {
        if (e.target === e.currentTarget && !open) setMounted(false);
      }}
    >
      <div className="min-h-0 overflow-hidden">{children}</div>
    </div>
  );
}
