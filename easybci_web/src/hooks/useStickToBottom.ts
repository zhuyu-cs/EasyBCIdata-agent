import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/** Distance from the bottom (px) that still counts as "at the bottom". */
const BOTTOM_SLACK = 4;
/** How long after a wheel/touch/key input a scroll event is attributed to the user. */
const USER_INTENT_MS = 250;

const SCROLL_KEYS = new Set(["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "]);

/**
 * Keep a scroll container pinned to its bottom while content grows.
 *
 * Only user input (wheel, touch, scroll keys, dragging the scrollbar) can
 * unpin it; programmatic scrolls and content growth never do, so a burst of
 * streamed tokens can't knock the view loose. Scrolling back to the bottom
 * re-pins. While pinned, every size change of the content jumps (no smooth
 * animation) to the bottom.
 *
 * `resetKey` re-pins and jumps to the bottom whenever it changes (e.g. the
 * session on screen).
 */
export function useStickToBottom(resetKey: unknown) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const stuckRef = useRef(true);
  const [stuck, setStuckState] = useState(true);
  // Content grew while the user was reading further up.
  const [hasNew, setHasNew] = useState(false);
  const userIntentUntil = useRef(0);
  const dragging = useRef(false);

  const setStuck = useCallback((v: boolean) => {
    stuckRef.current = v;
    setStuckState(v);
    if (v) setHasNew(false);
  }, []);

  const jump = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const scrollToBottom = useCallback(() => {
    setStuck(true);
    jump();
  }, [jump, setStuck]);

  useLayoutEffect(() => {
    stuckRef.current = true;
    jump();
    // Re-pin for the new key; the reset is the point of this effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStuckState(true);
    setHasNew(false);
  }, [resetKey, jump]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const markIntent = () => { userIntentUntil.current = performance.now() + USER_INTENT_MS; };
    const onWheel = (e: WheelEvent) => {
      markIntent();
      // Scrolling up unpins at once, before the scroll event lands.
      if (e.deltaY < 0 && stuckRef.current) setStuck(false);
    };
    const onKey = (e: KeyboardEvent) => { if (SCROLL_KEYS.has(e.key)) markIntent(); };
    // A pointerdown on the container itself (not a child) is the scrollbar.
    const onPointerDown = (e: PointerEvent) => { if (e.target === el) dragging.current = true; };
    const onPointerUp = () => { dragging.current = false; };
    const onScroll = () => {
      const userDriven = dragging.current || performance.now() < userIntentUntil.current;
      if (!userDriven) return;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_SLACK;
      if (atBottom !== stuckRef.current) setStuck(atBottom);
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("touchmove", markIntent, { passive: true });
    el.addEventListener("keydown", onKey);
    el.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchmove", markIntent);
      el.removeEventListener("keydown", onKey);
      el.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("scroll", onScroll);
    };
  }, [setStuck]);

  // Follow growth of the content (and of the viewport, e.g. the composer
  // growing) while pinned.
  useEffect(() => {
    const el = scrollRef.current;
    const content = contentRef.current;
    if (!el || !content || typeof ResizeObserver === "undefined") return;
    let lastHeight = content.getBoundingClientRect().height;
    const ro = new ResizeObserver(() => {
      const h = content.getBoundingClientRect().height;
      const grew = h > lastHeight + 0.5;
      lastHeight = h;
      if (stuckRef.current) jump();
      else if (grew) setHasNew(true);
    });
    ro.observe(content);
    ro.observe(el);
    return () => ro.disconnect();
  }, [jump]);

  return { scrollRef, contentRef, stuck, hasNew, scrollToBottom };
}
