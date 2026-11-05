import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState, lazy, Suspense } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useSessionStore } from "@/stores/sessionStore";
import { useSessionListStore } from "@/stores/sessionListStore";
import { getSlot, useConversationStore } from "@/stores/conversationStore";
import { TurnView } from "@/components/TurnView";
import { ChatInput } from "@/components/ChatInput";
import { Logo } from "@/components/Logo";
import { QuickStartChips } from "@/components/QuickStartChips";
import { ReconnectBanner, StreamHealthDot } from "@/components/ReconnectBanner";
import { RunStatusBar } from "@/components/RunStatusBar";
import { exportToMarkdown, downloadMarkdown } from "@/lib/export";
import { readExpandChoice } from "@/lib/stepExpansion";
import { splitTurn, turnSteps, type Turn } from "@/lib/turns";
import { conversationActions, useActiveSlot } from "@/hooks/useConversation";
import { hasEscapeLayer } from "@/hooks/useEscapeLayer";
import { useStickToBottom } from "@/hooks/useStickToBottom";
import { t } from "@/lib/i18n";

const ImageGallery = lazy(() => import("@/components/ImageGallery").then(m => ({ default: m.ImageGallery })));

const VIRTUAL_ENTER = 30;
const VIRTUAL_EXIT = 20;

// Heuristic per-turn height estimate. Real heights are measured via
// `measureElement` after first paint; this only gates initial layout.
function estimateTurnHeight(turn: Turn, isLatest: boolean): number {
  const { answer } = splitTurn(turn);
  const lines = answer.split("\n").length;
  const codeBlocks = (answer.match(/```/g) ?? []).length / 2;
  const tools = turnSteps(turn).length;
  const head = turn.user ? 72 + Math.min(200, turn.user.content.length / 4) : 0;
  const text = answer ? Math.max(40, Math.min(800, lines * 22 + answer.length / 8)) : 0;
  const code = codeBlocks * 140;
  const open = readExpandChoice(turn.id) ?? (isLatest || turn.status === "running");
  const steps = tools === 0 ? 0 : 36 + (open ? tools * 28 : 0);
  return Math.round(head + text + code + steps + 40);
}

function exportTurns(turns: Turn[]) {
  if (turns.length === 0) return;
  downloadMarkdown(exportToMarkdown(turns), `easybci-session-${new Date().toISOString().slice(0, 10)}.md`);
}

const Header = memo(function Header({
  sessionId,
  turnCount,
  onExport,
  approvalHidden,
}: {
  sessionId: string | null;
  turnCount: number;
  onExport: () => void;
  approvalHidden: boolean;
}) {
  const streamStatus = useActiveSlot("streamStatus");
  const storedTitle = useSessionListStore((s) => s.sessions.find((x) => x.id === sessionId)?.title);
  const title = !sessionId ? t("New Conversation") : storedTitle || t("Conversation");
  useEffect(() => {
    document.title = sessionId && storedTitle ? `${storedTitle} · EasyBCI` : "EasyBCI Agent";
  }, [sessionId, storedTitle]);
  return (
    <div className="flex items-center px-5 max-[1024px]:pr-14 py-3 border-b min-h-[46px]" style={{ borderColor: "var(--border-primary)" }}>
      <h2 className="text-heading-sm truncate min-w-0" style={{ color: "var(--text-primary)" }} title={title}>
        {title}
      </h2>
      {streamStatus && (
        <span className="ml-2 shrink-0"><StreamHealthDot status={streamStatus} /></span>
      )}
      {turnCount > 0 && (
        <span className="ml-2 shrink-0 text-[11px] tabular-nums" style={{ color: "var(--text-muted)" }}>{t(turnCount === 1 ? "{n} turn" : "{n} turns", { n: turnCount })}</span>
      )}
      {turnCount > 0 && (
        <button
          onClick={onExport}
          className="ml-2 shrink-0 w-6 h-6 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
          style={{ color: "var(--text-muted)" }}
          title={t("Export conversation")}
          aria-label={t("Export conversation")}
        >
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M6.5 1.5v8M6.5 9.5l-3-3M6.5 9.5l3-3M2 11.5h9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}
      {approvalHidden && (
        <button
          onClick={conversationActions.showApproval}
          className="ml-auto shrink-0 flex items-center gap-1.5 px-2 py-0.5 rounded-md border text-[11px] font-medium animate-fade-in"
          style={{ borderColor: "var(--border-warning)", color: "var(--text-warning)", background: "var(--bg-warning-subtle)" }}
          title={t("The run is waiting for your decision")}
        >
          <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--text-warning)" }} />
          {t("Approval pending · Review")}
        </button>
      )}
    </div>
  );
});

const Banners = memo(function Banners() {
  const streamStatus = useActiveSlot("streamStatus");
  const externalUpdate = useActiveSlot("externalUpdate");
  const error = useActiveSlot("error");
  const pendingCount = useConversationStore((s) => s.pending.length);
  return (
    <>
      {streamStatus && (
        <ReconnectBanner
          status={streamStatus}
          onRetry={conversationActions.retryStream}
          onCopyOutput={conversationActions.copyStreamedOutput}
        />
      )}

      {externalUpdate && (
        <div
          className="flex items-center justify-between gap-3 px-4 py-2 border-b text-[12px] animate-fade-in"
          style={{
            background: "var(--bg-tertiary)",
            borderColor: "var(--border-secondary)",
            color: "var(--text-secondary)",
          }}
          role="status"
        >
          <div className="flex items-center gap-2">
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
              <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" strokeWidth="1.2" />
              <path d="M6.5 4v3M6.5 8.5v.01" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            <span className="font-medium">{t("Session updated externally")}</span>
            <span className="opacity-70">{t("— another client modified this session")}</span>
          </div>
          <button
            onClick={conversationActions.reloadSession}
            className="px-2 py-0.5 rounded border font-medium hover:bg-[var(--bg-hover)] transition-colors"
            style={{ borderColor: "var(--border-primary)", color: "var(--text-primary)" }}
          >
            {t("Reload")}
          </button>
        </div>
      )}

      {pendingCount > 0 && (
        <div
          className="flex items-center justify-between gap-3 px-4 py-2 border-b text-[12px] animate-fade-in"
          style={{
            background: "var(--bg-warning-subtle)",
            borderColor: "var(--border-warning)",
            color: "var(--text-warning)",
          }}
          role="status"
        >
          <div className="flex items-center gap-2">
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
              <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" strokeWidth="1.2" />
              <path d="M6.5 3.5v3.5l2 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="font-medium">
              {t(pendingCount > 1 ? "{n} messages queued" : "{n} message queued", { n: pendingCount })}
            </span>
            <span className="opacity-70">{t("— will retry when gateway returns")}</span>
          </div>
          <button
            onClick={conversationActions.flushPending}
            className="px-2 py-0.5 rounded border font-medium hover:bg-[var(--bg-hover)] transition-colors"
            style={{ borderColor: "var(--border-warning)", color: "var(--text-warning)" }}
          >
            {t("Retry now")}
          </button>
          <button
            onClick={conversationActions.clearPending}
            aria-label={t("Dismiss queued messages")}
            title={t("Discard queued messages")}
            className="px-2 py-0.5 rounded font-medium opacity-70 hover:opacity-100 transition-opacity"
            style={{ color: "var(--text-warning)" }}
          >
            ✕
          </button>
        </div>
      )}

      {error && (
        <div className="mx-5 mt-3 px-3 py-2 rounded-md border text-[12px]" style={{ background: "var(--bg-error-subtle)", borderColor: "var(--border-error)", color: "var(--text-error)" }}>
          {error}
        </div>
      )}
    </>
  );
});

const QueueChips = memo(function QueueChips() {
  const queue = useActiveSlot("queue");
  if (queue.length === 0) return null;
  return (
    <div className="shrink-0 px-5 pt-2 flex flex-wrap gap-1.5" aria-label={t("Queued messages")}>
      {queue.map((q, i) => (
        <span
          key={q.id}
          className="flex items-center gap-1.5 max-w-full px-2 py-0.5 rounded-md border text-[11px] animate-fade-in"
          style={{ borderColor: "var(--border-secondary)", color: "var(--text-secondary)", background: "var(--bg-tertiary)" }}
          title={q.text}
        >
          <span style={{ color: "var(--text-muted)" }}>{i === 0 ? t("Next") : `#${i + 1}`}</span>
          <span className="truncate max-w-[360px]">{q.text}</span>
          <button
            onClick={() => conversationActions.removeQueued(q.id)}
            className="opacity-60 hover:opacity-100"
            aria-label={t("Remove queued message")}
            title={t("Remove from queue")}
          >
            ×
          </button>
        </span>
      ))}
    </div>
  );
});

function EmptyState() {
  return (
    <div className="relative flex flex-col items-center justify-center h-full text-center animate-fade-in">
      {/* Faint neural-network texture backdrop */}
      <svg
        className="pointer-events-none absolute inset-0 w-full h-full"
        style={{ opacity: 0.02, color: "var(--text-primary)" }}
        aria-hidden="true"
      >
        <defs>
          <pattern id="neural-grid" width="48" height="48" patternUnits="userSpaceOnUse">
            <circle cx="8" cy="8" r="1.5" fill="currentColor" />
            <circle cx="40" cy="24" r="1.5" fill="currentColor" />
            <circle cx="16" cy="40" r="1.5" fill="currentColor" />
            <path d="M8 8L40 24M40 24L16 40M8 8L16 40" stroke="currentColor" strokeWidth="0.6" fill="none" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#neural-grid)" />
      </svg>

      <div className="relative flex flex-col items-center">
        <Logo size={52} className="mb-4 text-[var(--text-muted)]" />
        <p className="text-heading-sm mb-1" style={{ color: "var(--text-secondary)" }}>{t("EasyBCI Agent")}</p>
        <p className="text-caption max-w-[280px]" style={{ color: "var(--text-muted)" }}>
          {t("Describe your neural data processing task — EEG, sEEG, ECoG, MEG, spike, or fNIRS")}
        </p>
        <QuickStartChips onSelect={conversationActions.send} />
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-5 animate-fade-in" aria-busy="true" aria-label={t("Loading conversation")}>
      {[0, 1, 2].map((i) => (
        <div key={i} className="space-y-2">
          <div className="h-3 w-24 rounded skeleton" />
          <div className="h-3 rounded skeleton" style={{ width: `${80 - i * 15}%` }} />
          <div className="h-3 rounded skeleton" style={{ width: `${60 - i * 10}%` }} />
        </div>
      ))}
    </div>
  );
}

/** A turn with the separator from the one before it. */
function TurnRow({ turn, index, count }: { turn: Turn; index: number; count: number }) {
  return (
    <div
      className={index > 0 ? "border-t pt-5 pb-5" : "pb-5"}
      style={index > 0 ? { borderColor: "var(--border-primary)" } : undefined}
    >
      <div className="mx-auto max-w-[880px]">
        <TurnView turn={turn} isLatest={index === count - 1} />
      </div>
    </div>
  );
}

// Its own component so only long conversations pay for the virtualizer.
function VirtualTurnList({ turns, scrollRef }: { turns: Turn[]; scrollRef: React.RefObject<HTMLDivElement | null> }) {
  // This build doesn't run the React Compiler, so its memoization warning about
  // this hook doesn't apply.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: turns.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (idx) => (turns[idx] ? estimateTurnHeight(turns[idx], idx === turns.length - 1) : 200),
    getItemKey: (idx) => turns[idx]?.id ?? idx,
    overscan: 4,
    // Start at the bottom: that's where a reopened conversation is read from.
    initialOffset: () => Number.MAX_SAFE_INTEGER,
    measureElement:
      typeof window !== "undefined" && "ResizeObserver" in window
        ? (el) => el.getBoundingClientRect().height
        : undefined,
  });
  return (
    <div style={{ height: `${virtualizer.getTotalSize()}px`, width: "100%", position: "relative" }}>
      {virtualizer.getVirtualItems().map((item) => (
        <div
          key={turns[item.index].id}
          data-index={item.index}
          ref={virtualizer.measureElement}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", transform: `translateY(${item.start}px)` }}
        >
          <TurnRow turn={turns[item.index]} index={item.index} count={turns.length} />
        </div>
      ))}
    </div>
  );
}

export function ConversationPanel({ disabled }: { disabled?: boolean }) {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const turns = useActiveSlot("turns");
  const run = useActiveSlot("run");
  const loadState = useActiveSlot("loadState");
  const progress = useActiveSlot("progress");
  const latestTurnEta = useActiveSlot("latestTurnEta");
  const lastTurnTotalSec = useActiveSlot("lastTurnTotalSec");
  const approval = useActiveSlot("approval");
  const approvalHidden = useActiveSlot("approvalHidden");
  const isStreaming = !!run;
  const isStopping = !!run?.stopping;

  const { scrollRef, contentRef, stuck, hasNew, scrollToBottom } = useStickToBottom(activeSessionId);

  // Virtual mode is decided during render (with hysteresis) so a long session
  // mounts virtualized from its first frame instead of switching after paint.
  const [useVirtual, setUseVirtual] = useState(turns.length > VIRTUAL_ENTER);
  const wantVirtual = useVirtual ? turns.length >= VIRTUAL_EXIT : turns.length > VIRTUAL_ENTER;
  if (wantVirtual !== useVirtual) setUseVirtual(wantVirtual);

  // A new session on screen means the next thing to do is type — unless a
  // dialog is open, which keeps focus until it closes.
  useEffect(() => {
    if (hasEscapeLayer()) return;
    document.querySelector<HTMLElement>("[data-chat-input]")?.focus();
  }, [activeSessionId]);

  // Switching sessions cross-fades instead of snapping. Not from "new": there
  // the id appears because the user just sent, and the view is already right.
  const shownSession = useRef(activeSessionId);
  useLayoutEffect(() => {
    const from = shownSession.current;
    shownSession.current = activeSessionId;
    if (from === activeSessionId || from === null) return;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    contentRef.current?.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 140, easing: "ease-out" });
  }, [activeSessionId, contentRef]);

  const send = useCallback((text: string) => {
    conversationActions.send(text);
    scrollToBottom();
  }, [scrollToBottom]);
  const onExport = useCallback(() => exportTurns(getSlot(activeSessionId).turns), [activeSessionId]);

  const isNewSession = !activeSessionId;
  const showSkeleton = loadState === "loading" && turns.length === 0;
  const showEmpty = turns.length === 0 && !isStreaming && !showSkeleton;

  return (
    <>
      <Header
        sessionId={activeSessionId}
        turnCount={turns.length}
        onExport={onExport}
        approvalHidden={!!approval && approvalHidden}
      />
      <Banners />

      <div className="relative flex-1 min-h-0">
        <div ref={scrollRef} className="absolute inset-0 overflow-y-auto px-5 py-4" style={{ overflowAnchor: "none" }}>
          <div ref={contentRef} className={showEmpty ? "h-full" : undefined}>
            {showSkeleton && <LoadingSkeleton />}
            {showEmpty && <EmptyState />}

            {wantVirtual ? (
              <VirtualTurnList turns={turns} scrollRef={scrollRef} />
            ) : (
              turns.map((turn, i) => <TurnRow key={turn.id} turn={turn} index={i} count={turns.length} />)
            )}

            {turns.length > 0 && (
              <Suspense fallback={null}>
                <ImageGallery turns={turns} excludeLast={isStreaming} />
              </Suspense>
            )}
          </div>
        </div>

        {!stuck && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-3 right-4 flex items-center gap-1.5 h-8 px-2.5 rounded-full border text-[11px] font-medium animate-fade-in z-10"
            style={{ background: "var(--bg-secondary)", borderColor: "var(--border-secondary)", boxShadow: "var(--shadow-md)", color: "var(--text-primary)" }}
            title={t("Scroll to bottom")}
            aria-label={t("Scroll to bottom")}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 2v10M7 12l-4-4M7 12l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {hasNew && <span>{t("New content")}</span>}
          </button>
        )}
      </div>

      {turns.length > 0 && (
        <RunStatusBar
          isStreaming={isStreaming}
          isStopping={isStopping}
          progress={progress}
          latestTurnEta={latestTurnEta}
          lastTurnTotalSec={lastTurnTotalSec}
        />
      )}

      <QueueChips />

      <ChatInput
        draftKey={activeSessionId ?? "new"}
        onSend={send}
        isStreaming={isStreaming}
        isStopping={isStopping}
        onInterrupt={conversationActions.interrupt}
        disabled={disabled}
        placeholder={
          disabled
            ? t("Gateway is not reachable — start it with: easybci dashboard")
            : isNewSession
              ? t("Describe your data processing task...")
              : t("Message EasyBCI Agent...")
        }
      />
    </>
  );
}
