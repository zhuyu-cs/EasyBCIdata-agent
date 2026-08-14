import { useEffect, useRef, useState, useCallback, lazy, Suspense } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useSessionStore } from "@/stores/sessionStore";
import { MessageBubble } from "@/components/MessageBubble";
import { ChatInput } from "@/components/ChatInput";
import { Logo } from "@/components/Logo";
import { QuickStartChips } from "@/components/QuickStartChips";
// ProgressBar removed: step counter in the conversation header was anxiety-
// inducing during long pipelines. Step detail lives inline in PipelineTimeline
// on the assistant message bubble.

import { ReconnectBanner, StreamHealthDot } from "@/components/ReconnectBanner";
import { exportToMarkdown, downloadMarkdown } from "@/lib/export";
import type { Message } from "@/hooks/useConversation";
import type { RunStatus } from "@/components/StepCard";
import { StageProgressBar, type StageProgress } from "@/components/StageProgressBar";
import type { StreamStatus } from "@/lib/runsClient";
import type { TurnEta } from "@/lib/turnEta";
import { formatTotalElapsed } from "@/lib/turnEta";
import { TurnEtaIndicator } from "@/components/TurnEtaIndicator";

const ImageGallery = lazy(() => import("@/components/ImageGallery").then(m => ({ default: m.ImageGallery })));

const VIRTUAL_ENTER = 60;
const VIRTUAL_EXIT = 40;

// Heuristic per-message height estimate. Real heights are measured via
// `measureElement` after first paint; this only gates initial layout.
function estimateMessageHeight(msg: Message): number {
  const content = msg.content ?? "";
  const lines = content.split("\n").length;
  const codeBlocks = (content.match(/```/g) ?? []).length / 2;
  const tools = msg.toolCalls?.length ?? 0;
  // Base avatar+header + content lines (approx 22px/line) + code block padding + tool card.
  const base = 56;
  const text = Math.max(40, Math.min(800, lines * 22 + content.length / 8));
  const code = codeBlocks * 140;
  const toolHeight = tools >= 3 ? 32 + tools * 28 : tools * 32;
  return Math.round(base + text + code + toolHeight);
}

interface Props {
  messages: Message[];
  isStreaming: boolean;
  runStatus?: RunStatus;
  /** Live StageProgress payload from the gateway (T1.6 — null when no active run). */
  progress?: StageProgress | null;
  /** Turn-scope ETA + turn-start timestamp for the typing-row indicator. */
  latestTurnEta?: TurnEta | null;
  turnStartedAtMs?: number | null;
  error: string | null;
  onSend: (text: string) => void;
  onInterrupt: () => void;
  disabled?: boolean;
  streamStatus?: StreamStatus | null;
  onDismissStreamError?: () => void;
  onCopyStreamedOutput?: () => void;
  externalUpdateAvailable?: boolean;
  onReloadSession?: () => void;
  pendingCount?: number;
  onFlushPending?: () => void;
  onClearPending?: () => void;
}

export function ConversationPanel({
  messages,
  isStreaming,
  runStatus = "unknown",
  progress = null,
  latestTurnEta = null,
  turnStartedAtMs = null,
  error,
  onSend,
  onInterrupt,
  disabled,
  streamStatus,
  onDismissStreamError,
  onCopyStreamedOutput,
  externalUpdateAvailable,
  onReloadSession,
  pendingCount,
  onFlushPending,
  onClearPending,
}: Props) {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Index of the last assistant message — only it represents the *current* run,
  // so only it gets the live runStatus. Every earlier assistant message is
  // already closed business: pass "completed" so its retry/abandoned classes
  // stay accurate but it can never render terminal-red.
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") { lastAssistantIdx = i; break; }
  }
  const runStatusFor = (idx: number): RunStatus =>
    idx === lastAssistantIdx ? runStatus : "completed";

  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const isAutoScrolling = useRef(true);

  // Hysteresis: enter virtual mode at >60 messages, exit at <40 — prevents
  // layout thrash when the count hovers near the threshold.
  const [useVirtual, setUseVirtual] = useState(messages.length > VIRTUAL_ENTER);
  useEffect(() => {
    setUseVirtual((prev) =>
      prev ? messages.length >= VIRTUAL_EXIT : messages.length > VIRTUAL_ENTER,
    );
  }, [messages.length]);

  // ProgressBar at the top-right was removed; the inline PipelineTimeline on
  // each assistant message gives the same information without claiming the
  // header. activeToolCalls computation is therefore no longer needed.

  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (idx) => {
      const m = messages[idx];
      return m ? estimateMessageHeight(m) : 120;
    },
    getItemKey: (idx) => messages[idx]?.id ?? idx,
    overscan: 6,
    enabled: useVirtual,
    measureElement:
      typeof window !== "undefined" && "ResizeObserver" in window
        ? (el) => el.getBoundingClientRect().height
        : undefined,
  });

  const scrollToBottom = useCallback(() => {
    if (useVirtual) {
      virtualizer.scrollToIndex(messages.length - 1, { align: "end", behavior: "smooth" });
    } else {
      const el = scrollRef.current;
      if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
    isAutoScrolling.current = true;
    setShowScrollBtn(false);
  }, [useVirtual, virtualizer, messages.length]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const handleScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      const atBottom = distFromBottom < 60;
      isAutoScrolling.current = atBottom;
      setShowScrollBtn(!atBottom);
    };

    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (isAutoScrolling.current) {
      if (useVirtual) {
        virtualizer.scrollToIndex(messages.length - 1, { align: "end" });
      } else {
        const el = scrollRef.current;
        if (el) el.scrollTop = el.scrollHeight;
      }
    }
  }, [messages, useVirtual, virtualizer]);

  // Capture the total elapsed seconds at the moment the run finishes so we
  // can show a one-shot "Total Xm Ys" in place of the countdown. Cleared
  // when the next turn starts (turnStartedAtMs flips back from null).
  const [lastTurnTotalSec, setLastTurnTotalSec] = useState<number | null>(null);
  const prevStreamingRef = useRef(isStreaming);
  useEffect(() => {
    const wasStreaming = prevStreamingRef.current;
    if (wasStreaming && !isStreaming && turnStartedAtMs != null) {
      const total = Math.max(0, Math.floor((Date.now() - turnStartedAtMs) / 1000));
      setLastTurnTotalSec(total);
    } else if (isStreaming && !wasStreaming) {
      setLastTurnTotalSec(null);
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming, turnStartedAtMs]);

  const isNewSession = !activeSessionId;

  return (
    <>
      <div className="flex items-center px-5 py-3 border-b" style={{ borderColor: "var(--border-primary)" }}>
        <h2 className="text-heading-sm" style={{ color: "var(--text-primary)" }}>
          {isNewSession ? "New Conversation" : "Conversation"}
        </h2>
        {streamStatus && (streamStatus.kind === "connected" || streamStatus.kind === "connecting" || streamStatus.kind === "reconnecting" || streamStatus.kind === "failed") && (
          <span className="ml-2"><StreamHealthDot status={streamStatus} /></span>
        )}
        {messages.length > 0 && (
          <span className="ml-2 text-[11px]" style={{ color: "var(--text-muted)" }}>{messages.length} messages</span>
        )}
        {messages.length > 0 && (
          <button
            onClick={() => {
              const md = exportToMarkdown(messages);
              const filename = `easybci-session-${new Date().toISOString().slice(0, 10)}.md`;
              downloadMarkdown(md, filename);
            }}
            className="ml-2 w-6 h-6 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--text-muted)" }}
            title="Export conversation"
          >
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
              <path d="M6.5 1.5v8M6.5 9.5l-3-3M6.5 9.5l3-3M2 11.5h9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
        {isStreaming && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--accent-green)" }} />
            Processing
          </span>
        )}
      </div>

      {streamStatus && (
        <ReconnectBanner
          status={streamStatus}
          onRetry={onDismissStreamError}
          onCopyOutput={onCopyStreamedOutput}
        />
      )}

      {externalUpdateAvailable && onReloadSession && (
        <div
          className="flex items-center justify-between gap-3 px-4 py-2 border-b text-[12px] animate-fade-in"
          style={{
            background: "var(--bg-info-subtle, var(--bg-tertiary))",
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
            <span className="font-medium">Session updated externally</span>
            <span className="opacity-70">— another client modified this session</span>
          </div>
          <button
            onClick={onReloadSession}
            className="px-2 py-0.5 rounded border font-medium hover:bg-[var(--bg-hover)] transition-colors"
            style={{ borderColor: "var(--border-primary)", color: "var(--text-primary)" }}
          >
            Reload
          </button>
        </div>
      )}

      {pendingCount && pendingCount > 0 ? (
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
              {pendingCount} message{pendingCount > 1 ? "s" : ""} queued
            </span>
            <span className="opacity-70">— will retry when gateway returns</span>
          </div>
          {onFlushPending && (
            <button
              onClick={onFlushPending}
              className="px-2 py-0.5 rounded border font-medium hover:bg-[var(--bg-hover)] transition-colors"
              style={{ borderColor: "var(--border-warning)", color: "var(--text-warning)" }}
            >
              Retry now
            </button>
          )}
          {onClearPending && (
            <button
              onClick={onClearPending}
              aria-label="Dismiss queued messages"
              title="Discard queued messages"
              className="px-2 py-0.5 rounded font-medium opacity-70 hover:opacity-100 transition-opacity"
              style={{ color: "var(--text-warning)" }}
            >
              ✕
            </button>
          )}
        </div>
      ) : null}

      {error && (
        <div className="mx-5 mt-3 px-3 py-2 rounded-md border text-[12px]" style={{ background: "var(--bg-error-subtle)", borderColor: "var(--border-error)", color: "var(--text-error)" }}>
          {error}
        </div>
      )}

      <div className="relative flex-1 min-h-0">
        <div ref={scrollRef} className="absolute inset-0 overflow-y-auto px-5 py-4">
          {messages.length === 0 && !isStreaming && (
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
                <Logo size={52} className="mb-4 text-[var(--text-faint)]" />
                <p className="text-heading-sm mb-1" style={{ color: "var(--text-muted)" }}>EasyBCI Agent</p>
                <p className="text-caption max-w-[280px]" style={{ color: "var(--text-faint)" }}>
                  Describe your neural data processing task — EEG, sEEG, ECoG, MEG, spike, or fNIRS
                </p>
                <QuickStartChips onSelect={onSend} />
              </div>
            </div>
          )}

          {useVirtual ? (
            <div style={{ height: `${virtualizer.getTotalSize()}px`, width: "100%", position: "relative" }}>
              {virtualizer.getVirtualItems().map((virtualItem) => {
                const msg = messages[virtualItem.index];
                return (
                  <div
                    key={msg.id}
                    data-index={virtualItem.index}
                    ref={virtualizer.measureElement}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      transform: `translateY(${virtualItem.start}px)`,
                    }}
                    className="pb-5"
                  >
                    <MessageBubble
                      message={msg}
                      onResend={onSend}
                      runStatus={runStatusFor(virtualItem.index)}
                      progress={virtualItem.index === messages.length - 1 ? progress : null}
                      streaming={isStreaming && virtualItem.index === messages.length - 1 && msg.role === "assistant"}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="space-y-5">
              {messages.map((msg, i) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  onResend={onSend}
                  runStatus={runStatusFor(i)}
                  progress={i === messages.length - 1 ? progress : null}
                  streaming={isStreaming && i === messages.length - 1 && msg.role === "assistant"}
                />
              ))}
            </div>
          )}

          {isStreaming && progress && (
            <div className="animate-fade-in mt-5">
              <StageProgressBar progress={progress} />
            </div>
          )}

          {isStreaming && turnStartedAtMs != null && (
            <div className="flex items-center gap-1.5 text-[12px] animate-fade-in mt-5" style={{ color: "var(--text-muted)" }}>
              <span className="flex items-center gap-[3px]">
                <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
                <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
                <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
              </span>
              <span>Agent is thinking</span>
              <TurnEtaIndicator latestTurnEta={latestTurnEta} />
            </div>
          )}

          {!isStreaming && lastTurnTotalSec != null && (
            <div className="text-[12px] mt-5 animate-fade-in" style={{ color: "var(--text-muted)" }}>
              Total {formatTotalElapsed(lastTurnTotalSec)}
            </div>
          )}

          {!isStreaming && messages.length > 0 && (
            <Suspense fallback={null}>
              <div className="mt-4">
                <ImageGallery messages={messages} />
              </div>
            </Suspense>
          )}
        </div>

        {showScrollBtn && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-4 right-4 w-8 h-8 flex items-center justify-center rounded-full border transition-all duration-200 animate-fade-in z-10"
            style={{ background: "var(--bg-secondary)", borderColor: "var(--border-secondary)", boxShadow: "var(--shadow-md)" }}
            title="Scroll to bottom"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-[var(--text-primary)]">
              <path d="M7 2v10M7 12l-4-4M7 12l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>

      <ChatInput
        onSend={onSend}
        isStreaming={isStreaming}
        onInterrupt={onInterrupt}
        disabled={disabled}
        placeholder={
          disabled
            ? "Start gateway with: python run_agent.py --serve"
            : isNewSession
              ? "Describe your data processing task..."
              : "Message EasyBCI Agent..."
        }
      />
    </>
  );
}
