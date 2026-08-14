import { useState, useCallback } from "react";
import type { ToolCall } from "@/hooks/useConversation";
import { type StageProgress } from "./StageProgressBar";

export type RunStatus = "running" | "completed" | "failed" | "unknown";

type StepKind = "normal" | "recovered-retry" | "abandoned" | "terminal-error";
interface ClassifiedCall { call: ToolCall; kind: StepKind; }

// classifyCalls: tool errors only become terminal red when the run is NOT
// actively streaming AND the failing call is the last one AND no later
// same-named call recovered it. "failed" (live run that errored out) and
// "unknown" (reloaded history — the run is already over, so a persisted
// last un-recovered error genuinely WAS terminal) both qualify. Only while a
// run is still streaming ("running") must a transient last-step error stay
// grey so it doesn't flash red before recovery. "completed" also stays grey
// (the run succeeded overall).
function classifyCalls(toolCalls: ToolCall[], runStatus: RunStatus): ClassifiedCall[] {
  const lastIdx = toolCalls.length - 1;
  return toolCalls.map((call, i) => {
    if (call.status !== "error") return { call, kind: "normal" as StepKind };
    const recovered = toolCalls
      .slice(i + 1)
      .some((later) => later.tool === call.tool && later.status === "done");
    if (recovered) return { call, kind: "recovered-retry" as StepKind };
    if ((runStatus === "failed" || runStatus === "unknown") && i === lastIdx) {
      return { call, kind: "terminal-error" as StepKind };
    }
    // running / completed, or a non-last error → grey.
    return { call, kind: "abandoned" as StepKind };
  });
}

type RenderRow =
  | { kind: "item"; classified: ClassifiedCall; origIndex: number }
  | { kind: "retry-group"; rows: { classified: ClassifiedCall; origIndex: number }[] };

function collapseRetries(classified: ClassifiedCall[]): RenderRow[] {
  const out: RenderRow[] = [];
  let buf: { classified: ClassifiedCall; origIndex: number }[] = [];
  const flush = () => {
    if (buf.length) {
      out.push({ kind: "retry-group", rows: buf });
      buf = [];
    }
  };
  classified.forEach((c, i) => {
    if (c.kind === "recovered-retry") {
      buf.push({ classified: c, origIndex: i });
    } else {
      flush();
      out.push({ kind: "item", classified: c, origIndex: i });
    }
  });
  flush();
  return out;
}

interface Props {
  toolCalls: ToolCall[];
  runStatus?: RunStatus;
  /** Stable message id — used as the sessionStorage key for the collapse
   *  state so each message remembers whether the user expanded it. */
  messageId?: string;
  /** Live four-stage progress payload (T1.6) — null when no run is active or
   *  the gateway hasn't emitted a progress event yet. Forwarded through props
   *  but not currently rendered (the "Plan (1/4)" mini-card UI was removed). */
  progress?: StageProgress | null;
}

function StepStatus({ status, duration, kind }: { status: ToolCall["status"]; duration?: number; kind: StepKind }) {
  if (status === "running") {
    return (
      <span className="flex items-center gap-1 text-[var(--text-muted)]">
        <span className="w-3 h-3 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin" />
        <span className="text-[11px]">Running</span>
      </span>
    );
  }
  if (status === "error") {
    if (kind === "terminal-error") {
      return (
        <span className="flex items-center gap-1 text-[var(--accent-red)]">
          <span className="text-sm">&#10007;</span>
          <span className="text-[11px]">Failed</span>
        </span>
      );
    }
    if (kind === "recovered-retry") {
      return (
        <span className="flex items-center gap-1 text-[var(--text-muted)]">
          <span className="text-sm">&#8635;</span>
          <span className="text-[11px]">Retried</span>
        </span>
      );
    }
    // abandoned
    return (
      <span className="flex items-center gap-1 text-[var(--text-muted)]">
        <span className="text-sm">&#8635;</span>
        <span className="text-[11px]">Skipped</span>
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-[var(--accent-green)]">
      <span className="text-sm">&#10003;</span>
      {duration != null && <span className="text-[11px]">{duration.toFixed(1)}s</span>}
    </span>
  );
}

function StepDot({ status, kind }: { status: ToolCall["status"]; kind: StepKind }) {
  if (status === "running") {
    return <span className="w-2 h-2 rounded-full shrink-0 bg-[var(--text-muted)] animate-pulse" />;
  }
  if (status === "error") {
    if (kind === "terminal-error") {
      return <span className="w-2 h-2 rounded-full shrink-0 bg-[var(--accent-red)]" />;
    }
    return <span className="w-2 h-2 rounded-full shrink-0 bg-[var(--text-muted)]" />;
  }
  return <span className="w-2 h-2 rounded-full shrink-0 bg-[var(--accent-green)]" />;
}

export function StepCard({ toolCalls, runStatus = "unknown", messageId, progress: _progress = null }: Props) {
  // Group collapse: the whole step list defaults to a single line so
  // the conversation pane stops "jumping" as tool.started/completed events grow
  // the array. Expansion is purely user-driven and remembered per message via
  // sessionStorage — NO effect ever flips this open in response to new events
  // or status changes.
  const storageKey = messageId ? `stepcard-expanded:${messageId}` : null;
  const [expanded, setExpanded] = useState<boolean>(() => {
    if (!storageKey) return false;
    try { return sessionStorage.getItem(storageKey) === "1"; } catch { return false; }
  });
  const toggle = useCallback(() => {
    setExpanded((prev) => {
      const next = !prev;
      if (storageKey) {
        try { sessionStorage.setItem(storageKey, next ? "1" : "0"); } catch { /* private mode */ }
      }
      return next;
    });
  }, [storageKey]);

  if (!toolCalls.length) return null;
  const classified = classifyCalls(toolCalls, runStatus);
  const grouped = collapseRetries(classified);

  // Overall status for the collapsed header dot.
  const hasTerminalError = classified.some((c) => c.kind === "terminal-error");
  const anyRunning = toolCalls.some((c) => c.status === "running");
  const overall: ToolCall["status"] = hasTerminalError ? "error" : anyRunning ? "running" : "done";
  const n = toolCalls.length;

  if (!expanded) {
    return (
      <div className="mb-2">
        <button
          onClick={toggle}
          className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md border border-[var(--border-primary)] bg-[var(--bg-code)] hover:bg-[var(--bg-code-header)] transition-colors duration-150 cursor-pointer"
        >
          <StepDot status={overall} kind={hasTerminalError ? "terminal-error" : "normal"} />
          <span className="text-[11px] text-[var(--text-muted)] flex-1">
            {n} step{n !== 1 ? "s" : ""} · click to expand
            {hasTerminalError && <span className="text-[var(--accent-red)]"> · last failed</span>}
          </span>
          <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0 text-[var(--text-muted)]">
            <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="mb-2 relative">
      <button
        onClick={toggle}
        className="flex items-center gap-1.5 px-2 py-1 mb-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors cursor-pointer"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0 rotate-90">
          <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        click to collapse
      </button>
      {/* Connecting line */}
      {grouped.length > 1 && (
        <div
          className="absolute left-[18px] top-[44px] w-[2px] bg-[var(--bg-active)] rounded-full"
          style={{ height: `calc(100% - 64px)` }}
        />
      )}
      <div className="space-y-1.5 relative">
        {grouped.map((row) => {
          if (row.kind === "retry-group") {
            return (
              <RetryGroup
                key={`rg-${row.rows[0].origIndex}`}
                rows={row.rows}
                toolCallsLength={toolCalls.length}
              />
            );
          }
          return (
            <StepItem
              key={row.origIndex}
              call={row.classified.call}
              index={row.origIndex + 1}
              isLast={row.origIndex === toolCalls.length - 1}
              kind={row.classified.kind}
            />
          );
        })}
      </div>
    </div>
  );
}

function StepItem({ call, index, isLast, kind }: { call: ToolCall; index: number; isLast: boolean; kind: StepKind }) {
  const [expanded, setExpanded] = useState(false);
  const hasReasoning = !!call.reasoning?.trim();

  return (
    <div className="animate-fade-in rounded-md border border-[var(--border-primary)] bg-[var(--bg-code)] overflow-hidden">
      <button
        onClick={() => hasReasoning && setExpanded(!expanded)}
        className={`w-full flex items-center gap-2 px-3 py-2 text-left ${hasReasoning ? "cursor-pointer hover:bg-[var(--bg-code-header)]" : "cursor-default"} transition-colors duration-150`}
      >
        <StepDot status={call.status} kind={kind} />
        <span className="text-[11px] font-medium text-[var(--text-muted)] w-4 shrink-0">
          {index}
        </span>
        <span className="font-mono text-[12px] text-[var(--text-secondary)] truncate flex-1">
          {call.tool}
        </span>
        {call.preview && (
          <span className="text-[11px] text-[var(--text-muted)] truncate max-w-[40%] hidden sm:inline">
            {call.preview}
          </span>
        )}
        <StepStatus status={call.status} duration={call.duration} kind={kind} />
        {hasReasoning && (
          <svg
            width="10" height="10" viewBox="0 0 10 10"
            className={`shrink-0 text-[var(--text-muted)] transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
          >
            <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      {/* Shimmer progress bar for running state */}
      {call.status === "running" && isLast && (
        <div className="h-[2px] w-full overflow-hidden">
          <div className="h-full w-full animate-shimmer" style={{ background: "var(--gradient-shimmer)" }} />
        </div>
      )}

      <div className={`grid transition-[grid-template-rows] duration-200 ease-out ${expanded && hasReasoning ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="px-3 pb-2.5 pt-0 border-t border-[var(--border-primary)]">
            <div className="mt-2 text-[12px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
              {call.reasoning}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RetryGroup({
  rows,
  toolCallsLength,
}: {
  rows: { classified: ClassifiedCall; origIndex: number }[];
  toolCallsLength: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const n = rows.length;

  return (
    <div className="animate-fade-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md border border-dashed border-[var(--border-primary)] bg-[var(--bg-code)] hover:bg-[var(--bg-code-header)] transition-colors duration-150 cursor-pointer"
      >
        <span className="w-2 h-2 rounded-full shrink-0 bg-[var(--text-muted)]" />
        <span className="text-[11px] text-[var(--text-muted)] flex-1">
          <span className="text-sm align-middle">&#8635;</span>{" "}
          {n} {n === 1 ? "retry" : "retries"} · click to {expanded ? "hide" : "show"}
        </span>
        <svg
          width="10" height="10" viewBox="0 0 10 10"
          className={`shrink-0 text-[var(--text-muted)] transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <div className={`grid transition-[grid-template-rows] duration-200 ease-out ${expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="space-y-1.5 mt-1.5">
            {rows.map(({ classified, origIndex }) => (
              <StepItem
                key={origIndex}
                call={classified.call}
                index={origIndex + 1}
                isLast={origIndex === toolCallsLength - 1}
                kind={classified.kind}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
