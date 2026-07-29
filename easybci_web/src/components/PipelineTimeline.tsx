import { useState, useCallback, type ReactNode } from "react";
import type { ToolCall } from "@/hooks/useConversation";
import type { RunStatus } from "./StepCard";
import { type StageProgress } from "./StageProgressBar";

interface Props {
  toolCalls: ToolCall[];
  runStatus?: RunStatus;
  /** Stable message id — used as the sessionStorage key for the collapse state
   *  so each message remembers whether the user expanded it. Mirrors StepCard. */
  messageId?: string;
  /** Live four-stage progress payload (T1.6). */
  progress?: StageProgress | null;
}

// 13×13 SVGs keyed by tool name. `currentColor` so the dot's color rule wins.
// No emoji.
const STEP_ICONS: Record<string, ReactNode> = {
  inspect_data: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><circle cx="5.5" cy="5.5" r="3.5" stroke="currentColor" strokeWidth="1.2" /><path d="M8.2 8.2L11 11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg>
  ),
  inspect_neural: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><circle cx="5.5" cy="5.5" r="3.5" stroke="currentColor" strokeWidth="1.2" /><path d="M8.2 8.2L11 11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg>
  ),
  preprocess: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><circle cx="6.5" cy="6.5" r="1.6" stroke="currentColor" strokeWidth="1.1" /><path d="M6.5 1.5v1.6M6.5 10.4v1.6M1.5 6.5h1.6M10.4 6.5h1.6M3 3l1.1 1.1M9 9l1.1 1.1M10 3L8.9 4.1M3 10l1.1-1.1" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" /></svg>
  ),
  generate_code: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M3 1.5h5l3 3v7a1 1 0 01-1 1H3a1 1 0 01-1-1v-9a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.1" /><path d="M8 1.5v3h3M4.5 7l-1.5 1.5L4.5 10M8.5 7L10 8.5 8.5 10" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" /></svg>
  ),
  execute_code: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="currentColor"><path d="M3.5 2.5v8L10.5 6.5z" /></svg>
  ),
  export_pipeline: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 4l4.5-2.5L11 4v5l-4.5 2.5L2 9z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" /><path d="M2 4l4.5 2.5L11 4M6.5 6.5v5" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" /></svg>
  ),
  quality_check: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2.5 7l3 3 5-7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
  ),
  file_read: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M3 1.5h5l3 3v7a1 1 0 01-1 1H3a1 1 0 01-1-1v-9a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.1" /><path d="M8 1.5v3h3M4 7h5M4 9h3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" /></svg>
  ),
  file_write: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M3 1.5h5l3 3v7a1 1 0 01-1 1H3a1 1 0 01-1-1v-9a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.1" /><path d="M8 1.5v3h3" stroke="currentColor" strokeWidth="1.1" /><path d="M9 6.5L5 10.5l-2 0.5 0.5-2L7.5 5z" stroke="currentColor" strokeWidth="1" strokeLinejoin="round" /></svg>
  ),
  terminal: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none"><rect x="1.5" y="2" width="10" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" /><path d="M3.5 5l1.5 1.5L3.5 8M6 8h3.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" /></svg>
  ),
};

const DEFAULT_ICON = (
  <svg width="13" height="13" viewBox="0 0 13 13" fill="currentColor"><circle cx="6.5" cy="6.5" r="1.5" /></svg>
);

function getStepIcon(tool: string): ReactNode {
  return STEP_ICONS[tool] ?? DEFAULT_ICON;
}

function formatDuration(ms: number | undefined): string {
  if (!ms) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function PipelineTimeline({ toolCalls, runStatus = "unknown", messageId, progress: _progress = null }: Props) {
  // Expansion is purely user-driven (matches StepCard). Letting it auto-flip
  // open in response to per-step status churn caused the conversation pane to
  // strobe between collapse/expand at every tool.completed → tool.started
  // transition (brief window with zero "running" calls), and to jolt-expand at
  // the StepCard→PipelineTimeline boundary (4→5 calls). Persist per messageId
  // via sessionStorage so the user's choice survives re-renders / reloads.
  const storageKey = messageId ? `pipeline-timeline-expanded:${messageId}` : null;
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

  const running = toolCalls.some((tc) => tc.status === "running");

  if (toolCalls.length < 3) return null;

  // Build "a → b → c … (+N more)" summary from the step names.
  const names = toolCalls.map((tc) => tc.tool);
  const head = names.slice(0, 3);
  const moreCount = names.length - head.length;
  // False-fail filtering: the pipeline is only "failed" when the run
  // as a whole failed AND its last step errored — transient mid-run errors that
  // got recovered must not paint the summary red.
  const lastIdx = toolCalls.length - 1;
  const terminalFailure =
    runStatus === "failed" && toolCalls[lastIdx]?.status === "error";

  return (
    <div className="my-3 pl-1">
      <button
        onClick={toggle}
        className="flex items-center gap-2 w-full text-left rounded-md px-2 py-1.5 -ml-1 transition-colors hover:bg-[var(--bg-tertiary)]"
        title={expanded ? "Collapse steps" : "Show all steps"}
      >
        <svg
          width="11" height="11" viewBox="0 0 11 11" fill="none"
          className="shrink-0 transition-transform"
          style={{ transform: expanded ? "rotate(90deg)" : "none", color: "var(--text-muted)" }}
        >
          <path d="M4 2.5L7.5 5.5L4 8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-[11px] font-medium tabular-nums shrink-0" style={{ color: "var(--text-secondary)" }}>
          {toolCalls.length} steps
        </span>
        {!expanded && (
          <span className="text-[11px] font-mono truncate" style={{ color: "var(--text-muted)" }}>
            {head.join(" → ")}{moreCount > 0 ? ` … (+${moreCount} more)` : ""}
          </span>
        )}
        {running && (
          <span className="text-[10px] font-medium ml-auto shrink-0" style={{ color: "var(--accent-green)" }}>running</span>
        )}
        {!running && terminalFailure && (
          <span className="text-[10px] font-medium ml-auto shrink-0" style={{ color: "var(--accent-red)" }}>failed</span>
        )}
      </button>

      {expanded && (
      <div className="relative mt-1">
        {/* Vertical line */}
        <div
          className="absolute left-[11px] top-3 w-[2px] rounded-full"
          style={{
            height: `calc(100% - 24px)`,
            background: "var(--border-primary, #e8e5e0)",
          }}
        />

        <div className="space-y-0">
          {toolCalls.map((tc, i) => (
            <TimelineNode key={i} call={tc} index={i} total={toolCalls.length} terminalFailure={terminalFailure} />
          ))}
        </div>
      </div>
      )}
    </div>
  );
}

function TimelineNode({ call, index, total, terminalFailure }: { call: ToolCall; index: number; total: number; terminalFailure: boolean }) {
  const isLast = index === total - 1;
  const isRunning = call.status === "running";
  const hasReasoning = !!call.reasoning?.trim();
  const [reasoningOpen, setReasoningOpen] = useState(false);
  // Only the genuinely-terminal failing step is red; every other error is a
  // recovered/abandoned mid-run blip rendered grey.
  const isTerminalError = call.status === "error" && isLast && terminalFailure;

  return (
    <div className="flex items-start gap-3 py-1.5 relative">
      {/* Node dot — wraps the ping ring so it's perfectly centered on the dot */}
      <div className={`relative w-[22px] h-[22px] rounded-full flex items-center justify-center shrink-0 z-[1] ${
        isRunning
          ? "bg-[var(--bg-hover,#ebebea)] ring-2 ring-[var(--accent-green,#2d8a4e)] ring-opacity-50 animate-pulse text-[var(--accent-green,#2d8a4e)]"
          : isTerminalError
            ? "bg-[var(--bg-error-subtle)] text-[var(--accent-red,#d1242f)]"
            : "bg-[var(--bg-tertiary,#f1f1ef)] text-[var(--text-secondary,#5f5e5b)]"
      }`}>
        {getStepIcon(call.tool)}
        {/* Pulse ring — concentric with the dot via inset-0; no manual offset */}
        {isRunning && isLast && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-0 rounded-full border-2 border-[var(--accent-green,#2d8a4e)] opacity-30 animate-ping"
          />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 pt-0.5">
        <button
          type="button"
          onClick={() => hasReasoning && setReasoningOpen((v) => !v)}
          className={`w-full flex items-center gap-2 text-left ${hasReasoning ? "cursor-pointer" : "cursor-default"}`}
          disabled={!hasReasoning}
        >
          <span className="font-mono text-[12px] text-[var(--text-secondary,#5f5e5b)] truncate">
            {call.tool}
          </span>
          {call.duration != null && (
            <span className="text-[10px] text-[var(--text-muted,#9b9a97)] tabular-nums">
              {formatDuration(call.duration)}
            </span>
          )}
          {isRunning && (
            <span className="text-[10px] text-[var(--accent-green,#2d8a4e)] font-medium">
              running
            </span>
          )}
          {call.status === "error" && (
            <span
              className="text-[10px] font-medium"
              style={{ color: isTerminalError ? "var(--accent-red,#d1242f)" : "var(--text-muted,#9b9a97)" }}
            >
              {isTerminalError ? "failed" : "skipped"}
            </span>
          )}
          {hasReasoning && (
            <svg
              width="9" height="9" viewBox="0 0 10 10"
              className="shrink-0 ml-auto text-[var(--text-muted)] transition-transform duration-200"
              style={{ transform: reasoningOpen ? "rotate(90deg)" : "none" }}
            >
              <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
        {call.preview && (
          <p className="text-[11px] text-[var(--text-muted,#9b9a97)] truncate mt-0.5">
            {call.preview}
          </p>
        )}
        {hasReasoning && (
          <div className={`grid transition-[grid-template-rows] duration-200 ease-out ${reasoningOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
            <div className="overflow-hidden">
              <div className="mt-1.5 px-2.5 py-1.5 rounded-md bg-[var(--bg-input)] border border-[var(--border-primary)] text-[11.5px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap overflow-y-auto" style={{ maxHeight: 240 }}>
                {call.reasoning}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
