import { memo, useCallback, useMemo, useState, type ReactNode } from "react";
import { turnSteps, type Step, type TurnItem, type TurnStatus } from "@/lib/turns";
import { formatDuration } from "@/lib/utils";
import { readExpandChoice, saveExpandChoice } from "@/lib/stepExpansion";
import { Collapse } from "./Collapse";
import { ThinkingBlock } from "./ThinkingBlock";
import { t } from "@/lib/i18n";

type StepKind = "normal" | "recovered-retry" | "abandoned" | "terminal-error";

// A tool error only becomes terminal red when the turn ended without a reply
// (failed, or a stored turn that never finished), the failing call is the last
// one, and no later same-named call recovered it. While a run is still
// streaming a transient last-step error stays grey so it doesn't flash red
// before recovery; "completed" also stays grey (the run succeeded).
function classifySteps(steps: Step[], status: TurnStatus): Map<string, StepKind> {
  const kinds = new Map<string, StepKind>();
  steps.forEach((step, i) => {
    let kind: StepKind = "normal";
    if (step.status === "error") {
      const recovered = steps.slice(i + 1).some((later) => later.tool === step.tool && later.status === "done");
      if (recovered) kind = "recovered-retry";
      else if ((status === "failed" || status === "incomplete") && i === steps.length - 1) kind = "terminal-error";
      else kind = "abandoned";
    }
    kinds.set(step.id, kind);
  });
  return kinds;
}

interface NumberedStep {
  step: Step;
  index: number;
  kind: StepKind;
  /** The reasoning that led to this call; shown when the step is opened. */
  thinking?: string;
}
type Row =
  | { type: "note"; key: string; item: Exclude<TurnItem, { kind: "step" }> }
  | { type: "step"; key: string; row: NumberedStep }
  | { type: "retries"; key: string; rows: NumberedStep[] };

/**
 * Timeline rows in order; adjacent recovered retries fold into one row. Thinking
 * ahead of a step goes inside that step, so the timeline reads as one row per
 * call; thinking with no step after it (still deciding) keeps its own row.
 */
function buildRows(items: TurnItem[], kinds: Map<string, StepKind>): Row[] {
  const rows: Row[] = [];
  let buf: NumberedStep[] = [];
  let n = 0;
  let thinking: string[] = [];
  const flush = () => {
    if (buf.length) rows.push({ type: "retries", key: `rg-${buf[0].step.id}`, rows: buf });
    buf = [];
  };
  items.forEach((item, i) => {
    if (item.kind === "thinking" && items.slice(i + 1).find((it) => it.kind !== "thinking")?.kind === "step") {
      thinking.push(item.text);
      return;
    }
    if (item.kind !== "step") {
      flush();
      rows.push({ type: "note", key: `n${i}`, item });
      return;
    }
    n += 1;
    const row: NumberedStep = {
      step: item.step,
      index: n,
      kind: kinds.get(item.step.id) ?? "normal",
      thinking: thinking.join("\n\n") || undefined,
    };
    thinking = [];
    if (row.kind === "recovered-retry") buf.push(row);
    else {
      flush();
      rows.push({ type: "step", key: item.step.id, row });
    }
  });
  flush();
  return rows;
}

// 13×13 SVGs keyed by tool name; `currentColor` so the row's colour wins.
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

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="10" height="10" viewBox="0 0 10 10"
      className="shrink-0 transition-transform duration-200"
      style={{ transform: open ? "rotate(90deg)" : "none", color: "var(--text-muted)" }}
      aria-hidden="true"
    >
      <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StatusTag({ step, kind }: { step: Step; kind: StepKind }) {
  if (step.status === "running") {
    return (
      <span className="flex items-center gap-1 text-[11px] font-medium shrink-0" style={{ color: "var(--accent-green)" }}>
        <span className="w-2.5 h-2.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
        {t("running")}
      </span>
    );
  }
  if (step.status === "error") {
    const label = kind === "terminal-error" ? t("failed") : kind === "recovered-retry" ? t("retried") : t("skipped");
    return (
      <span
        className="text-[11px] font-medium shrink-0"
        style={{ color: kind === "terminal-error" ? "var(--accent-red)" : "var(--text-muted)" }}
      >
        {label}
      </span>
    );
  }
  const d = formatDuration(step.durationSec);
  return d ? (
    <span className="text-[11px] tabular-nums shrink-0" style={{ color: "var(--text-muted)" }}>{d}</span>
  ) : null;
}

const RESULT_PREVIEW_LINES = 20;

/** Tool results are mostly JSON; indented it reads far better than one line. */
function prettyResult(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function StepResult({ step }: { step: Step }) {
  const [full, setFull] = useState(false);
  const text = useMemo(() => prettyResult(step.result ?? ""), [step.result]);
  const lines = useMemo(() => text.split("\n"), [text]);
  const clipped = !full && lines.length > RESULT_PREVIEW_LINES;
  const cut = (step.resultChars ?? 0) - (step.result?.length ?? 0);
  return (
    <div className="ml-8 mr-2 mb-1 rounded-md border" style={{ background: "var(--bg-input)", borderColor: "var(--border-primary)" }}>
      <pre
        className="px-2.5 py-1.5 text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words overflow-y-auto"
        style={{ maxHeight: full ? 480 : undefined, color: "var(--text-secondary)" }}
      >
        {clipped ? lines.slice(0, RESULT_PREVIEW_LINES).join("\n") : text}
      </pre>
      {clipped && (
        <button
          type="button"
          onClick={() => setFull(true)}
          className="w-full py-1 text-[11px] font-medium border-t hover:bg-[var(--bg-code-header)]"
          style={{ color: "var(--text-muted)", borderColor: "var(--border-primary)" }}
        >
          {t("Show all {n} lines", { n: lines.length })}
        </button>
      )}
      {cut > 0 && (
        <div className="px-2.5 py-1 text-[11px] border-t" style={{ color: "var(--text-muted)", borderColor: "var(--border-primary)" }}>
          {t("{n} more characters — the full output appears when the turn ends", { n: cut.toLocaleString() })}
        </div>
      )}
    </div>
  );
}

const StepRow = memo(function StepRow({ step, index, kind, thinking }: NumberedStep) {
  const [open, setOpen] = useState(false);
  const hasResult = !!step.result;
  const hasDetail = hasResult || !!thinking;
  const tone =
    step.status === "running"
      ? "var(--accent-green)"
      : kind === "terminal-error"
        ? "var(--accent-red)"
        : step.status === "error"
          ? "var(--text-muted)"
          : "var(--text-secondary)";

  return (
    <div className="rounded-md">
      <button
        type="button"
        onClick={() => hasDetail && setOpen((v) => !v)}
        className={`w-full flex items-center gap-2 px-2 py-1 text-left rounded-md transition-colors ${hasDetail ? "cursor-pointer hover:bg-[var(--bg-code-header)]" : "cursor-default"}`}
        aria-expanded={hasDetail ? open : undefined}
        title={step.preview}
      >
        <span className="text-[11px] tabular-nums w-4 shrink-0 text-right" style={{ color: "var(--text-faint)" }}>{index}</span>
        <span className="shrink-0" style={{ color: tone }}>{STEP_ICONS[step.tool] ?? DEFAULT_ICON}</span>
        <span className="font-mono text-[12px] truncate shrink-0 max-w-[45%]" style={{ color: "var(--text-secondary)" }}>{step.tool}</span>
        <span className="text-[11px] truncate flex-1 min-w-0" style={{ color: "var(--text-muted)" }}>{step.preview}</span>
        <StatusTag step={step} kind={kind} />
        {hasDetail && <Chevron open={open} />}
      </button>
      {hasDetail && (
        <Collapse open={open}>
          {thinking && (
            <div className="ml-8 mr-2 mb-1 px-2.5 py-1.5 rounded-md text-[11px] leading-relaxed whitespace-pre-wrap overflow-y-auto" style={{ maxHeight: 240, background: "var(--bg-input)", color: "var(--text-secondary)" }}>
              <div className="font-medium mb-0.5" style={{ color: "var(--text-muted)" }}>{t("Thinking")}</div>
              {thinking}
            </div>
          )}
          {hasResult && <StepResult step={step} />}
        </Collapse>
      )}
    </div>
  );
});

function RetryGroup({ rows }: { rows: NumberedStep[] }) {
  const [open, setOpen] = useState(false);
  const n = rows.length;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-2 py-1 text-left rounded-md hover:bg-[var(--bg-code-header)]"
        aria-expanded={open}
      >
        <span className="w-4 shrink-0" />
        <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
          &#8635; {n === 1 ? t("1 retry") : t("{n} retries", { n })}
        </span>
        <span className="flex-1" />
        <Chevron open={open} />
      </button>
      <Collapse open={open}>
        {rows.map((r) => <StepRow key={r.step.id} {...r} />)}
      </Collapse>
    </div>
  );
}

function Note({ item }: { item: Exclude<TurnItem, { kind: "step" }> }) {
  if (item.kind === "thinking") {
    return <div className="px-2 pt-1"><ThinkingBlock thinking={item.text} /></div>;
  }
  return (
    <p className="px-2 py-1 ml-6 text-[12px] leading-relaxed whitespace-pre-wrap" style={{ color: "var(--text-secondary)" }}>
      {item.text}
    </p>
  );
}

interface Props {
  /** The turn's process items — thinking, narration and steps, in order. */
  items: TurnItem[];
  status: TurnStatus;
  /** The owning turn's id — keys the remembered open/closed choice. */
  expandKey: string;
  defaultExpanded: boolean;
}

// The items array is rebuilt per flush but its elements are stable until they
// change, so the reply streaming below leaves this list alone.
function sameProcess(a: Props, b: Props): boolean {
  return (
    a.status === b.status &&
    a.expandKey === b.expandKey &&
    a.defaultExpanded === b.defaultExpanded &&
    a.items.length === b.items.length &&
    a.items.every((it, i) => it === b.items[i])
  );
}

/**
 * A turn's process area. With steps: a fixed one-line header (count, step
 * names, live state) over a timeline that follows the default for the turn's
 * place in the conversation until the user toggles it. Steps arriving never
 * change the header's shape, so a growing run doesn't jolt the conversation.
 * Without steps there is nothing to summarise, so thinking shows on its own.
 */
export const StepList = memo(function StepList({ items, status, expandKey, defaultExpanded }: Props) {
  const [override, setOverride] = useState(() => readExpandChoice(expandKey));
  const expanded = override ?? defaultExpanded;
  const toggle = useCallback(() => {
    const next = !expanded;
    setOverride(next);
    saveExpandChoice(expandKey, next);
  }, [expanded, expandKey]);

  const steps = useMemo(() => turnSteps({ items }), [items]);
  const rows = useMemo(() => buildRows(items, classifySteps(steps, status)), [items, steps, status]);

  if (!items.length) return null;
  if (!steps.length) {
    return <div className="mb-2 w-full">{rows.map((r) => r.type === "note" && <Note key={r.key} item={r.item} />)}</div>;
  }

  const n = steps.length;
  const running = steps.some((s) => s.status === "running");
  const failed = steps.some((s, i) => s.status === "error" && i === n - 1 && (status === "failed" || status === "incomplete"));
  const names = steps.map((s) => s.tool);
  const head = names.slice(0, 3).join(" → ") + (n > 3 ? ` … +${n - 3}` : "");
  const totalSec = steps.reduce((acc, s) => acc + (s.durationSec ?? 0), 0);

  return (
    <div className="mb-2 w-full rounded-md border" style={{ borderColor: "var(--border-primary)", background: "var(--bg-code)" }}>
      <button
        type="button"
        onClick={toggle}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left rounded-md hover:bg-[var(--bg-code-header)] transition-colors"
        aria-expanded={expanded}
        title={expanded ? t("Collapse steps") : t("Show all steps")}
      >
        <Chevron open={expanded} />
        <span
          className={`w-2 h-2 rounded-full shrink-0 ${running ? "animate-pulse" : ""}`}
          style={{ background: running ? "var(--accent-green)" : failed ? "var(--accent-red)" : "var(--text-muted)" }}
        />
        <span className="text-[11px] font-medium tabular-nums shrink-0" style={{ color: "var(--text-secondary)" }}>
          {n === 1 ? t("1 step") : t("{n} steps", { n })}
        </span>
        <span className="text-[11px] font-mono truncate flex-1 min-w-0" style={{ color: "var(--text-muted)" }}>{head}</span>
        {running ? (
          <span className="text-[11px] font-medium shrink-0" style={{ color: "var(--accent-green)" }}>{t("running")}</span>
        ) : failed ? (
          <span className="text-[11px] font-medium shrink-0" style={{ color: "var(--accent-red)" }}>{t("last step failed")}</span>
        ) : totalSec > 0 ? (
          <span className="text-[11px] tabular-nums shrink-0" style={{ color: "var(--text-muted)" }}>{formatDuration(totalSec)}</span>
        ) : null}
      </button>
      <Collapse open={expanded}>
        <div className="px-1 pb-1 border-t" style={{ borderColor: "var(--border-primary)" }}>
          {rows.map((r) =>
            r.type === "note" ? (
              <Note key={r.key} item={r.item} />
            ) : r.type === "retries" ? (
              <RetryGroup key={r.key} rows={r.rows} />
            ) : (
              <StepRow key={r.key} {...r.row} />
            ),
          )}
        </div>
      </Collapse>
    </div>
  );
}, sameProcess);
