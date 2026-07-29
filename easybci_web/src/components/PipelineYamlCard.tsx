import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "@/lib/api";

interface PipelineStep {
  name: string;
  params?: Record<string, string | number | boolean>;
}

interface Props {
  yaml: string;
  /** Optional explicit goal override; when omitted the
   *  card parses ``input.analysis_goal`` from the YAML. The chip is
   *  always shown so users can spot-check what the LLM picked. */
  analysisGoal?: string | null;
  /** When set, the goal chip becomes a click-to-expand
   *  selector that posts the new value back to the parent. The parent is
   *  responsible for re-running ``propose_pipeline`` with the new value;
   *  this card never rewrites parameters on its own. */
  onGoalChange?: (goal: string) => void;
}

// Compact 12×12 SVG icons keyed by step substring. All use `currentColor`
// so they pick up the surrounding text color via CSS. No emoji.
const STEP_ICONS: { match: string; icon: ReactNode }[] = [
  // Drop non-data channels — broom / cleanup
  { match: "drop_nondata_channels", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M9 2L7 4M3 11l3-3 2 2-3 3z" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/><path d="M6 5l3 3M11 4l-2 2" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round"/></svg>
  )},
  // Notch — narrow notch in flat line
  { match: "notch_filter", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M1 6h3l1.5-3 1 6 1-3H11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
  )},
  // Bandpass — twin bars w/ band
  { match: "bandpass", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M1 9V5M11 9V5M3.5 9V3M8.5 9V3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /><path d="M3.5 6h5" stroke="currentColor" strokeWidth="1" strokeDasharray="1.5 1" /></svg>
  )},
  // Resample — circular arrow
  { match: "resample", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M10 5.5A4 4 0 102.5 7.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /><path d="M10 2.5v3h-3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
  )},
  // ICA — connected nodes (mini network)
  { match: "ica", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="2.5" cy="3" r="1.1" stroke="currentColor" strokeWidth="1" /><circle cx="9.5" cy="3" r="1.1" stroke="currentColor" strokeWidth="1" /><circle cx="6" cy="9" r="1.1" stroke="currentColor" strokeWidth="1" /><path d="M3.3 3.7l2 4.5M8.7 3.7l-2 4.5M3.5 3h5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" /></svg>
  )},
  // CAR — average baseline arrow
  { match: "car", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M1.5 8.5h9M1.5 4h2M5 4h2M8.5 4h2M6 6.5v3M4.5 8l1.5 1.5L7.5 8" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" /></svg>
  )},
  // Epoch — scissor / split
  { match: "epoch", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 9.5L9.5 2.5M2.5 2.5l7 7" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" /><circle cx="3" cy="3" r="1.2" stroke="currentColor" strokeWidth="1.1" /><circle cx="3" cy="9" r="1.2" stroke="currentColor" strokeWidth="1.1" /></svg>
  )},
  // Baseline — horizontal line
  { match: "baseline", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M1 6h10M1 9h10" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" /><path d="M1 3h10" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeDasharray="1.5 1" /></svg>
  )},
  // Reject — circle with slash
  { match: "reject", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.1" /><path d="M3 9l6-6" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" /></svg>
  )},
  // Interpolate — dotted bridge
  { match: "interpolate", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="2" cy="6" r="1" fill="currentColor" /><circle cx="10" cy="6" r="1" fill="currentColor" /><path d="M3.5 6h5" stroke="currentColor" strokeWidth="1.1" strokeDasharray="1.2 1" /></svg>
  )},
  // Pick channels — small grid
  { match: "pick_channels", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><rect x="1.5" y="1.5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" /><rect x="7.5" y="1.5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" /><rect x="1.5" y="7.5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" fill="currentColor" /><rect x="7.5" y="7.5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" /></svg>
  )},
  // Hilbert — envelope / sine
  { match: "hilbert", icon: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M1 6c1.5-3 3-3 4.5 0s3 3 4.5 0" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" /><path d="M1 6c1.5 3 3 3 4.5 0s3-3 4.5 0" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" opacity="0.4" /></svg>
  )},
];

const DEFAULT_ICON = (
  <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><circle cx="6" cy="6" r="1.5" /></svg>
);

function getIcon(name: string): ReactNode {
  const lower = name.toLowerCase();
  for (const { match, icon } of STEP_ICONS) {
    if (lower.includes(match)) return icon;
  }
  return DEFAULT_ICON;
}

function parseSteps(yaml: string): PipelineStep[] {
  const steps: PipelineStep[] = [];
  const lines = yaml.split("\n");
  let currentStep: PipelineStep | null = null;

  for (const line of lines) {
    const stepMatch = line.match(/^\s*-\s*name:\s*(.+)/);
    if (stepMatch) {
      if (currentStep) steps.push(currentStep);
      currentStep = { name: stepMatch[1].trim().replace(/^["']|["']$/g, "") };
      continue;
    }

    if (currentStep) {
      const paramMatch = line.match(/^\s+(\w+):\s*(.+)/);
      if (paramMatch && paramMatch[1] !== "name") {
        if (!currentStep.params) currentStep.params = {};
        currentStep.params[paramMatch[1]] = paramMatch[2].trim().replace(/^["']|["']$/g, "");
      }
    }
  }
  if (currentStep) steps.push(currentStep);

  return steps;
}

/** Pull ``input.analysis_goal`` out of the YAML so the goal
 *  chip works even when the parent didn't pass an explicit ``analysisGoal``
 *  prop. Tolerant of quoting; never throws. */
function parseAnalysisGoal(yaml: string): string | null {
  for (const line of yaml.split("\n")) {
    const m = line.match(/^\s*analysis_goal:\s*(.+?)\s*$/);
    if (m) {
      return m[1].trim().replace(/^["']|["']$/g, "");
    }
  }
  return null;
}

/** Cache one fetch per session — the enum doesn't change between rebuilds. */
let _goalEnumCache: string[] | null = null;
async function loadGoalEnum(): Promise<string[]> {
  if (_goalEnumCache && _goalEnumCache.length > 0) return _goalEnumCache;
  try {
    const r = await api.getAnalysisGoalEnum();
    if (Array.isArray(r.options) && r.options.length > 0) {
      _goalEnumCache = r.options;
      return r.options;
    }
  } catch {
    // fall through to defaults
  }
  // Defensive fallback so the selector still renders if /api/schema/goal-enum
  // is offline. Mirrors PLAN_PIPELINE_SCHEMA.properties.analysis_goal.enum.
  _goalEnumCache = [
    "classification",
    "source_localization",
    "feature_extraction",
    "clinical_screening",
    "exploratory",
    "generic",
  ];
  return _goalEnumCache;
}

export function PipelineYamlCard({ yaml, analysisGoal, onGoalChange }: Props) {
  const steps = useMemo(() => parseSteps(yaml), [yaml]);
  const yamlGoal = useMemo(() => parseAnalysisGoal(yaml), [yaml]);
  const currentGoal = (analysisGoal ?? yamlGoal ?? "").trim();

  // Selector is opt-in; chip is always read-only when no
  // onGoalChange handler is wired up. Default-collapsed per north-star.
  const [open, setOpen] = useState(false);
  const [enumOptions, setEnumOptions] = useState<string[] | null>(null);
  const detailsId = useId();
  const lastLoadedRef = useRef(false);

  useEffect(() => {
    if (!open || !onGoalChange || lastLoadedRef.current) return;
    lastLoadedRef.current = true;
    void loadGoalEnum().then((opts) => setEnumOptions(opts));
  }, [open, onGoalChange]);

  if (steps.length === 0) return null;

  const interactive = !!onGoalChange;
  const chipLabel = currentGoal ? `goal: ${currentGoal}` : "goal: (unspecified)";

  return (
    <div className="my-2 rounded-lg border border-[var(--border-primary,#e8e5e0)] overflow-hidden">
      <div className="px-3 py-1.5 bg-[var(--bg-tertiary,#f1f1ef)] border-b border-[var(--border-primary,#e8e5e0)] flex items-center gap-2">
        <span className="text-[11px] font-medium text-[var(--text-muted,#9b9a97)]">Pipeline</span>
        <span className="text-[10px] text-[var(--text-faint,#c4c3c0)]">{steps.length} steps</span>
        <button
          type="button"
          aria-expanded={interactive ? open : undefined}
          aria-controls={interactive ? detailsId : undefined}
          onClick={interactive ? () => setOpen((o) => !o) : undefined}
          disabled={!interactive}
          className={
            "text-[10px] px-1.5 py-0.5 rounded border tabular-nums " +
            "border-[var(--border-primary,#e8e5e0)] bg-[var(--bg-secondary,#fafaf8)] " +
            "text-[var(--text-secondary,#5f5e5b)] font-mono " +
            (interactive
              ? "cursor-pointer hover:bg-[var(--bg-hover,#ebebea)]"
              : "cursor-default")
          }
          title={
            interactive
              ? "Click to change analysis goal — re-runs propose_pipeline"
              : "Inferred analysis goal (from plan/proposal.json)"
          }
        >
          {chipLabel}
          {interactive && <span className="ml-1 opacity-50">{open ? "−" : "+"}</span>}
        </button>
        <span
          className="ml-auto text-[10px] px-1.5 py-0.5 rounded border border-emerald-300 bg-emerald-50 text-emerald-800"
          title="All randomness (numpy, random, PYTHONHASHSEED, ICA, splits) pinned to seed 42"
        >
          reproducible · seed 42
        </span>
      </div>
      {interactive && open && (
        <div
          id={detailsId}
          className="px-3 py-2 bg-[var(--bg-secondary,#fafaf8)] border-b border-[var(--border-primary,#e8e5e0)] text-[11px] text-[var(--text-secondary,#5f5e5b)]"
        >
          <p className="mb-1.5 text-[var(--text-muted,#9b9a97)]">
            Pick the downstream analysis. Leave on the LLM-inferred value to keep current pipeline; switching re-runs propose_pipeline. <span className="font-mono">generic</span> if you want broad-coverage defaults.
          </p>
          <div className="flex flex-wrap gap-1.5">
            {(enumOptions ?? []).map((opt) => {
              const active = opt === currentGoal;
              return (
                <button
                  key={opt}
                  type="button"
                  onClick={() => {
                    if (!active) onGoalChange!(opt);
                  }}
                  className={
                    "text-[10px] px-2 py-0.5 rounded border font-mono " +
                    (active
                      ? "border-[var(--border-active,#d4d3cf)] bg-[var(--bg-active,#ebebea)] text-[var(--text-primary,#37352f)]"
                      : "border-[var(--border-primary,#e8e5e0)] bg-[var(--bg-base,#ffffff)] text-[var(--text-secondary,#5f5e5b)] hover:bg-[var(--bg-hover,#ebebea)] cursor-pointer")
                  }
                  aria-pressed={active}
                  disabled={active}
                >
                  {opt}
                </button>
              );
            })}
            {enumOptions === null && (
              <span className="text-[10px] text-[var(--text-faint,#c4c3c0)]">loading…</span>
            )}
          </div>
        </div>
      )}
      <div className="p-3 space-y-1.5">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-2.5">
            <span
              className="w-5 h-5 rounded flex items-center justify-center bg-[var(--bg-tertiary,#f1f1ef)] shrink-0"
              style={{ color: "var(--text-secondary,#5f5e5b)" }}
            >
              {getIcon(step.name)}
            </span>
            <span className="text-[11px] font-medium text-[var(--text-muted,#9b9a97)] w-4 shrink-0">
              {i + 1}
            </span>
            <span className="text-[12px] font-mono text-[var(--text-secondary,#5f5e5b)]">
              {step.name}
            </span>
            {step.params && Object.entries(step.params).slice(0, 3).map(([k, v]) => (
              <span
                key={k}
                className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-hover,#ebebea)] text-[var(--text-muted,#9b9a97)] font-mono"
              >
                {k}={String(v)}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
