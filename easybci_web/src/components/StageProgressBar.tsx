export type StageProgress = {
  stage: "plan" | "codegen" | "preprocess" | "qc";
  stage_index: number;
  stage_total: number;
  sub_step: string;
  sub_index: number;
  sub_total: number;
  percent: number | null;
  eta_seconds: number | null;
  confidence: "high" | "medium" | "low" | "unknown";
  elapsed_seconds: number;
  heartbeat_ts: number;
  /** SSE dispatch key. Defaults to "stage" when missing. */
  scope?: "stage" | "turn";
};

const STAGE_LABELS: Record<StageProgress["stage"], string> = {
  plan: "Plan",
  codegen: "Codegen",
  preprocess: "Preprocess",
  qc: "QC",
};

export function StageProgressBar({ progress }: { progress: StageProgress | null }) {
  if (!progress) return null;

  const pct = progress.percent ?? 0;

  return (
    <div className="px-3 py-2 rounded-md bg-[var(--bg-tertiary)] text-xs space-y-1">
      <div className="flex justify-between items-center">
        <span className="font-medium">
          {STAGE_LABELS[progress.stage]} ({progress.stage_index + 1}/{progress.stage_total})
        </span>
      </div>
      {progress.percent != null && (
        <div className="h-1.5 bg-black/10 rounded overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      <div className="text-[var(--text-muted)] truncate">{progress.sub_step || "…"}</div>
    </div>
  );
}
