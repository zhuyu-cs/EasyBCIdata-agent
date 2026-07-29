import type { QCMetrics } from "./qcMetrics";

interface Props {
  metrics: QCMetrics;
}

function getLevel(key: string, value: number): "good" | "warn" | "bad" {
  if (key === "snr") return value >= 10 ? "good" : value >= 5 ? "warn" : "bad";
  if (key === "artifact_ratio") return value <= 0.1 ? "good" : value <= 0.3 ? "warn" : "bad";
  if (key === "quality_score") return value >= 0.8 ? "good" : value >= 0.5 ? "warn" : "bad";
  return value >= 0.5 ? "good" : "warn";
}

const LEVEL_STYLES = {
  good: { bg: "bg-[var(--bg-success-subtle)]", border: "border-[var(--border-success)]", text: "text-[var(--text-success)]", dot: "bg-[#22c55e]" },
  warn: { bg: "bg-[var(--bg-warning-subtle)]", border: "border-[var(--border-warning)]", text: "text-[var(--text-warning)]", dot: "bg-[#eab308]" },
  bad: { bg: "bg-[var(--bg-error-subtle)]", border: "border-[var(--border-error)]", text: "text-[var(--text-error)]", dot: "bg-[#ef4444]" },
};

const METRIC_LABELS: Record<string, string> = {
  snr: "SNR",
  artifact_ratio: "Artifact Ratio",
  quality_score: "Quality Score",
};

function formatValue(key: string, value: number): string {
  if (key === "artifact_ratio") return `${(value * 100).toFixed(1)}%`;
  if (key === "quality_score") return `${(value * 100).toFixed(0)}%`;
  return value.toFixed(2);
}

export function QCCard({ metrics }: Props) {
  const entries = Object.entries(metrics).filter(
    ([k, v]) => typeof v === "number" && (k === "snr" || k === "artifact_ratio" || k === "quality_score"),
  ) as [string, number][];

  if (entries.length === 0) return null;

  return (
    <div className="my-2 rounded-lg border border-[var(--border-primary,#e8e5e0)] overflow-hidden">
      <div className="px-3 py-1.5 bg-[var(--bg-tertiary,#f1f1ef)] border-b border-[var(--border-primary,#e8e5e0)]">
        <span className="text-[11px] font-medium text-[var(--text-muted,#9b9a97)]">Quality Check Results</span>
      </div>
      <div className="p-3 grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(entries.length, 3)}, 1fr)` }}>
        {entries.map(([key, value]) => {
          const level = getLevel(key, value);
          const styles = LEVEL_STYLES[level];
          return (
            <div
              key={key}
              className={`rounded-md px-3 py-2 border ${styles.bg} ${styles.border}`}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className={`w-2 h-2 rounded-full ${styles.dot}`} />
                <span className="text-[10px] font-medium text-[var(--text-muted,#9b9a97)] uppercase tracking-wide">
                  {METRIC_LABELS[key] || key}
                </span>
              </div>
              <div className={`text-[18px] font-semibold tabular-nums ${styles.text}`}>
                {formatValue(key, value)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
