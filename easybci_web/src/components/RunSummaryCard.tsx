import { memo, useState } from "react";
import { fileServeUrl, type SessionArtifacts } from "@/lib/api";
import { formatDuration } from "@/lib/utils";
import { t } from "@/lib/i18n";

const QC_LABELS: Record<string, string> = {
  snr: "SNR",
  artifact_ratio: "Artifacts",
  quality_score: "Quality",
};

function qcTone(grade: string): string {
  const g = grade.toLowerCase();
  if (g.startsWith("pass") || g === "good") return "var(--text-success)";
  if (g.startsWith("fail") || g === "bad") return "var(--text-error)";
  return "var(--text-warning)";
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-[11px] leading-relaxed">
      <span className="w-[58px] shrink-0" style={{ color: "var(--text-faint)" }}>{t(label)}</span>
      <div className="min-w-0 flex-1" style={{ color: "var(--text-secondary)" }}>{children}</div>
    </div>
  );
}

interface Props {
  artifacts: SessionArtifacts;
  onOpenFile: (path: string) => void;
}

/** What the session's pipeline did, from its pipeline_record: one line until
 *  opened. Shows only what the record has; old records have less. */
export const RunSummaryCard = memo(function RunSummaryCard({ artifacts, onOpenFile }: Props) {
  const [open, setOpen] = useState(false);
  const { analysis_goal: goal, steps, qc_grade: grade, elapsed_s: elapsed, qc, figures, figure_count: figureCount } = artifacts;
  const recordPath = artifacts.pipeline_record_path;
  const stepCount = steps?.length ?? 0;
  const window = artifacts.epoch_window;
  const qcEntries = Object.entries(qc ?? {}).filter(([k, v]) => k in QC_LABELS && typeof v === "number") as [string, number][];

  if (!goal && !stepCount && !grade && !recordPath) return null;

  return (
    <div className="mx-3 mb-3 rounded-md border" style={{ borderColor: "var(--border-primary)", background: "var(--bg-code)" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={t("Run summary")}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left rounded-md hover:bg-[var(--bg-code-header)] transition-colors"
      >
        <svg
          width="10" height="10" viewBox="0 0 10 10"
          className={`shrink-0 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
          style={{ color: "var(--text-muted)" }}
          aria-hidden="true"
        >
          <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-[11px] font-medium truncate" style={{ color: "var(--text-secondary)" }}>{goal ?? t("Pipeline run")}</span>
        {stepCount > 0 && (
          <span className="text-[11px] tabular-nums shrink-0" style={{ color: "var(--text-muted)" }}>· {t("{n} steps", { n: stepCount })}</span>
        )}
        {grade && <span className="ml-auto text-[11px] font-medium shrink-0" style={{ color: qcTone(grade) }}>{t("QC {g}", { g: grade })}</span>}
      </button>
      {open && (
        <div className="px-2.5 pb-2 pt-1 space-y-1 border-t" style={{ borderColor: "var(--border-primary)" }}>
          {stepCount > 0 && (
            <Row label="Steps">
              <span className="font-mono break-words">{steps!.join(" → ")}</span>
            </Row>
          )}
          {window && <Row label="Epoch">{window[0]} – {window[1]} s</Row>}
          {artifacts.deliverables && <Row label="Outputs">{artifacts.deliverables.join(", ")}</Row>}
          {elapsed != null && <Row label="Took">{formatDuration(elapsed)}</Row>}
          {qcEntries.length > 0 && (
            <Row label="QC">
              {qcEntries.map(([k, v]) => `${t(QC_LABELS[k])} ${v.toFixed(2)}`).join(" · ")}
            </Row>
          )}
          {!!figures?.length && (
            <div className="pt-1">
              <div className="grid grid-cols-4 gap-1">
                {figures.map((path) => (
                  <button
                    key={path}
                    type="button"
                    onClick={() => onOpenFile(path)}
                    title={path.split("/").pop()}
                    className="aspect-square rounded overflow-hidden border hover:opacity-80 transition-opacity"
                    style={{ borderColor: "var(--border-primary)" }}
                  >
                    <img src={fileServeUrl(path)} alt={path.split("/").pop()} loading="lazy" className="w-full h-full object-cover" />
                  </button>
                ))}
              </div>
              {(figureCount ?? 0) > figures.length && (
                <div className="mt-1 text-[11px]" style={{ color: "var(--text-faint)" }}>
                  {t("{a} of {b} figures — the rest are in the Output folder", { a: figures.length, b: figureCount ?? 0 })}
                </div>
              )}
            </div>
          )}
          {recordPath && (
            <button
              type="button"
              onClick={() => onOpenFile(recordPath)}
              className="text-[11px] font-mono hover:underline"
              style={{ color: "var(--text-muted)" }}
            >
              pipeline_record.json
            </button>
          )}
        </div>
      )}
    </div>
  );
});
