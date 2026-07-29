import type { DirSummary } from "@/lib/api";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Top-N extensions of the histogram, rendered as ".edf×4 .json×1". */
function formatExtHistogram(hist: Record<string, number>, topN = 4): string {
  return Object.entries(hist)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([ext, n]) => `${ext}×${n}`)
    .join(" ");
}

/** Compact, at-a-glance Source-directory card. Default view for the
 *  Source panel — replaces the full directory tree so the common case is "read
 *  one line, done". The user expands into the tree on demand.
 *
 *  Line 1: `📁 5 files · 12.4 MB                         [Expand ▾]`
 *  Line 2: `EEG · 64 ch · 120s @ 1000Hz · .edf×4 .json×1`
 *  When the neural probe failed, line 2 degrades to just the ext histogram. */
export function SourceSummaryCard({
  summary,
  onExpand,
  onContextMenu,
}: {
  summary: DirSummary;
  onExpand: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}) {
  const neural = summary.neural;
  const extLine = formatExtHistogram(summary.ext_histogram);

  const detailParts: string[] = [];
  if (neural) {
    if (neural.modality_guess && neural.modality_guess !== "unknown") {
      detailParts.push(neural.modality_guess);
    }
    if (neural.n_channels > 0) detailParts.push(`${neural.n_channels} ch`);
    if (neural.duration_sec > 0 && neural.sample_rate_hz > 0) {
      detailParts.push(`${neural.duration_sec.toFixed(0)}s @ ${neural.sample_rate_hz.toFixed(0)}Hz`);
    } else if (neural.duration_sec > 0) {
      detailParts.push(`${neural.duration_sec.toFixed(0)}s`);
    }
  }
  if (extLine) detailParts.push(extLine);

  return (
    <div className="px-3 mb-3 animate-fade-in">
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wide">
          Source Data
        </span>
      </div>
      <div
        onContextMenu={onContextMenu}
        title={summary.path}
        className="rounded-md border border-[var(--border-primary)] bg-[var(--bg-secondary)] px-3 py-2"
      >
        <div className="flex items-center gap-2">
          <span className="text-[12px] text-[var(--text-primary)] tabular-nums">
            {summary.file_count} file{summary.file_count !== 1 ? "s" : ""}
            {summary.total_size_bytes > 0 && ` · ${formatSize(summary.total_size_bytes)}`}
          </span>
          <button
            onClick={onExpand}
            className="ml-auto shrink-0 text-[11px] text-[var(--accent-blue)] hover:underline"
          >
            Expand ▾
          </button>
        </div>
        {detailParts.length > 0 && (
          <div className="mt-1 text-[11px] text-[var(--text-muted)] truncate" title={detailParts.join(" · ")}>
            {detailParts.join(" · ")}
          </div>
        )}
        {summary.available === false && (
          <div className="mt-1 text-[11px] text-[var(--text-faint)] italic">
            Summary unavailable{summary.reason ? ` — ${summary.reason}` : ""}
          </div>
        )}
      </div>
    </div>
  );
}
