import { memo } from "react";
import type { StageProgress } from "@/lib/stageProgress";
import { TurnEtaIndicator } from "./TurnEtaIndicator";
import { formatTotalElapsed, type TurnEta } from "@/lib/turnEta";
import { t } from "@/lib/i18n";

const STAGE_LABELS: Record<StageProgress["stage"], string> = {
  plan: "Plan",
  codegen: "Codegen",
  preprocess: "Preprocess",
  qc: "QC",
  train: "Train",
  evaluate: "Evaluate",
};

interface Props {
  isStreaming: boolean;
  isStopping: boolean;
  progress: StageProgress | null;
  latestTurnEta: TurnEta | null;
  lastTurnTotalSec: number | null;
}

/**
 * The single "what is the agent doing" line under the conversation. Always
 * mounted at a fixed height — its content swaps between thinking / stage
 * progress / stopping / last-turn time, so a turn starting or ending never
 * shifts the messages above it.
 */
export const RunStatusBar = memo(function RunStatusBar({
  isStreaming,
  isStopping,
  progress,
  latestTurnEta,
  lastTurnTotalSec,
}: Props) {
  let body: React.ReactNode = null;
  if (isStreaming) {
    const pct = progress?.percent;
    body = (
      <>
        <span className="flex items-center gap-[3px] shrink-0" aria-hidden="true">
          <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
          <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
          <span className="w-[5px] h-[5px] rounded-full typing-dot" style={{ background: "var(--text-muted)" }} />
        </span>
        {isStopping ? (
          <span>{t("Stopping…")}</span>
        ) : progress ? (
          <>
            <span className="font-medium shrink-0" style={{ color: "var(--text-secondary)" }}>
              {STAGE_LABELS[progress.stage] ? t(STAGE_LABELS[progress.stage]) : progress.stage} {progress.stage_index + 1}/{progress.stage_total}
            </span>
            {pct != null && (
              <span className="w-16 h-1 rounded overflow-hidden shrink-0" style={{ background: "var(--bg-active)" }}>
                <span className="block h-full transition-[width] duration-300" style={{ width: `${pct}%`, background: "var(--accent-green)" }} />
              </span>
            )}
            {progress.sub_step && <span className="truncate min-w-0">{progress.sub_step}</span>}
          </>
        ) : (
          <span>{t("Agent is thinking")}</span>
        )}
        {!isStopping && <TurnEtaIndicator latestTurnEta={latestTurnEta} />}
      </>
    );
  } else if (lastTurnTotalSec != null) {
    body = <span>{t("Last turn took {d}", { d: formatTotalElapsed(lastTurnTotalSec) })}</span>;
  }

  return (
    <div
      className="shrink-0 h-7 px-5 flex items-center gap-2 text-[12px] min-w-0 overflow-hidden"
      style={{ color: "var(--text-muted)" }}
      role="status"
      aria-live="polite"
    >
      {body}
    </div>
  );
});
