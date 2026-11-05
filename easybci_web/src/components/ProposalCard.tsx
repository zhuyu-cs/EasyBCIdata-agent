import { lazy, Suspense } from "react";
import type { ProposalShape } from "./proposalJson";
import { t } from "@/lib/i18n";

const EvidencePanel = lazy(() =>
  import("./EvidencePanel").then((m) => ({ default: m.EvidencePanel })),
);

interface Props {
  proposal: ProposalShape;
}

export function ProposalCard({ proposal }: Props) {
  return (
    <div className="my-2 rounded-lg border border-[var(--border-primary)] overflow-hidden">
      <div className="px-3 py-1.5 bg-[var(--bg-tertiary)] border-b border-[var(--border-primary)] flex items-center gap-2">
        <span className="text-[11px] font-medium text-[var(--text-muted)]">
          {t("Proposal")}
        </span>
        <span className="text-[11px] text-[var(--text-faint)]">
          {t("{n} steps", { n: proposal.steps.length })}
        </span>
        {proposal.modality && (
          <span className="text-[11px] text-[var(--text-faint)]">
            · {proposal.modality}
          </span>
        )}
        {proposal.paradigm && (
          <span className="text-[11px] text-[var(--text-faint)]">
            · {proposal.paradigm}
          </span>
        )}
      </div>
      <div className="p-3 space-y-2">
        {proposal.steps.map((step, i) => (
          <div key={i} className="text-[12px]">
            <div className="font-mono text-[var(--text-secondary)]">
              {i + 1}. {step.operator}
              {step.params &&
                Object.keys(step.params).length > 0 &&
                ` (${Object.entries(step.params)
                  .slice(0, 4)
                  .map(([k, v]) => `${k}=${String(v)}`)
                  .join(", ")})`}
            </div>
            {step.param_evidence &&
              Object.keys(step.param_evidence).length > 0 && (
                <Suspense fallback={null}>
                  <EvidencePanel
                    stepIndex={i + 1}
                    operator={step.operator}
                    evidenceByParam={step.param_evidence}
                  />
                </Suspense>
              )}
          </div>
        ))}
      </div>
    </div>
  );
}
