import { lazy, Suspense } from "react";

const EvidencePanel = lazy(() =>
  import("./EvidencePanel").then((m) => ({ default: m.EvidencePanel })),
);

interface ProposalStep {
  operator: string;
  params?: Record<string, unknown>;
  param_evidence?: Record<string, import("./EvidencePanel").ParameterEvidence>;
}

interface ProposalShape {
  modality?: string;
  paradigm?: string;
  registry_version?: string;
  steps: ProposalStep[];
  rationale?: string[];
}

export function isProposalJson(text: string): ProposalShape | null {
  try {
    const obj = JSON.parse(text);
    if (
      obj &&
      Array.isArray(obj.steps) &&
      obj.steps.length > 0 &&
      obj.steps.every(
        (s: { operator?: unknown; param_evidence?: unknown }) =>
          typeof s.operator === "string" &&
          typeof s.param_evidence === "object" &&
          s.param_evidence !== null,
      )
    ) {
      return obj as ProposalShape;
    }
  } catch {
    /* not JSON */
  }
  return null;
}

interface Props {
  proposal: ProposalShape;
}

export function ProposalCard({ proposal }: Props) {
  return (
    <div className="my-2 rounded-lg border border-[var(--border-primary,#e8e5e0)] overflow-hidden">
      <div className="px-3 py-1.5 bg-[var(--bg-tertiary,#f1f1ef)] border-b border-[var(--border-primary,#e8e5e0)] flex items-center gap-2">
        <span className="text-[11px] font-medium text-[var(--text-muted,#9b9a97)]">
          Proposal
        </span>
        <span className="text-[10px] text-[var(--text-faint,#c4c3c0)]">
          {proposal.steps.length} steps
        </span>
        {proposal.modality && (
          <span className="text-[10px] text-[var(--text-faint,#c4c3c0)]">
            · {proposal.modality}
          </span>
        )}
        {proposal.paradigm && (
          <span className="text-[10px] text-[var(--text-faint,#c4c3c0)]">
            · {proposal.paradigm}
          </span>
        )}
      </div>
      <div className="p-3 space-y-2">
        {proposal.steps.map((step, i) => (
          <div key={i} className="text-[12px]">
            <div className="font-mono text-[var(--text-secondary,#5f5e5b)]">
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
