// Detection helper for proposal-JSON code blocks. Kept separate from
// ProposalCard.tsx so the card file exports only components (fast refresh) and
// Markdown can import the cheap test on its own.

interface ProposalStep {
  operator: string;
  params?: Record<string, unknown>;
  param_evidence?: Record<string, import("./EvidencePanel").ParameterEvidence>;
}

export interface ProposalShape {
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
