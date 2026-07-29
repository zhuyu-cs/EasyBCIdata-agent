// Detection + parsing helpers for QC-metrics JSON blocks. Kept separate from
// QCCard.tsx so the card can stay a lazy-loaded, component-only module (and so
// MessageBubble can import the cheap parser without pulling in the card).

export interface QCMetrics {
  snr?: number;
  artifact_ratio?: number;
  quality_score?: number;
  [key: string]: unknown;
}

export function tryParseQCMetrics(jsonStr: string): QCMetrics | null {
  try {
    const parsed = JSON.parse(jsonStr);
    if (typeof parsed !== "object" || parsed === null) return null;
    if ("snr" in parsed || "artifact_ratio" in parsed || "quality_score" in parsed) {
      return parsed as QCMetrics;
    }
  } catch { /* not JSON */ }
  return null;
}
