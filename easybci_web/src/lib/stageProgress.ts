/** Live stage progress emitted by the gateway as SSE `progress` events. */
export type StageProgress = {
  stage: "plan" | "codegen" | "preprocess" | "qc" | "train" | "evaluate";
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
