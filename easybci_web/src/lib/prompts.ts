// Shared BCI task prompt templates.
//
// Single source of truth for the canned prompts surfaced in the empty-state
// QuickStartChips (A4), the ChatInput template button (B5), and the Command
// Palette actions (B3). Keeping them here avoids the prompt text drifting
// between three places.

export interface BciPromptTemplate {
  /** Short label shown on chips / menu rows. */
  label: string;
  /** The full prompt inserted into the input or sent directly. */
  prompt: string;
  /** Stable id for keys + relevance ranking. */
  id: string;
}

export const BCI_PROMPT_TEMPLATES: BciPromptTemplate[] = [
  {
    id: "preprocess-eeg",
    label: "Preprocess my EEG file",
    prompt:
      "Preprocess my EEG file — load it, inspect quality, and run a standard cleaning pipeline.",
  },
  {
    id: "ica-artifact",
    label: "Run ICA artifact removal",
    prompt:
      "Run ICA-based artifact removal on my recording to remove eye blinks and muscle noise.",
  },
  {
    id: "inspect-quality",
    label: "Inspect data quality",
    prompt:
      "Inspect my neural recording and report signal quality metrics (SNR, artifacts, bad channels).",
  },
  {
    id: "compare-pipelines",
    label: "Compare pipelines",
    prompt:
      "Compare two preprocessing pipelines on my data and tell me which produces better signal quality.",
  },
  {
    id: "epoch-erp",
    label: "Epoch & ERP analysis",
    prompt:
      "Epoch my data around stimulus events and compute the event-related potentials with baseline correction.",
  },
  {
    id: "spike-sorting",
    label: "Spike sorting",
    prompt:
      "Run spike sorting on my extracellular recording and summarize the detected units.",
  },
];
