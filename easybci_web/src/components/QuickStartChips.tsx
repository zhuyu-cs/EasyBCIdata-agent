import type { ReactNode } from "react";
import { BCI_PROMPT_TEMPLATES } from "@/lib/prompts";

interface Props {
  onSelect: (text: string) => void;
}

// Per-template icons keyed by the shared template id. Only the four primary
// templates surface as empty-state chips; the rest live in the slash/template
// menu and command palette.
const CHIP_ICONS: Record<string, ReactNode> = {
  "preprocess-eeg": (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path
        d="M1 7h2.5l1.5-4 2 8 1.5-5 1 3 1-2H13"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
  "ica-artifact": (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M1.5 3.5h11M3.5 7h7M5.5 10.5h3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  ),
  "inspect-quality": (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M1.5 9a5.5 5.5 0 0111 0" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M7 9l2.5-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="7" cy="9" r="0.9" fill="currentColor" />
    </svg>
  ),
  "compare-pipelines": (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M7 1.5v11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M2 4.5l-1 2 2 .5M2 4.5l1 2-2 .5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 4.5l-1 2 2 .5M12 4.5l1 2-2 .5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
};

const CHIPS = BCI_PROMPT_TEMPLATES.filter((t) => t.id in CHIP_ICONS);

export function QuickStartChips({ onSelect }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2 mt-6 w-full max-w-[420px]">
      {CHIPS.map((chip) => (
        <button
          key={chip.id}
          onClick={() => onSelect(chip.prompt)}
          className="group flex items-center gap-2 px-3 py-2.5 rounded-lg border text-left transition-all duration-150 bg-[var(--bg-secondary)] border-[var(--border-primary)] hover:bg-[var(--bg-hover)] hover:-translate-y-[1px]"
          style={{ boxShadow: "var(--shadow-sm)" }}
        >
          <span className="shrink-0 text-[var(--accent-green)]">{CHIP_ICONS[chip.id]}</span>
          <span className="text-caption" style={{ color: "var(--text-secondary)" }}>
            {chip.label}
          </span>
        </button>
      ))}
    </div>
  );
}
