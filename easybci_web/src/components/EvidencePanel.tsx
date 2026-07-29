import { useState } from "react";

export interface ParameterEvidence {
  operator: string;
  parameter: string;
  value: unknown;
  source: "web" | "empirical_default" | "registry_miss" | "user_provided" | string;
  confidence: number;
  citations?: Array<{ url: string; title?: string; snippet?: string }>;
  summary?: string;
  default_origin?: string;
  fallback_reason?: string;
  previous_evidence?: ParameterEvidence | null;
}

interface Props {
  stepIndex: number;
  operator: string;
  evidenceByParam: Record<string, ParameterEvidence>;
}

const SOURCE_BG: Record<string, string> = {
  web: "bg-emerald-50 border-emerald-300 text-emerald-900",
  empirical_default: "bg-zinc-50 border-zinc-300 text-zinc-700",
  registry_miss: "bg-rose-50 border-rose-300 text-rose-900",
  user_provided: "bg-sky-50 border-sky-300 text-sky-900",
};

export function EvidencePanel({ stepIndex, operator, evidenceByParam }: Props) {
  const [open, setOpen] = useState(false);
  const entries = Object.entries(evidenceByParam ?? {});
  if (entries.length === 0) return null;
  return (
    <div className="mt-2 border border-[var(--border-primary,#e8e5e0)] rounded">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full text-left px-3 py-1 text-[11px] text-[var(--text-secondary,#5f5e5b)] hover:bg-[var(--bg-hover,#ebebea)]"
      >
        {open ? "▼" : "▶"} Evidence — Step {stepIndex} {operator} ({entries.length})
      </button>
      {open && (
        <ul className="px-3 py-2 space-y-1 text-[12px]">
          {entries.map(([param, ev]) => (
            <li
              key={param}
              className={`px-2 py-1 rounded border ${SOURCE_BG[ev.source] ?? "bg-zinc-50 border-zinc-300"}`}
            >
              <div className="flex flex-wrap items-baseline gap-2">
                <strong>{param}</strong>
                <span className="text-[11px]">= {String(ev.value)}</span>
                <span className="text-[10px] opacity-70">
                  {ev.source === "web"
                    ? `web · ${(ev.confidence ?? 0).toFixed(2)}`
                    : ev.source === "empirical_default"
                    ? ev.fallback_reason
                      ? `default · ${ev.fallback_reason}`
                      : "default"
                    : ev.source}
                </span>
              </div>
              {ev.summary && (
                <div className="text-[10px] mt-1 opacity-80">{ev.summary}</div>
              )}
              {ev.citations && ev.citations.length > 0 && (
                <div className="text-[10px] mt-1">
                  Sources:{" "}
                  {ev.citations.map((c, i) => (
                    <a
                      key={i}
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline mr-2"
                    >
                      {c.title || c.url}
                    </a>
                  ))}
                </div>
              )}
              {ev.default_origin && (
                <div className="text-[10px] opacity-70">{ev.default_origin}</div>
              )}
              {ev.previous_evidence && (
                <div className="text-[10px] opacity-70 mt-1">
                  override (was {String(ev.previous_evidence.value)})
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
