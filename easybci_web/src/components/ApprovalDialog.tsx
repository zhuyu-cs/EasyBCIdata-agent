import { useState, useEffect, useCallback } from "react";
import { stopRun } from "@/lib/runsClient";
import { useDialog } from "@/hooks/useDialog";
import { t } from "@/lib/i18n";

interface ApprovalRequest {
  runId: string;
  command?: string;
  description?: string;
  choices: string[];
}

interface Props {
  request: ApprovalRequest;
  onResolved: () => void;
  /** Close the dialog without answering; the request stays pending. */
  onHide: () => void;
}

const CHOICE_LABELS: Record<string, { label: string; desc: string; color: string; key?: string }> = {
  once: { label: "Allow Once", desc: "Approve this specific command", color: "bg-[var(--btn-allow)] hover:bg-[var(--btn-allow-hover)]", key: "Y" },
  session: { label: "Allow Session", desc: "Approve this pattern for this session", color: "bg-[var(--btn-session)] hover:bg-[var(--btn-session-hover)]" },
  always: { label: "Always Allow", desc: "Remember and never ask again", color: "bg-[var(--btn-always)] hover:bg-[var(--btn-always-hover)]" },
  deny: { label: "Deny", desc: "Reject this command", color: "bg-[var(--btn-deny)] hover:bg-[var(--btn-deny-hover)]", key: "N" },
};

async function submitApproval(runId: string, choice: string): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const key = import.meta.env?.VITE_API_SERVER_KEY as string | undefined;
  if (key) headers["Authorization"] = `Bearer ${key}`;

  const res = await fetch(`/v1/runs/${runId}/approval`, {
    method: "POST",
    headers,
    body: JSON.stringify({ choice }),
  });
  // A non-2xx means the backend did NOT record the choice — the run is still
  // blocked server-side. Throw so the caller's catch path (stopRun) runs
  // instead of silently dismissing the dialog over a hung run (P1-3).
  if (!res.ok) {
    throw new Error(`Approval rejected by server: ${res.status}`);
  }
}

export function ApprovalDialog({ request, onResolved, onHide }: Props) {
  const [submitting, setSubmitting] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const [expandedCommand, setExpandedCommand] = useState(false);
  // "always" is sticky and easy to regret — require an explicit second click.
  const [confirmAlways, setConfirmAlways] = useState(false);

  const handleChoice = useCallback(async (choice: string) => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await submitApproval(request.runId, choice);
    } catch {
      await stopRun(request.runId).catch(() => {});
    }
    onResolved();
  }, [submitting, request.runId, onResolved]);

  // Escape only hides the dialog — answering "deny" must be a deliberate
  // choice (N or the Deny button), never a side effect of closing overlays.
  const dialogRef = useDialog<HTMLDivElement>(onHide);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (submitting) return;
      // Don't hijack keystrokes the user is typing into an input/textarea
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.key === "y" || e.key === "Y" || e.key === "Enter") {
        e.preventDefault();
        handleChoice("once");
      } else if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        handleChoice("deny");
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [submitting, handleChoice]);

  // Which granular choices the backend actually offers, beyond once/deny.
  const offered = request.choices.length > 0 ? request.choices : ["once", "session", "always", "deny"];
  const hasSession = offered.includes("session");
  const hasAlways = offered.includes("always");
  const hasMore = hasSession || hasAlways;

  // Threshold for collapsing long commands — keeps the actions visible on
  // small screens. ~10 lines / 600 chars covers most multi-line tool args
  // (terminal pipes, write_file diffs) without truncating short commands.
  const COLLAPSE_LINE_THRESHOLD = 10;
  const COLLAPSE_CHAR_THRESHOLD = 600;
  const commandText = request.command || "";
  const _lineCount = commandText.split("\n").length;
  const _shouldCollapse = _lineCount > COLLAPSE_LINE_THRESHOLD || commandText.length > COLLAPSE_CHAR_THRESHOLD;
  const displayCommand = _shouldCollapse && !expandedCommand
    ? commandText.split("\n").slice(0, COLLAPSE_LINE_THRESHOLD).join("\n").slice(0, COLLAPSE_CHAR_THRESHOLD) + "\n…"
    : commandText;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px] animate-backdrop-in" />
      <div ref={dialogRef} tabIndex={-1} role="alertdialog" aria-modal="true" aria-label={t("Permission required")} className="relative w-[90vw] max-w-md max-h-[85vh] bg-[var(--bg-secondary)] rounded-t-xl sm:rounded-xl shadow-xl overflow-hidden animate-approval-slide-up flex flex-col">
        {/* Header (fixed) */}
        <div className="px-5 pt-5 pb-3 shrink-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 flex items-center justify-center rounded-full bg-[var(--bg-warning-subtle)] text-[var(--text-warning)] text-[12px] font-bold">!</span>
            <h3 className="text-[14px] font-semibold text-[var(--text-primary)]">{t("Permission Required")}</h3>
            <button
              type="button"
              onClick={onHide}
              className="ml-auto text-[11px] px-1.5 py-0.5 rounded hover:bg-[var(--bg-hover)]"
              style={{ color: "var(--text-muted)" }}
              title={t("Decide later (Esc) — the run waits")}
            >
              {t("Later")}
              <kbd className="ml-1 text-[11px] opacity-70 font-mono">Esc</kbd>
            </button>
          </div>
          {request.description && (
            <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">{request.description}</p>
          )}
        </div>

        {/* Command block (scrollable, collapsible) — flex-1 so it shrinks
            before the action row, keeping Allow / Deny always visible. */}
        {request.command && (
          <div className="mx-5 mb-3 px-3 py-2.5 rounded-md bg-[var(--bg-code)] border border-[var(--border-code)] flex-1 min-h-0 overflow-y-auto">
            <pre className="text-[12px] font-mono text-[var(--text-primary)] whitespace-pre-wrap break-all leading-relaxed">
              {displayCommand}
            </pre>
            {_shouldCollapse && (
              <button
                type="button"
                onClick={() => setExpandedCommand((v) => !v)}
                className="mt-2 text-[11px] font-medium underline hover:no-underline"
                style={{ color: "var(--text-muted)" }}
              >
                {expandedCommand
                  ? t("Collapse ({n} lines)", { n: _lineCount })
                  : t("Show all {n} lines / {chars} chars", { n: _lineCount, chars: commandText.length.toLocaleString() })}
              </button>
            )}
          </div>
        )}

        {/* Primary actions: Allow / Deny only by default (always visible) */}
        <div className="px-5 pb-3 pt-1 flex flex-wrap gap-2 shrink-0 border-t" style={{ borderColor: "var(--border-primary)" }}>
          <button
            onClick={() => handleChoice("once")}
            disabled={submitting}
            autoFocus
            className={`px-3 py-1.5 rounded-md text-[12px] font-medium text-white transition-colors disabled:opacity-50 ${CHOICE_LABELS.once.color}`}
            title={`${t(CHOICE_LABELS.once.desc)} (Enter)`}
          >
            {t("Allow")}
            <kbd className="ml-1.5 text-[11px] opacity-70 font-mono">⏎ / Y</kbd>
          </button>
          <button
            onClick={() => handleChoice("deny")}
            disabled={submitting}
            className={`px-3 py-1.5 rounded-md text-[12px] font-medium text-white transition-colors disabled:opacity-50 ${CHOICE_LABELS.deny.color}`}
            title={`${t(CHOICE_LABELS.deny.desc)} (N)`}
          >
            {t("Deny")}
            <kbd className="ml-1.5 text-[11px] opacity-70 font-mono">N</kbd>
          </button>
          {hasMore && (
            <button
              onClick={() => setShowMore((v) => !v)}
              disabled={submitting}
              className="px-3 py-1.5 rounded-md text-[12px] font-medium transition-colors disabled:opacity-50 ml-auto"
              style={{ color: "var(--text-secondary)", background: "var(--bg-tertiary)" }}
              title={t("More approval scopes")}
            >
              {t("More")}
              <svg width="10" height="10" viewBox="0 0 11 11" fill="none" className="inline-block ml-1" style={{ transform: showMore ? "rotate(180deg)" : "none" }}>
                <path d="M3 4.5L5.5 7L8 4.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>

        {/* Granular scopes — hidden until "More" */}
        {showMore && hasMore && (
          <div className="px-5 pb-5 flex flex-col gap-2 animate-fade-in border-t pt-3 shrink-0" style={{ borderColor: "var(--border-primary)" }}>
            {hasSession && (
              <button
                onClick={() => handleChoice("session")}
                disabled={submitting}
                className="text-left px-3 py-2 rounded-md text-[12px] font-medium transition-colors disabled:opacity-50 border"
                style={{ color: "var(--text-primary)", borderColor: "var(--border-secondary)" }}
              >
                <span className="block">{t(CHOICE_LABELS.session.label)}</span>
                <span className="block text-[11px] font-normal" style={{ color: "var(--text-muted)" }}>{t(CHOICE_LABELS.session.desc)}</span>
              </button>
            )}
            {hasAlways && (
              confirmAlways ? (
                <button
                  onClick={() => handleChoice("always")}
                  disabled={submitting}
                  className={`text-left px-3 py-2 rounded-md text-[12px] font-medium text-white transition-colors disabled:opacity-50 ${CHOICE_LABELS.always.color}`}
                >
                  <span className="block">{t("Confirm: Always Allow")}</span>
                  <span className="block text-[11px] font-normal opacity-80">{t("Click again to permanently remember this")}</span>
                </button>
              ) : (
                <button
                  onClick={() => setConfirmAlways(true)}
                  disabled={submitting}
                  className="text-left px-3 py-2 rounded-md text-[12px] font-medium transition-colors disabled:opacity-50 border"
                  style={{ color: "var(--text-primary)", borderColor: "var(--border-secondary)" }}
                >
                  <span className="block">{t(CHOICE_LABELS.always.label)}</span>
                  <span className="block text-[11px] font-normal" style={{ color: "var(--text-muted)" }}>{t(CHOICE_LABELS.always.desc)} — requires confirm</span>
                </button>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
