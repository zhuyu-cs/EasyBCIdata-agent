import type { ConnectionState } from "@/hooks/useConnectionStatus";
import { t } from "@/lib/i18n";

interface Props {
  state: ConnectionState;
  onRetry: () => void;
}

export function ConnectionBanner({ state, onRetry }: Props) {
  if (state === "both_ok" || state === "checking") return null;

  const messages: Record<string, { text: string; hint: string }> = {
    both_down: {
      text: t("Backend not connected"),
      hint: t("Start with: easybci dashboard"),
    },
    dashboard_only: {
      text: t("Gateway not running"),
      hint: t("Start with: easybci dashboard"),
    },
    gateway_only: {
      text: t("Dashboard not running"),
      hint: t("Start with: easybci dashboard"),
    },
  };

  const msg = messages[state];
  if (!msg) return null;

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-[var(--bg-warning-subtle)] border-b border-[var(--border-warning)] text-[12px]">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 bg-[var(--text-warning)] rounded-full" />
        <span className="text-[var(--text-warning)] font-medium">{msg.text}</span>
        <span className="text-[var(--text-warning)] opacity-80">&mdash; {msg.hint}</span>
      </div>
      <button
        onClick={onRetry}
        className="text-[var(--text-warning)] underline hover:no-underline ml-2"
      >
        {t("Retry")}
      </button>
    </div>
  );
}

/**
 * GatewayRestartBanner — shown after the gateway comes back from a restart.
 * Any run that was streaming before the restart is gone server-side, so the
 * banner tells the user plainly to re-send rather than wait (B15).
 */
export function GatewayRestartBanner({ show, onDismiss }: { show: boolean; onDismiss: () => void }) {
  if (!show) return null;
  return (
    <div
      className="flex items-center justify-between px-4 py-2 border-b text-[12px] animate-fade-in"
      style={{
        background: "var(--bg-warning-subtle)",
        borderColor: "var(--border-warning)",
        color: "var(--text-warning)",
      }}
      role="alert"
    >
      <div className="flex items-center gap-2 min-w-0">
        <svg width="13" height="13" viewBox="0 0 14 14" fill="none" className="shrink-0">
          <path d="M12 7a5 5 0 11-1.46-3.54M12 2v3h-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-medium shrink-0">{t("Gateway restarted")}</span>
        <span className="opacity-80 truncate">{t("— in-flight runs were lost, please re-send your message")}</span>
      </div>
      <button
        onClick={onDismiss}
        className="underline hover:no-underline ml-2 shrink-0"
        style={{ color: "var(--text-warning)" }}
      >
        {t("Dismiss")}
      </button>
    </div>
  );
}
