import { Component, type ReactNode } from "react";
import { t } from "@/lib/i18n";

interface Props {
  children: ReactNode;
  panelName?: string;
  /** Override how long to wait before auto-retrying transient errors (ms). */
  autoRetryDelayMs?: number;
}

interface State {
  hasError: boolean;
  error: Error | null;
  copied: boolean;
  retryCountdown: number | null;
}

const TRANSIENT_PATTERNS = [
  /network/i,
  /timeout/i,
  /load failed/i,
  /loading chunk \d+ failed/i,
  /failed to fetch/i,
  /the operation was aborted/i,
];

function isTransient(error: Error | null): boolean {
  if (!error) return false;
  const msg = `${error.name} ${error.message}`;
  return TRANSIENT_PATTERNS.some((re) => re.test(msg));
}

function summarize(error: Error | null): string {
  if (!error) return t("An unexpected error occurred");
  const raw = error.message || String(error);
  // Trim noisy prefixes & truncate to first line + ~200 chars.
  const firstLine = raw.split("\n")[0].trim();
  return firstLine.length > 200 ? `${firstLine.slice(0, 197)}…` : firstLine;
}

function fullErrorText(error: Error | null): string {
  if (!error) return "(no error)";
  const lines = [`${error.name}: ${error.message}`];
  if (error.stack) lines.push("", error.stack);
  return lines.join("\n");
}

// After this many automatic retries a recurring error stays on screen, so a
// panel that keeps failing doesn't flicker forever.
const MAX_AUTO_RETRIES = 3;

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, copied: false, retryCountdown: null };
  private autoRetries = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private countdownTimer: ReturnType<typeof setInterval> | null = null;

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error) {
    if (isTransient(error) && this.autoRetries < MAX_AUTO_RETRIES) {
      this.autoRetries += 1;
      const delaySec = Math.ceil((this.props.autoRetryDelayMs ?? 3000) / 1000);
      this.setState({ retryCountdown: delaySec });
      this.countdownTimer = setInterval(() => {
        this.setState((s) =>
          s.retryCountdown && s.retryCountdown > 1
            ? { retryCountdown: s.retryCountdown - 1 }
            : { retryCountdown: 0 },
        );
      }, 1000);
      this.retryTimer = setTimeout(
        () => this.retry(),
        this.props.autoRetryDelayMs ?? 3000,
      );
    }
  }

  componentWillUnmount() {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    if (this.countdownTimer) clearInterval(this.countdownTimer);
  }

  private clearTimers() {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
      this.countdownTimer = null;
    }
  }

  /** The Retry button: the user asked, so automatic retries start over. */
  handleRetry = () => {
    this.autoRetries = 0;
    this.retry();
  };

  private retry() {
    this.clearTimers();
    this.setState({ hasError: false, error: null, copied: false, retryCountdown: null });
  }

  handleCopy = () => {
    const text = fullErrorText(this.state.error);
    navigator.clipboard.writeText(text).then(
      () => {
        this.setState({ copied: true });
        setTimeout(() => this.setState({ copied: false }), 2000);
      },
      () => {
        /* clipboard denied — silently fail */
      },
    );
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    const summary = summarize(this.state.error);
    const transient = isTransient(this.state.error);
    const countdown = this.state.retryCountdown;
    const copyLabel = this.state.copied ? t("Copied") : t("Copy error");

    if (this.props.panelName) {
      return (
        <div className="flex flex-col items-center justify-center h-full p-6 text-center">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center mb-3"
            style={{ background: "var(--bg-error-subtle)" }}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 6v4M10 14h.01" stroke="var(--text-error)" strokeWidth="2" strokeLinecap="round" />
              <circle cx="10" cy="10" r="8" stroke="var(--text-error)" strokeWidth="1.5" fill="none" />
            </svg>
          </div>
          <p className="text-[13px] font-medium mb-1" style={{ color: "var(--text-primary)" }}>
            {this.props.panelName} crashed
          </p>
          <p
            className="text-[11px] mb-3 max-w-[260px] break-words"
            style={{ color: "var(--text-muted)" }}
            title={summary}
          >
            {summary}
          </p>
          {transient && countdown !== null && countdown > 0 && (
            <p className="text-[11px] mb-2" style={{ color: "var(--text-faint)" }}>
              {t("Retrying in {n}s…", { n: countdown })}
            </p>
          )}
          <div className="flex items-center gap-2">
            <button
              onClick={this.handleRetry}
              className="px-3 py-1.5 text-[12px] font-medium rounded-md border transition-colors"
              style={{
                color: "var(--text-primary)",
                background: "var(--bg-primary)",
                borderColor: "var(--border-primary)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "var(--bg-primary)")}
            >
              {t("Retry")}
            </button>
            <button
              onClick={this.handleCopy}
              className="px-3 py-1.5 text-[12px] rounded-md border transition-colors"
              style={{
                color: "var(--text-secondary)",
                background: "transparent",
                borderColor: "var(--border-secondary)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              {copyLabel}
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-center h-screen" style={{ background: "var(--bg-primary)" }}>
        <div className="max-w-md p-6 text-center">
          <h1 className="text-lg font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
            {t("Something went wrong")}
          </h1>
          <p className="text-[13px] mb-4 break-words" style={{ color: "var(--text-muted)" }}>
            {summary}
          </p>
          {transient && countdown !== null && countdown > 0 && (
            <p className="text-[11px] mb-3" style={{ color: "var(--text-faint)" }}>
              {t("Auto-retrying in {n}s…", { n: countdown })}
            </p>
          )}
          <div className="flex items-center justify-center gap-2">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 text-[13px] rounded-md transition-colors"
              style={{ background: "var(--composer-btn-bg)", color: "var(--composer-btn-fg)" }}
            >
              {t("Reload page")}
            </button>
            <button
              onClick={this.handleRetry}
              className="px-3 py-2 text-[13px] rounded-md border"
              style={{ borderColor: "var(--border-primary)", color: "var(--text-primary)" }}
            >
              {t("Retry")}
            </button>
            <button
              onClick={this.handleCopy}
              className="px-3 py-2 text-[13px] rounded-md border"
              style={{ borderColor: "var(--border-secondary)", color: "var(--text-secondary)" }}
            >
              {copyLabel}
            </button>
          </div>
        </div>
      </div>
    );
  }
}
