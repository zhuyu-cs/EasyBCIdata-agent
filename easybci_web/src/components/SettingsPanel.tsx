import { memo, useEffect, useState } from "react";
import {
  api,
  type ModelInfoResponse,
  type ModelOptionsResponse,
  type ToolsetInfo,
  type StatusResponse,
  type EnvVarInfo,
  type WebSearchStatus,
} from "@/lib/api";
import { useThemeStore } from "@/stores/themeStore";
import { useToast } from "@/stores/toastStore";
import { useDialog } from "@/hooks/useDialog";
import { setLang, t, useLangStore, type Lang } from "@/lib/i18n";

interface Props {
  open: boolean;
  onClose: () => void;
}

type Theme = "light" | "dark" | "system";

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-label uppercase tracking-wide mb-2" style={{ color: "var(--text-faint)" }}>
      {children}
    </h3>
  );
}

function ThemeSegment() {
  const { theme, setTheme } = useThemeStore();
  const options: { value: Theme; label: string }[] = [
    { value: "light", label: t("Light") },
    { value: "dark", label: t("Dark") },
    { value: "system", label: t("System") },
  ];
  return (
    <div className="flex rounded-md p-0.5 gap-0.5" style={{ background: "var(--bg-tertiary)" }}>
      {options.map((o) => {
        const active = theme === o.value;
        return (
          <button
            key={o.value}
            onClick={() => setTheme(o.value)}
            className="flex-1 text-[12px] py-1 rounded transition-colors"
            style={{
              background: active ? "var(--bg-secondary)" : "transparent",
              color: active ? "var(--text-primary)" : "var(--text-muted)",
              boxShadow: active ? "var(--shadow-sm)" : undefined,
              fontWeight: active ? 600 : 400,
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// ── Web Search providers ──────────────────────────────────────────────────
// Maps a UI label → the `web.backend` config string (must match the registry
// names exactly, note the hyphen in "brave-free") → the env var holding its key.
interface WebProvider {
  backend: string;
  label: string;
  envKey: string | null;
  isPassword: boolean;
  note?: string;
}

const WEB_PROVIDERS: WebProvider[] = [
  { backend: "", label: "Auto (any available)", envKey: null, isPassword: false },
  { backend: "tavily", label: "Tavily", envKey: "TAVILY_API_KEY", isPassword: true },
  { backend: "exa", label: "Exa", envKey: "EXA_API_KEY", isPassword: true },
];

interface WebSearchSectionProps {
  config: Record<string, unknown> | null;
  envVars: Record<string, EnvVarInfo>;
  webConfigured: boolean;
  onConfigChange: (web: { backend: string; search_backend: string }) => void;
  onEnvChange: (key: string, info: Partial<EnvVarInfo>) => void;
}

interface PermissionsSectionProps {
  config: Record<string, unknown> | null;
  onConfigChange: (approvals: { mode: string }) => void;
}

function PermissionsSection({ config, onConfigChange }: PermissionsSectionProps) {
  const { toast } = useToast();
  const current = ((config?.approvals as { mode?: string } | undefined)?.mode ?? "off").toLowerCase();
  const [mode, setMode] = useState<string>(current);
  const [busy, setBusy] = useState(false);

  // Follow the stored value when config (re)loads — adjusted during render.
  const [prevCurrent, setPrevCurrent] = useState(current);
  if (current !== prevCurrent) {
    setPrevCurrent(current);
    setMode(current);
  }

  const handleChange = async (value: string) => {
    const prev = mode;
    setMode(value);
    setBusy(true);
    try {
      await api.saveConfig({ approvals: { mode: value } });
      onConfigChange({ mode: value });
      toast(t("Approval mode set to {v}", { v: value }), "success");
    } catch (e) {
      setMode(prev);
      toast(e instanceof Error ? e.message : t("Failed to save approval mode"), "error");
    } finally {
      setBusy(false);
    }
  };

  const isYolo = mode === "off";

  return (
    <section>
      <SectionTitle>{t("Permissions")}</SectionTitle>
      <div
        className="rounded-md border p-3 space-y-2.5"
        style={{ borderColor: "var(--border-primary)", background: "var(--bg-tertiary)" }}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[12px] font-medium" style={{ color: "var(--text-primary)" }}>
              {t("Command approval mode")}
            </div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-secondary)" }}>
              {t("Off — auto-run every command (catastrophic deletes still blocked). Manual — prompt before each potentially-dangerous command. Smart — auxiliary LLM auto-approves low-risk ones.")}
            </div>
          </div>
          <select
            value={mode}
            disabled={busy}
            onChange={(e) => handleChange(e.target.value)}
            className="text-[12px] rounded px-2 py-1 shrink-0"
            style={{
              background: "var(--bg-primary)",
              color: "var(--text-primary)",
              border: "1px solid var(--border-primary)",
              minWidth: 90,
            }}
          >
            <option value="off">off</option>
            <option value="manual">manual</option>
            <option value="smart">smart</option>
          </select>
        </div>
        <div
          className="flex items-center gap-1.5 text-[11px]"
          style={{ color: isYolo ? "var(--accent-green)" : "var(--text-muted)" }}
        >
          <span className="w-2 h-2 rounded-full" style={{ background: isYolo ? "var(--accent-green)" : "var(--text-faint)" }} />
          {isYolo
            ? t("Run-anything mode — only rm -rf /, mkfs, dd to /dev/sd*, shutdown/reboot, fork bomb stay blocked.")
            : t("You will be asked before any flagged command runs.")}
        </div>
      </div>
    </section>
  );
}

function WebSearchSection({
  config,
  envVars,
  webConfigured,
  onConfigChange,
  onEnvChange,
}: WebSearchSectionProps) {
  const { toast } = useToast();
  const initialBackend =
    (config?.web as { backend?: string } | undefined)?.backend ?? "";
  const [backend, setBackend] = useState(initialBackend);
  const [keyInput, setKeyInput] = useState("");
  const [revealed, setRevealed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [wsStatus, setWsStatus] = useState<WebSearchStatus | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  const provider =
    WEB_PROVIDERS.find((p) => p.backend === backend) ?? WEB_PROVIDERS[0];
  const envInfo = provider.envKey ? envVars[provider.envKey] : undefined;
  const keyIsSet = !!envInfo?.is_set;

  // Pull provider status once on mount + after every backend / key change.
  // Used to render the activation badge ("✓ exa" / "✗ none usable") and
  // the (why?) diagnostic panel that explains each unavailable provider.
  const refreshStatus = async () => {
    try {
      const s = await api.getWebSearchStatus();
      setWsStatus(s);
    } catch {
      // Best-effort — older Gateway / proxy paths may 404 the new route.
      setWsStatus(null);
    }
  };
  useEffect(() => {
    let alive = true;
    api.getWebSearchStatus()
      .then((s) => { if (alive) setWsStatus(s); })
      .catch(() => { if (alive) setWsStatus(null); });
    return () => { alive = false; };
  }, []);

  const handleBackendChange = async (value: string) => {
    setBackend(value);
    setKeyInput("");
    setRevealed(null);
    setBusy(true);
    try {
      await api.saveConfig({ web: { backend: value, search_backend: value } });
      onConfigChange({ backend: value, search_backend: value });

      const label = t(WEB_PROVIDERS.find((p) => p.backend === value)?.label ?? t("Auto"));

      if ((value === "tavily" || value === "exa") && !keyIsSet) {
        // Paid backend selected but no key on disk — explicit notice so
        // the user doesn't think the toast = ready-to-use.
        toast(
          t("{l} selected — no API key configured. Enter the key below to activate.", { l: label }),
          "info",
        );
      } else {
        toast(t("Web search backend set to {l}", { l: label }), "success");
      }
      // Always refresh status so the badge reflects the new picture.
      await refreshStatus();
    } catch (e) {
      setBackend(initialBackend);
      toast(e instanceof Error ? e.message : t("Failed to set backend"), "error");
    }
    setBusy(false);
  };

  const handleSaveKey = async () => {
    if (!provider.envKey || !keyInput.trim()) return;
    setBusy(true);
    try {
      await api.setEnvVar(provider.envKey, keyInput.trim());
      onEnvChange(provider.envKey, { is_set: true, redacted_value: "••••••" });
      setKeyInput("");
      setRevealed(null);
      toast(t("{l} key saved", { l: t(provider.label) }), "success");
      await refreshStatus();
    } catch (e) {
      toast(e instanceof Error ? e.message : t("Failed to save key"), "error");
    }
    setBusy(false);
  };

  const handleClearKey = async () => {
    if (!provider.envKey) return;
    setBusy(true);
    try {
      await api.deleteEnvVar(provider.envKey);
      onEnvChange(provider.envKey, { is_set: false, redacted_value: null });
      setKeyInput("");
      setRevealed(null);
      toast(t("{l} key cleared", { l: t(provider.label) }), "info");
      await refreshStatus();
    } catch (e) {
      toast(e instanceof Error ? e.message : t("Failed to clear key"), "error");
    }
    setBusy(false);
  };

  const handleReveal = async () => {
    if (!provider.envKey) return;
    if (revealed !== null) {
      setRevealed(null);
      return;
    }
    try {
      const res = await api.revealEnvVar(provider.envKey);
      setRevealed(res.value);
    } catch (e) {
      toast(e instanceof Error ? e.message : t("Failed to reveal key"), "error");
    }
  };

  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <SectionTitle>{t("Web Search")}</SectionTitle>
        <span
          className="text-[11px] px-1.5 py-0.5 rounded-full"
          style={
            webConfigured
              ? { background: "var(--bg-success-subtle)", color: "var(--text-success)" }
              : { background: "var(--bg-tertiary)", color: "var(--text-muted)" }
          }
        >
          {webConfigured ? t("configured") : t("no key")}
        </span>
      </div>

      {/* Activation badge — strict_available reflects research_preprocessing's
          actual must-call detection gate. */}
      {wsStatus && (
        <div className="text-[11px] flex items-center gap-2 mb-2" style={{ color: "var(--text-faint)" }}>
          <span>{t("Active:")}</span>
          {wsStatus.active_search_strict ? (
            <span style={{ color: "var(--text-success)" }}>
              ✓ {wsStatus.active_search_strict}
            </span>
          ) : (
            <span style={{ color: "var(--text-warning)" }}>
              {t("✗ none usable")}
            </span>
          )}
          {wsStatus.providers.some((p) => !p.is_available && p.diagnostic) && (
            <button
              type="button"
              onClick={() => setShowDiagnostics((v) => !v)}
              className="underline text-[11px]"
              style={{ color: "var(--text-faint)" }}
            >
              {showDiagnostics ? t("(hide)") : t("(why?)")}
            </button>
          )}
        </div>
      )}

      {/* Diagnostic detail panel — collapsed by default to keep the panel uncluttered.
          Lists each registered, search-capable provider with its specific
          activation reason and one-line fix hint. */}
      {wsStatus && showDiagnostics && (
        <div
          className="rounded-md border p-2 mb-2 text-[11px] space-y-1.5"
          style={{ background: "var(--bg-tertiary)", borderColor: "var(--border-secondary)" }}
        >
          {wsStatus.providers
            .filter((p) => p.supports_search)
            .map((p) => (
              <div key={p.name}>
                <span style={{ color: p.is_available ? "var(--text-success)" : "var(--text-warning)" }}>
                  {p.is_available ? "✓" : "✗"} {p.display_name}
                </span>
                {p.diagnostic && !p.is_available && (
                  <div className="ml-3 mt-0.5" style={{ color: "var(--text-secondary)" }}>
                    {p.diagnostic.reason}
                    {p.diagnostic.fix_hint && (
                      <div className="text-[11px] mt-0.5" style={{ color: "var(--text-faint)" }}>
                        Fix: {p.diagnostic.fix_hint}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {/* Provider picker */}
      <div className="relative">
        <select
          value={backend}
          onChange={(e) => handleBackendChange(e.target.value)}
          disabled={busy}
          className="w-full appearance-none text-[13px] rounded-md border pl-3 pr-8 py-2 outline-none transition-colors disabled:opacity-60"
          style={{ background: "var(--bg-input)", borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
        >
          {WEB_PROVIDERS.map((p) => (
            <option key={p.backend || "auto"} value={p.backend}>
              {t(p.label)}
            </option>
          ))}
        </select>
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "var(--text-muted)" }}>
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {/* API key field — only for providers that take a key */}
      {provider.envKey && (
        <div className="mt-2 space-y-1.5">
          <div className="flex gap-1.5">
            <div className="relative flex-1">
              <input
                type={revealed !== null ? "text" : "password"}
                value={revealed !== null ? revealed : keyInput}
                onChange={(e) => {
                  if (revealed !== null) setRevealed(null);
                  setKeyInput(e.target.value);
                }}
                placeholder={
                  keyIsSet
                    ? (envInfo?.redacted_value ?? t("key set — enter to replace"))
                    : provider.isPassword
                      ? t("Paste API key")
                      : t("Enter value")
                }
                disabled={busy}
                className="w-full text-[13px] rounded-md border pl-3 pr-9 py-2 outline-none transition-colors disabled:opacity-60"
                style={{ background: "var(--bg-input)", borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
              />
              {keyIsSet && (
                <button
                  type="button"
                  onClick={handleReveal}
                  title={revealed !== null ? t("Hide") : t("Reveal")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 w-5 h-5 flex items-center justify-center rounded hover:bg-[var(--bg-hover)]"
                  style={{ color: "var(--text-muted)" }}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M1 7s2.2-4 6-4 6 4 6 4-2.2 4-6 4-6-4-6-4z" stroke="currentColor" strokeWidth="1.2" />
                    <circle cx="7" cy="7" r="1.6" stroke="currentColor" strokeWidth="1.2" />
                  </svg>
                </button>
              )}
            </div>
            <button
              type="button"
              onClick={handleSaveKey}
              disabled={busy || !keyInput.trim()}
              className="shrink-0 text-[12px] px-3 rounded-md border transition-colors hover:bg-[var(--bg-tertiary)] disabled:opacity-50"
              style={{ borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
            >
              {t("Save")}
            </button>
            {keyIsSet && (
              <button
                type="button"
                onClick={handleClearKey}
                disabled={busy}
                className="shrink-0 text-[12px] px-3 rounded-md border transition-colors hover:bg-[var(--bg-tertiary)] disabled:opacity-50"
                style={{ borderColor: "var(--border-secondary)", color: "var(--text-muted)" }}
              >
                {t("Clear")}
              </button>
            )}
          </div>
          {provider.envKey && (
            <a
              href={
                provider.backend === "tavily" ? "https://app.tavily.com/home"
                : provider.backend === "exa" ? "https://exa.ai/"
                : "#"
              }
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] hover:underline"
              style={{ color: "var(--text-faint)" }}
            >
              {t("Get an API key →")}
            </a>
          )}
        </div>
      )}

      {provider.note && (
        <p className="text-[11px] mt-1.5" style={{ color: "var(--text-faint)" }}>
          {provider.note}
        </p>
      )}
      {backend === "" && (
        <p className="text-[11px] mt-1.5" style={{ color: "var(--text-faint)" }}>
          {t("Auto-selects whichever provider has a key configured. Web search is fully optional — the agent works normally without it.")}
        </p>
      )}
    </section>
  );
}

export const SettingsPanel = memo(function SettingsPanel({ open, onClose }: Props) {
  const { toast } = useToast();
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelOptions, setModelOptions] = useState<ModelOptionsResponse | null>(null);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [envVars, setEnvVars] = useState<Record<string, EnvVarInfo>>({});
  // The spinner shows only until the first load. Later opens render the data
  // already on hand and refresh it silently in the background; saves update
  // the local copy directly.
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [toolsetsExpanded, setToolsetsExpanded] = useState(false);

  const loading = open && !loaded;

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    Promise.allSettled([
      api.getModelInfo(),
      api.getModelOptions(),
      api.getToolsets(),
      api.getStatus(),
      api.getConfig(),
      api.getEnvVars(),
    ]).then((results) => {
      if (cancelled) return;
      const [mi, mo, ts, st, cfg, ev] = results;
      if (mi.status === "fulfilled") setModelInfo(mi.value);
      if (mo.status === "fulfilled") setModelOptions(mo.value);
      if (ts.status === "fulfilled") setToolsets(ts.value);
      if (st.status === "fulfilled") setStatus(st.value);
      if (cfg.status === "fulfilled") setConfig(cfg.value);
      if (ev.status === "fulfilled") setEnvVars(ev.value);
      setLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const dialogRef = useDialog<HTMLElement>(onClose, open);
  const lang = useLangStore((s) => s.lang);

  // Takes effect at once; the shared config key also sets the agent's language.
  const changeLanguage = (value: Lang) => {
    setLang(value);
    api.saveConfig({ display: { language: value } }).catch((e) => {
      toast(e instanceof Error ? e.message : t("Failed to save language"), "error");
    });
  };

  const handleModelChange = async (value: string) => {
    const sep = value.indexOf("::");
    if (sep === -1) return;
    const provider = value.slice(0, sep);
    const model = value.slice(sep + 2);
    setSaving(true);
    try {
      await api.setModelAssignment({ scope: "main", provider, model });
      setModelInfo((prev) => (prev ? { ...prev, model, provider } : prev));
      toast(t("Model set to {m}", { m: model }), "success");
    } catch (e) {
      toast(e instanceof Error ? e.message : t("Failed to set model"), "error");
    }
    setSaving(false);
  };

  const terminalBackend =
    (config?.terminal as { backend?: string } | undefined)?.backend ?? "local";
  const executionMode =
    (config?.code_execution as { mode?: string } | undefined)?.mode ?? "project";

  const currentValue = modelInfo ? `${modelInfo.provider}::${modelInfo.model}` : "";

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-[55] bg-black/30 transition-opacity duration-200 ${open ? "opacity-100" : "opacity-0 pointer-events-none"}`}
        onClick={onClose}
      />
      {/* Drawer — slides in from the LEFT */}
      <aside
        className={`fixed top-0 left-0 z-[56] h-full w-[360px] max-w-[92vw] flex flex-col transition-transform duration-250 ease-out ${open ? "translate-x-0" : "-translate-x-full"}`}
        style={{ background: "var(--bg-secondary)", boxShadow: "var(--shadow-lg)", borderRight: "1px solid var(--border-primary)" }}
        ref={dialogRef}
        tabIndex={-1}
        inert={!open}
        role="dialog"
        aria-modal={open}
        aria-label={t("Settings")}
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b shrink-0" style={{ borderColor: "var(--border-primary)" }}>
          <h2 className="text-heading-sm" style={{ color: "var(--text-primary)" }}>{t("Settings")}</h2>
          <button
            onClick={onClose}
            aria-label={t("Close")}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--text-muted)" }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
          {loading && (
            <div className="flex items-center gap-2 text-[12px]" style={{ color: "var(--text-muted)" }}>
              <span className="w-3 h-3 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin" />
              {t("Loading settings…")}
            </div>
          )}

          {!loading && (
            <>
              {/* Model & Theme */}
              <section>
                <SectionTitle>{t("Model & Theme")}</SectionTitle>
                <div className="relative">
                  <select
                    value={currentValue}
                    onChange={(e) => handleModelChange(e.target.value)}
                    disabled={saving || !modelOptions?.providers?.length}
                    className="w-full appearance-none text-[13px] rounded-md border pl-3 pr-8 py-2 outline-none transition-colors disabled:opacity-60"
                    style={{ background: "var(--bg-input)", borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
                  >
                    {!modelOptions?.providers?.length && modelInfo && (
                      <option value={currentValue}>{modelInfo.model}</option>
                    )}
                    {modelOptions?.providers?.map((p) => (
                      <optgroup key={p.slug} label={p.name}>
                        {(p.models ?? []).map((m) => (
                          <option key={`${p.slug}::${m}`} value={`${p.slug}::${m}`}>
                            {m}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: "var(--text-muted)" }}>
                    <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                {modelInfo && (
                  <div className="flex items-center gap-2 mt-2 flex-wrap">
                    <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}>
                      {modelInfo.provider}
                    </span>
                    {modelInfo.capabilities?.context_window && (
                      <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}>
                        {Math.round(modelInfo.capabilities.context_window / 1000)}k ctx
                      </span>
                    )}
                    {modelInfo.capabilities?.supports_vision && (
                      <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}>vision</span>
                    )}
                    {modelInfo.capabilities?.supports_reasoning && (
                      <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}>reasoning</span>
                    )}
                  </div>
                )}
                <div className="mt-3">
                  <ThemeSegment />
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Language")}</span>
                  <select
                    value={lang}
                    onChange={(e) => changeLanguage(e.target.value as Lang)}
                    className="text-[12px] rounded-md border px-2 py-1 outline-none"
                    style={{ background: "var(--bg-input)", borderColor: "var(--border-secondary)", color: "var(--text-primary)" }}
                  >
                    <option value="en">English</option>
                    <option value="zh">简体中文</option>
                  </select>
                </div>
              </section>

              {/* Runtime: execution environment + collapsible toolsets */}
              <section>
                <SectionTitle>{t("Runtime")}</SectionTitle>
                <div className="rounded-md border p-3 space-y-2" style={{ borderColor: "var(--border-primary)", background: "var(--bg-tertiary)" }}>
                  <div className="flex items-center justify-between">
                    <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Sandbox")}</span>
                    <span className="text-[12px] font-mono capitalize" style={{ color: "var(--text-primary)" }}>{terminalBackend}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Code execution")}</span>
                    <span className="text-[12px] font-mono" style={{ color: "var(--text-primary)" }}>{executionMode}</span>
                  </div>
                </div>

                {/* Toolsets — collapsed by default (avg 8-10 items is long) */}
                <button
                  onClick={() => setToolsetsExpanded((v) => !v)}
                  className="mt-2 w-full flex items-center justify-between gap-2 rounded-md px-3 py-2 border transition-colors hover:bg-[var(--bg-tertiary)]"
                  style={{ borderColor: "var(--border-primary)" }}
                >
                  <span className="flex items-center gap-2 text-[12px]" style={{ color: "var(--text-secondary)" }}>
                    <svg
                      width="11" height="11" viewBox="0 0 11 11" fill="none"
                      className="transition-transform"
                      style={{ transform: toolsetsExpanded ? "rotate(90deg)" : "none", color: "var(--text-muted)" }}
                    >
                      <path d="M4 2.5L7.5 5.5L4 8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {t("Toolsets")}
                  </span>
                  <span className="text-[11px] tabular-nums" style={{ color: "var(--text-faint)" }}>
                    {toolsets.filter((ts) => ts.enabled).length}/{toolsets.length} {t("on")}
                  </span>
                </button>
                {toolsetsExpanded && (
                  <div className="space-y-1.5 mt-1.5 animate-fade-in">
                    {toolsets.length === 0 && (
                      <p className="text-[12px]" style={{ color: "var(--text-muted)" }}>{t("No toolsets reported.")}</p>
                    )}
                    {toolsets.map((ts) => (
                      <div key={ts.name} className="flex items-center justify-between gap-2 rounded-md px-3 py-2 border" style={{ borderColor: "var(--border-primary)" }}>
                        <div className="min-w-0">
                          <div className="text-[12px] font-medium truncate" style={{ color: "var(--text-primary)" }}>{ts.label || ts.name}</div>
                          {ts.description && (
                            <div className="text-[11px] truncate" style={{ color: "var(--text-muted)" }}>{ts.description}</div>
                          )}
                        </div>
                        <span
                          className="shrink-0 text-[11px] px-1.5 py-0.5 rounded-full"
                          style={
                            ts.enabled
                              ? { background: "var(--bg-success-subtle)", color: "var(--text-success)" }
                              : { background: "var(--bg-tertiary)", color: "var(--text-muted)" }
                          }
                        >
                          {ts.enabled ? t("on") : t("off")}
                        </span>
                      </div>
                    ))}
                    <p className="text-[11px] pt-1" style={{ color: "var(--text-faint)" }}>
                      Toolset availability is configured via the CLI / config file.
                    </p>
                  </div>
                )}
              </section>

              {/* Web Search */}
              <WebSearchSection
                config={config}
                envVars={envVars}
                webConfigured={!!toolsets.find((ts) => ts.name === "web")?.configured}
                onConfigChange={(web) =>
                  setConfig((prev) => ({ ...(prev ?? {}), web: { ...((prev?.web as object) ?? {}), ...web } }))
                }
                onEnvChange={(key, info) =>
                  setEnvVars((prev) => ({
                    ...prev,
                    [key]: { ...(prev[key] ?? ({} as EnvVarInfo)), ...info },
                  }))
                }
              />

              {/* Permissions */}
              <PermissionsSection
                config={config}
                onConfigChange={(approvals) =>
                  setConfig((prev) => ({
                    ...(prev ?? {}),
                    approvals: { ...((prev?.approvals as object) ?? {}), ...approvals },
                  }))
                }
              />

              {/* About */}
              <section>
                <SectionTitle>{t("About")}</SectionTitle>
                <div className="rounded-md border p-3 space-y-2" style={{ borderColor: "var(--border-primary)", background: "var(--bg-tertiary)" }}>
                  <div className="flex items-center justify-between">
                    <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Version")}</span>
                    <span className="text-[12px] font-mono" style={{ color: "var(--text-primary)" }}>{status?.version ?? "—"}</span>
                  </div>
                  {status?.release_date && (
                    <div className="flex items-center justify-between">
                      <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Released")}</span>
                      <span className="text-[12px] font-mono" style={{ color: "var(--text-primary)" }}>{status.release_date}</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Active sessions")}</span>
                    <span className="text-[12px] font-mono" style={{ color: "var(--text-primary)" }}>{status?.active_sessions ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[12px]" style={{ color: "var(--text-secondary)" }}>{t("Gateway")}</span>
                    <span className="flex items-center gap-1.5 text-[12px]" style={{ color: status?.gateway_running ? "var(--text-success)" : "var(--text-muted)" }}>
                      <span className="w-2 h-2 rounded-full" style={{ background: status?.gateway_running ? "var(--accent-green)" : "var(--text-faint)" }} />
                      {status?.gateway_running ? t("running") : t("stopped")}
                    </span>
                  </div>
                </div>
              </section>
            </>
          )}
        </div>
      </aside>
    </>
  );
});
