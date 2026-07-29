// The dashboard can be served either at the root of its host (e.g.
// https://kanban.tilos.com/) or under a URL prefix when reverse-proxied
// (e.g. https://mission-control.tilos.com/easybci/). The Python backend
// injects ``window.__EASYBCI_BASE_PATH__`` into index.html based on the
// incoming ``X-Forwarded-Prefix`` header so the SPA can address its own
// ``/api/...`` and ``/dashboard-plugins/...`` URLs correctly without a
// rebuild. Empty string means "served at root".
function normalizePrefix(raw: string): string {
  if (!raw) return "";
  const withLead = raw.startsWith("/") ? raw : `/${raw}`;
  return withLead.replace(/\/+$/, "");
}

// Fallback when the injected prefix is missing/empty but we're actually served
// from a sub-path — happens when a reverse proxy rewrites the URL but strips
// the ``X-Forwarded-Prefix`` header, so the backend injects "" yet the browser
// sits at e.g. ``/easybci/``. We recover the mount directory from
// ``location.pathname``. This is safe because the SPA uses hash-based routing
// (no client-side *path* routes), so the pathname only ever reflects the mount
// point, never app navigation state (B14).
function pathnamePrefix(): string {
  if (typeof window === "undefined") return "";
  const p = window.location.pathname || "/";
  // Drop a trailing index.html and any trailing slash → the mount directory.
  const dir = p.replace(/\/index\.html?$/i, "/").replace(/\/+$/, "");
  return dir; // "" at root, "/easybci" (or "/apps/easybci") under a prefix
}

function readBasePath(): string {
  if (typeof window === "undefined") return "";
  const injected = window.__EASYBCI_BASE_PATH__;
  // An explicitly-injected value (including "") is authoritative when it points
  // at a real prefix. Only fall back when it's absent or empty AND the current
  // location is clearly under a sub-path.
  const fromInjected = normalizePrefix(injected ?? "");
  if (fromInjected) return fromInjected;
  return pathnamePrefix();
}

export const EASYBCI_BASE_PATH = readBasePath();
const BASE = EASYBCI_BASE_PATH;

/** Prefix a dashboard-relative path (``/api/...``) with the resolved base path
 *  so it stays correct under a reverse-proxy mount (B14). */
export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

export interface DashboardTheme {
  name: string;
  label: string;
  description: string;
  palette: Record<string, unknown>;
  typography: Record<string, unknown>;
  layout: Record<string, unknown>;
  [key: string]: unknown;
}

// Ephemeral session token for protected endpoints.
// Injected into index.html by the server — never fetched via API.
declare global {
  interface Window {
    __EASYBCI_SESSION_TOKEN__?: string;
    __EASYBCI_BASE_PATH__?: string;
  }
}
let _sessionToken: string | null = null;
const SESSION_HEADER = "X-EasyBCI-Session-Token";

function setSessionHeader(headers: Headers, token: string): void {
  if (!headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, token);
  }
}

// Structured API error so callers can branch on HTTP status (e.g. the 409
// "need_confirm" path for large deletes) and inspect the parsed JSON body.
// FastAPI serialises HTTPException(detail=...) as `{ "detail": <detail> }`, so
// `body` holds the parsed envelope and `detail` the unwrapped inner value.
export class ApiError extends Error {
  status: number;
  body: unknown;
  detail: unknown;
  constructor(status: number, body: unknown, rawText: string) {
    super(`${status}: ${rawText}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.detail =
      body && typeof body === "object" && "detail" in body
        ? (body as { detail: unknown }).detail
        : body;
  }
}

export async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  // Inject the session token into all /api/ requests.
  const headers = new Headers(init?.headers);
  const token = window.__EASYBCI_SESSION_TOKEN__;
  if (token) {
    setSessionHeader(headers, token);
  }
  const res = await fetch(`${BASE}${url}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    let parsed: unknown = undefined;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = undefined;
    }
    throw new ApiError(res.status, parsed, text);
  }
  return res.json();
}

async function getSessionToken(): Promise<string> {
  if (_sessionToken) return _sessionToken;
  const injected = window.__EASYBCI_SESSION_TOKEN__;
  if (injected) {
    _sessionToken = injected;
    return _sessionToken;
  }
  throw new Error("Session token not available — page must be served by the EasyBCI dashboard server");
}

/**
 * Build a URL for the binary file-serving endpoint, suitable for use directly
 * in native ``<img src>`` / ``<embed>`` elements that cannot attach the
 * session-token header. The token is appended as a ``?token=`` query param,
 * which the backend accepts for this route only (B3). The base path prefix is
 * applied so the URL is correct under a reverse-proxy mount.
 */
export function fileServeUrl(path: string): string {
  const token = typeof window !== "undefined" ? window.__EASYBCI_SESSION_TOKEN__ : undefined;
  let url = `${BASE}/api/files/serve?path=${encodeURIComponent(path)}`;
  if (token) url += `&token=${encodeURIComponent(token)}`;
  return url;
}

export const api = {
  getStatus: () => fetchJSON<StatusResponse>("/api/status"),
  getSessions: (limit = 20, offset = 0) =>
    fetchJSON<PaginatedSessions>(`/api/sessions?limit=${limit}&offset=${offset}`),
  getSessionMessages: (id: string) =>
    fetchJSON<SessionMessagesResponse>(`/api/sessions/${encodeURIComponent(id)}/messages`),
  getSessionArtifacts: (id: string) =>
    fetchJSON<SessionArtifacts>(`/api/sessions/${encodeURIComponent(id)}/artifacts`),
  /** Read the canonical analysis_goal enum from the
   *  Dashboard schema endpoint so the goal selector stays in sync with
   *  PLAN_PIPELINE_SCHEMA without hard-coded duplication. */
  getAnalysisGoalEnum: () =>
    fetchJSON<{ options: string[]; description?: string; default?: string; error?: string }>(
      "/api/schema/goal-enum",
    ),
  getSessionVersion: (id: string) =>
    fetchJSON<SessionVersionResponse>(`/api/sessions/${encodeURIComponent(id)}/version`),
  /**
   * Conditional version poll: sends If-None-Match with the last known version.
   * Returns ``{ notModified: true }`` on a 304 so idle polling stays cheap (B7).
   */
  getSessionVersionConditional: async (
    id: string,
    knownVersion: string | null,
  ): Promise<{ notModified: true } | { notModified: false; data: SessionVersionResponse }> => {
    const headers = new Headers();
    const token = typeof window !== "undefined" ? window.__EASYBCI_SESSION_TOKEN__ : undefined;
    if (token) headers.set(SESSION_HEADER, token);
    if (knownVersion) headers.set("If-None-Match", knownVersion);
    const res = await fetch(`${BASE}/api/sessions/${encodeURIComponent(id)}/version`, { headers });
    if (res.status === 304) return { notModified: true };
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`${res.status}: ${text}`);
    }
    return { notModified: false, data: (await res.json()) as SessionVersionResponse };
  },
  getSessionLatestDescendant: (id: string) =>
    fetchJSON<SessionLatestDescendantResponse>(
      `/api/sessions/${encodeURIComponent(id)}/latest-descendant`,
    ),
  deleteSession: (id: string) =>
    fetchJSON<{ ok: boolean; next_id?: string | null }>(`/api/sessions/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  renameSession: (id: string, title: string) =>
    fetchJSON<{ ok: boolean; id: string; title: string | null }>(
      `/api/sessions/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      },
    ),
  getLogs: (params: { file?: string; lines?: number; level?: string; component?: string }) => {
    const qs = new URLSearchParams();
    if (params.file) qs.set("file", params.file);
    if (params.lines) qs.set("lines", String(params.lines));
    if (params.level && params.level !== "ALL") qs.set("level", params.level);
    if (params.component && params.component !== "all") qs.set("component", params.component);
    return fetchJSON<LogsResponse>(`/api/logs?${qs.toString()}`);
  },
  getAnalytics: (days: number) =>
    fetchJSON<AnalyticsResponse>(`/api/analytics/usage?days=${days}`),
  getModelsAnalytics: (days: number) =>
    fetchJSON<ModelsAnalyticsResponse>(`/api/analytics/models?days=${days}`),
  getConfig: () => fetchJSON<Record<string, unknown>>("/api/config"),
  getDefaults: () => fetchJSON<Record<string, unknown>>("/api/config/defaults"),
  getSchema: () => fetchJSON<{ fields: Record<string, unknown>; category_order: string[] }>("/api/config/schema"),
  getModelInfo: () => fetchJSON<ModelInfoResponse>("/api/model/info"),
  getModelOptions: () => fetchJSON<ModelOptionsResponse>("/api/model/options"),
  getAuxiliaryModels: () => fetchJSON<AuxiliaryModelsResponse>("/api/model/auxiliary"),
  setModelAssignment: (body: ModelAssignmentRequest) =>
    fetchJSON<ModelAssignmentResponse>("/api/model/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  saveConfig: (config: Record<string, unknown>) =>
    fetchJSON<{ ok: boolean }>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }),
  getConfigRaw: () => fetchJSON<{ yaml: string }>("/api/config/raw"),
  saveConfigRaw: (yaml_text: string) =>
    fetchJSON<{ ok: boolean }>("/api/config/raw", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text }),
    }),
  getEnvVars: () => fetchJSON<Record<string, EnvVarInfo>>("/api/env"),
  setEnvVar: (key: string, value: string) =>
    fetchJSON<{ ok: boolean }>("/api/env", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, value }),
    }),
  deleteEnvVar: (key: string) =>
    fetchJSON<{ ok: boolean }>("/api/env", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }),
  revealEnvVar: async (key: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ key: string; value: string }>("/api/env/reveal", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [SESSION_HEADER]: token,
      },
      body: JSON.stringify({ key }),
    });
  },

  // Web Search activation — closes the loop on the "use web search when
  // available" mandate so a user picking a provider in Settings actually
  // gets a usable backend (auto-installs the package), and a "Tavily but no
  // key" choice surfaces a warning rather than silently writing config and
  // failing later.
  getWebSearchStatus: () =>
    fetchJSON<WebSearchStatus>("/api/web-search/status"),
  ensureWebSearch: async (backend: "exa" | "tavily") => {
    const token = await getSessionToken();
    return fetchJSON<{
      ok: boolean;
      backend: string;
      installed: boolean;
      available_after: boolean;
      message: string;
    }>("/api/web-search/ensure", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        [SESSION_HEADER]: token,
      },
      body: JSON.stringify({ backend }),
    });
  },

  // Profiles (minimal)
  getProfiles: () =>
    fetchJSON<{ profiles: ProfileInfo[] }>("/api/profiles"),
  createProfile: (body: { name: string; clone_from_default: boolean }) =>
    fetchJSON<{ ok: boolean; name: string; path: string }>("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  renameProfile: (name: string, newName: string) =>
    fetchJSON<{ ok: boolean; name: string; path: string }>(
      `/api/profiles/${encodeURIComponent(name)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName }),
      },
    ),
  deleteProfile: (name: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  getProfileSetupCommand: (name: string) =>
    fetchJSON<{ command: string }>(
      `/api/profiles/${encodeURIComponent(name)}/setup-command`,
    ),
  getProfileSoul: (name: string) =>
    fetchJSON<{ content: string; exists: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}/soul`,
    ),
  updateProfileSoul: (name: string, content: string) =>
    fetchJSON<{ ok: boolean }>(
      `/api/profiles/${encodeURIComponent(name)}/soul`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    ),

  // Skills & Toolsets
  getSkills: () => fetchJSON<SkillInfo[]>("/api/skills"),
  toggleSkill: (name: string, enabled: boolean) =>
    fetchJSON<{ ok: boolean }>("/api/skills/toggle", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, enabled }),
    }),
  getToolsets: () => fetchJSON<ToolsetInfo[]>("/api/tools/toolsets"),

  // Session search (FTS5)
  searchSessions: (q: string) =>
    fetchJSON<SessionSearchResponse>(`/api/sessions/search?q=${encodeURIComponent(q)}`),

  // OAuth provider management
  getOAuthProviders: () =>
    fetchJSON<OAuthProvidersResponse>("/api/providers/oauth"),
  disconnectOAuthProvider: async (providerId: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ ok: boolean; provider: string }>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}`,
      {
        method: "DELETE",
        headers: { [SESSION_HEADER]: token },
      },
    );
  },
  startOAuthLogin: async (providerId: string) => {
    const token = await getSessionToken();
    return fetchJSON<OAuthStartResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [SESSION_HEADER]: token,
        },
        body: "{}",
      },
    );
  },
  submitOAuthCode: async (providerId: string, sessionId: string, code: string) => {
    const token = await getSessionToken();
    return fetchJSON<OAuthSubmitResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [SESSION_HEADER]: token,
        },
        body: JSON.stringify({ session_id: sessionId, code }),
      },
    );
  },
  pollOAuthSession: (providerId: string, sessionId: string) =>
    fetchJSON<OAuthPollResponse>(
      `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`,
    ),
  cancelOAuthSession: async (sessionId: string) => {
    const token = await getSessionToken();
    return fetchJSON<{ ok: boolean }>(
      `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
        headers: { [SESSION_HEADER]: token },
      },
    );
  },

  // Gateway / update actions
  restartGateway: () =>
    fetchJSON<ActionResponse>("/api/gateway/restart", { method: "POST" }),
  updateEasyBCI: () =>
    fetchJSON<ActionResponse>("/api/easybci/update", { method: "POST" }),
  getActionStatus: (name: string, lines = 200) =>
    fetchJSON<ActionStatusResponse>(
      `/api/actions/${encodeURIComponent(name)}/status?lines=${lines}`,
    ),

  // Dashboard plugins
  getPlugins: () =>
    fetchJSON<PluginManifestResponse[]>("/api/dashboard/plugins"),
  rescanPlugins: () =>
    fetchJSON<{ ok: boolean; count: number }>("/api/dashboard/plugins/rescan"),

  getPluginsHub: () => fetchJSON<PluginsHubResponse>("/api/dashboard/plugins/hub"),

  installAgentPlugin: (body: AgentPluginInstallRequest) =>
    fetchJSON<AgentPluginInstallResponse>("/api/dashboard/agent-plugins/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body }),
    }),

  enableAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string; unchanged?: boolean }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/enable`,
      { method: "POST" },
    ),

  disableAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string; unchanged?: boolean }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/disable`,
      { method: "POST" },
    ),

  updateAgentPlugin: (name: string) =>
    fetchJSON<AgentPluginUpdateResponse>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}/update`,
      { method: "POST" },
    ),

  removeAgentPlugin: (name: string) =>
    fetchJSON<{ ok: boolean; name: string }>(
      `/api/dashboard/agent-plugins/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),

  savePluginProviders: (body: PluginProvidersPutRequest) =>
    fetchJSON<{ ok: boolean }>("/api/dashboard/plugin-providers", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  setPluginVisibility: (name: string, hidden: boolean) =>
    fetchJSON<{ ok: boolean; name: string; hidden: boolean }>(
      `/api/dashboard/plugins/${encodeURIComponent(name)}/visibility`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hidden }),
      },
    ),

  // Dashboard themes
  getThemes: () =>
    fetchJSON<DashboardThemesResponse>("/api/dashboard/themes"),
  setTheme: (name: string) =>
    fetchJSON<{ ok: boolean; theme: string }>("/api/dashboard/theme", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  // File system
  getFileTree: (path: string, maxDepth = 2) =>
    fetchJSON<{ files: FileTreeNode[] }>(
      `/api/files/tree?path=${encodeURIComponent(path)}&max_depth=${maxDepth}`,
    ),
  readFile: (path: string) =>
    fetchJSON<FileReadResponse>(
      `/api/files/read?path=${encodeURIComponent(path)}`,
    ),
  getDirSummary: (path: string) =>
    fetchJSON<DirSummary>(
      `/api/files/summary?path=${encodeURIComponent(path)}`,
    ),
  // Delete a file/dir inside a mini-repo work_dir. May throw a 409 whose body
  // carries `{ need_confirm, files, bytes }` — callers should surface that as
  // a confirmation dialog and retry with `confirm=true`.
  deleteFile: (path: string, confirm = false) =>
    fetchJSON<DeleteFileResponse>(
      `/api/files?path=${encodeURIComponent(path)}&confirm=${confirm ? "true" : "false"}`,
      { method: "DELETE" },
    ),
};

export interface DirSummary {
  path: string;
  available: boolean;
  file_count: number;
  total_size_bytes: number;
  ext_histogram: Record<string, number>;
  neural?: {
    modality_guess: string;
    n_channels: number;
    duration_sec: number;
    sample_rate_hz: number;
    channel_names_preview: string[];
  };
  reason?: string;
}

export interface DeleteFileResponse {
  ok: boolean;
  path: string;
  files_deleted: number;
  bytes_deleted: number;
}

export interface ActionResponse {
  name: string;
  ok: boolean;
  pid: number;
}

export interface ActionStatusResponse {
  exit_code: number | null;
  lines: string[];
  name: string;
  pid: number | null;
  running: boolean;
}

export interface PlatformStatus {
  error_code?: string;
  error_message?: string;
  state: string;
  updated_at: string;
}

export interface StatusResponse {
  active_sessions: number;
  config_path: string;
  config_version: number;
  env_path: string;
  gateway_exit_reason: string | null;
  gateway_health_url: string | null;
  gateway_pid: number | null;
  gateway_platforms: Record<string, PlatformStatus>;
  gateway_running: boolean;
  gateway_state: string | null;
  gateway_updated_at: string | null;
  easybci_home: string;
  latest_config_version: number;
  release_date: string;
  version: string;
}

export interface SessionInfo {
  id: string;
  source: string | null;
  model: string | null;
  title: string | null;
  started_at: number;
  ended_at: number | null;
  last_active: number;
  is_active: boolean;
  message_count: number;
  tool_call_count: number;
  input_tokens: number;
  output_tokens: number;
  preview: string | null;
  parent_session_id?: string | null;
}

export interface SessionLatestDescendantResponse {
  requested_session_id: string;
  session_id: string;
  path: string[];
  changed: boolean;
}

export interface PaginatedSessions {
  sessions: SessionInfo[];
  total: number;
  limit: number;
  offset: number;
}

export interface EnvVarInfo {
  is_set: boolean;
  redacted_value: string | null;
  description: string;
  url: string | null;
  category: string;
  is_password: boolean;
  tools: string[];
  advanced: boolean;
}

export interface WebSearchProviderStatus {
  name: string;
  display_name: string;
  registered: boolean;
  supports_search: boolean;
  supports_extract: boolean;
  supports_crawl: boolean;
  is_available: boolean;
  configured_as_search: boolean;
  configured_as_extract: boolean;
  diagnostic: { reason: string; fix_hint: string } | null;
}

export interface WebSearchStatus {
  providers: WebSearchProviderStatus[];
  active_search: string | null;
  active_search_strict: string | null;
  active_extract: string | null;
  active_crawl: string | null;
}

export interface SessionMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null;
  tool_calls?: Array<{
    id: string;
    function: { name: string; arguments: string };
  }>;
  tool_name?: string;
  tool_call_id?: string;
  timestamp?: number;
  /** Tool-role outcome status. NULL on legacy rows. */
  tool_status?: "done" | "error" | null;
  /** Tool-role wall-clock duration in seconds. */
  tool_duration?: number | null;
  /** Assistant reasoning text (provider-dependent). */
  reasoning?: string | null;
  /** Alternate reasoning payload from some providers. */
  reasoning_content?: string | null;
  /** Structured reasoning trace (provider-specific shape). */
  reasoning_details?: unknown;
}

export interface SessionArtifacts {
  session_id: string;
  available: boolean;
  source_dir?: string;
  output_dir?: string;
  work_dir?: string;
  pipeline_yaml?: string;
  /** Absolute path to the pipeline source file (code/pipeline.py) — present
   *  when it exists on disk. Used by the Pipeline card's Reveal/Delete actions. */
  pipeline_yaml_path?: string;
  /** Absolute path to the QC_out directory — present when it exists. Used by
   *  the QC card's Reveal/Delete actions. */
  qc_dir_path?: string;
  qc?: {
    snr?: number;
    artifact_ratio?: number;
    quality_score?: number;
  };
  evidence?: Array<{ step: string; figure: string; note?: string }>;
  pipeline_record_path?: string | null;
  parse_error?: string;
  /** analysis_goal (single source of truth from
   *  pipeline_record.json or plan/goal.json). One of:
   *  classification | source_localization | feature_extraction |
   *  clinical_screening | exploratory | generic */
  analysis_goal?: string;
  /** web_evidence captured at propose_pipeline time. ``status``
   *  is "ok" when research_preprocessing was called successfully and the
   *  recommendations list is non-empty; "unavailable" otherwise. */
  web_evidence?: {
    status?: "ok" | "unavailable" | string;
    provider?: string | null;
    confidence?: number | null;
    question?: string;
    reason?: string;
    recommendations?: Array<{ param?: string; value?: string | number }>;
    citations?: Array<{ url?: string; title?: string }>;
    applied_to_steps?: string[];
    conflicts?: Array<{
      param: string;
      skill_default?: unknown;
      web_recommended?: unknown;
      decision: string;
      reason?: string;
    }>;
  };
  /** layout_repair summary — populated by finalize_work_dir when
   *  verify_and_repair ran on the mini-repo. Absent when the run predates
   *  the strict-layout-enforcement plan or finalize was never invoked. */
  layout_repair?: {
    initial_violations: number;
    remaining_violations: number;
    rounds: number;
    unrepairable?: string[];
    report_available?: boolean;
    report_path?: string;
  };
}

export interface SessionMessagesResponse {
  session_id: string;
  messages: SessionMessage[];
  /** Optional version stamp — present after C3 backend roll-out. */
  version?: string;
}

export interface SessionVersionResponse {
  session_id: string;
  version: string;
  message_count: number;
  last_timestamp: number;
}

export interface LogsResponse {
  file: string;
  lines: string[];
}

export interface AnalyticsDailyEntry {
  day: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  reasoning_tokens: number;
  estimated_cost: number;
  actual_cost: number;
  sessions: number;
  api_calls: number;
}

export interface AnalyticsModelEntry {
  model: string;
  input_tokens: number;
  output_tokens: number;
  estimated_cost: number;
  sessions: number;
  api_calls: number;
}

export interface AnalyticsSkillEntry {
  skill: string;
  view_count: number;
  manage_count: number;
  total_count: number;
  percentage: number;
  last_used_at: number | null;
}

export interface AnalyticsSkillsSummary {
  total_skill_loads: number;
  total_skill_edits: number;
  total_skill_actions: number;
  distinct_skills_used: number;
}

export interface AnalyticsResponse {
  daily: AnalyticsDailyEntry[];
  by_model: AnalyticsModelEntry[];
  totals: {
    total_input: number;
    total_output: number;
    total_cache_read: number;
    total_reasoning: number;
    total_estimated_cost: number;
    total_actual_cost: number;
    total_sessions: number;
    total_api_calls: number;
  };
  skills: {
    summary: AnalyticsSkillsSummary;
    top_skills: AnalyticsSkillEntry[];
  };
}

export interface ProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
}

export interface ModelsAnalyticsModelEntry {
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  reasoning_tokens: number;
  estimated_cost: number;
  actual_cost: number;
  sessions: number;
  api_calls: number;
  tool_calls: number;
  last_used_at: number;
  avg_tokens_per_session: number;
  capabilities: {
    supports_tools?: boolean;
    supports_vision?: boolean;
    supports_reasoning?: boolean;
    context_window?: number;
    max_output_tokens?: number;
    model_family?: string;
  };
}

export interface ModelsAnalyticsResponse {
  models: ModelsAnalyticsModelEntry[];
  totals: {
    distinct_models: number;
    total_input: number;
    total_output: number;
    total_cache_read: number;
    total_reasoning: number;
    total_estimated_cost: number;
    total_actual_cost: number;
    total_sessions: number;
    total_api_calls: number;
  };
  period_days: number;
}

export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  enabled: boolean;
}

export interface ToolsetInfo {
  name: string;
  label: string;
  description: string;
  enabled: boolean;
  configured: boolean;
  tools: string[];
}

export interface SessionSearchResult {
  session_id: string;
  snippet: string;
  role: string | null;
  source: string | null;
  model: string | null;
  session_started: number | null;
}

export interface SessionSearchResponse {
  results: SessionSearchResult[];
}

// ── Model info types ──────────────────────────────────────────────────

export interface ModelInfoResponse {
  model: string;
  provider: string;
  auto_context_length: number;
  config_context_length: number;
  effective_context_length: number;
  capabilities: {
    supports_tools?: boolean;
    supports_vision?: boolean;
    supports_reasoning?: boolean;
    context_window?: number;
    max_output_tokens?: number;
    model_family?: string;
  };
}

// ── Model options / assignment types ──────────────────────────────────

export interface ModelOptionProvider {
  name: string;
  slug: string;
  models?: string[];
  total_models?: number;
  is_current?: boolean;
  is_user_defined?: boolean;
  source?: string;
  warning?: string;
}

export interface ModelOptionsResponse {
  model?: string;
  provider?: string;
  providers?: ModelOptionProvider[];
}

export interface AuxiliaryTaskAssignment {
  task: string;
  provider: string;
  model: string;
  base_url: string;
}

export interface AuxiliaryModelsResponse {
  tasks: AuxiliaryTaskAssignment[];
  main: { provider: string; model: string };
}

export interface ModelAssignmentRequest {
  scope: "main" | "auxiliary";
  provider: string;
  model: string;
  /** For auxiliary: task slot name, "" for all, "__reset__" to reset all. */
  task?: string;
}

export interface ModelAssignmentResponse {
  ok: boolean;
  scope?: string;
  provider?: string;
  model?: string;
  tasks?: string[];
  reset?: boolean;
}

// ── OAuth provider types ────────────────────────────────────────────────

export interface OAuthProviderStatus {
  logged_in: boolean;
  source?: string | null;
  source_label?: string | null;
  token_preview?: string | null;
  expires_at?: string | null;
  has_refresh_token?: boolean;
  last_refresh?: string | null;
  error?: string;
}

export interface OAuthProvider {
  id: string;
  name: string;
  /** "pkce" (browser redirect + paste code), "device_code" (show code + URL),
   *  or "external" (delegated to a separate CLI like Claude Code or Qwen). */
  flow: "pkce" | "device_code" | "external";
  cli_command: string;
  docs_url: string;
  status: OAuthProviderStatus;
}

export interface OAuthProvidersResponse {
  providers: OAuthProvider[];
}

/** Discriminated union — the shape of /start depends on the flow. */
export type OAuthStartResponse =
  | {
      session_id: string;
      flow: "pkce";
      auth_url: string;
      expires_in: number;
    }
  | {
      session_id: string;
      flow: "device_code";
      user_code: string;
      verification_url: string;
      expires_in: number;
      poll_interval: number;
    };

export interface OAuthSubmitResponse {
  ok: boolean;
  status: "approved" | "error";
  message?: string;
}

export interface OAuthPollResponse {
  session_id: string;
  status: "pending" | "approved" | "denied" | "expired" | "error";
  error_message?: string | null;
  expires_at?: number | null;
}

// ── Dashboard theme types ──────────────────────────────────────────────

export interface DashboardThemeSummary {
  description: string;
  label: string;
  name: string;
  /** Full theme definition for user themes; undefined for built-ins
   *  (which the frontend already has locally). */
  definition?: DashboardTheme;
}

export interface DashboardThemesResponse {
  active: string;
  themes: DashboardThemeSummary[];
}

// ── Dashboard plugin types ─────────────────────────────────────────────

export interface PluginManifestResponse {
  name: string;
  label: string;
  description: string;
  icon: string;
  version: string;
  tab: {
    path: string;
    position?: string;
    override?: string;
    hidden?: boolean;
  };
  slots?: string[];
  entry: string;
  css?: string | null;
  has_api: boolean;
  source: string;
}

export interface HubAgentPluginRow {
  name: string;
  version: string;
  description: string;
  source: string;
  runtime_status: "disabled" | "enabled" | "inactive";
  has_dashboard_manifest: boolean;
  dashboard_manifest: PluginManifestResponse | null;
  path: string;
  can_remove: boolean;
  can_update_git: boolean;
  auth_required: boolean;
  auth_command: string;
  user_hidden: boolean;
}

export interface PluginsHubProviders {
  memory_provider: string;
  memory_options: Array<{ name: string; description: string }>;
  context_engine: string;
  context_options: Array<{ name: string; description: string }>;
}

export interface PluginsHubResponse {
  plugins: HubAgentPluginRow[];
  orphan_dashboard_plugins: PluginManifestResponse[];
  providers: PluginsHubProviders;
}

export interface AgentPluginInstallRequest {
  identifier: string;
  force?: boolean;
  enable?: boolean;
}

export interface AgentPluginInstallResponse {
  ok: boolean;
  plugin_name?: string;
  warnings?: string[];
  missing_env?: string[];
  after_install_path?: string | null;
  enabled?: boolean;
  error?: string;
}

export interface AgentPluginUpdateResponse {
  ok: boolean;
  name?: string;
  output?: string;
  unchanged?: boolean;
  error?: string;
}

export interface PluginProvidersPutRequest {
  memory_provider?: string;
  context_engine?: string;
}

// ── File system types ─────────────────────────────────────────────────

export interface FileTreeNode {
  name: string;
  path: string;
  type: "file" | "folder";
  size?: number;
  children?: FileTreeNode[];
}

export interface FileReadResponse {
  path: string;
  content?: string;
  size: number;
  is_text: boolean;
  truncated?: boolean;
}
