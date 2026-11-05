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

// Ephemeral session token for protected endpoints.
// Injected into index.html by the server — never fetched via API.
declare global {
  interface Window {
    __EASYBCI_SESSION_TOKEN__?: string;
    __EASYBCI_BASE_PATH__?: string;
  }
}
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

let tokenRefresh: Promise<boolean> | null = null;

/**
 * The token lives as long as the dashboard process, so a page that outlived a
 * restart holds a dead one. index.html carries the current token; true when it
 * differs from ours. Concurrent 401s share one read.
 */
function refreshSessionToken(): Promise<boolean> {
  tokenRefresh ??= fetch(`${BASE}/`, { cache: "no-store" })
    .then((res) => (res.ok ? res.text() : ""))
    .then((html) => {
      const fresh = html.match(/__EASYBCI_SESSION_TOKEN__="([^"]+)"/)?.[1];
      if (!fresh || fresh === window.__EASYBCI_SESSION_TOKEN__) return false;
      window.__EASYBCI_SESSION_TOKEN__ = fresh;
      return true;
    })
    .catch(() => false)
    .finally(() => {
      tokenRefresh = null;
    });
  return tokenRefresh;
}

export async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const send = () => {
    // Inject the session token into all /api/ requests.
    const headers = new Headers(init?.headers);
    const token = window.__EASYBCI_SESSION_TOKEN__;
    if (token) {
      setSessionHeader(headers, token);
    }
    return fetch(`${BASE}${url}`, { ...init, headers });
  };
  let res = await send();
  if (res.status === 401 && (await refreshSessionToken())) res = await send();
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
  /**
   * Version stamp for a session. With a known version it is sent as
   * If-None-Match and an unchanged session answers 304 → `notModified`, which
   * keeps idle polling near-free.
   */
  getSessionVersionConditional: async (
    id: string,
    knownVersion: string | null,
  ): Promise<{ notModified: true } | { notModified: false; data: SessionVersionResponse }> => {
    try {
      const data = await fetchJSON<SessionVersionResponse>(
        `/api/sessions/${encodeURIComponent(id)}/version`,
        knownVersion ? { headers: { "If-None-Match": knownVersion } } : undefined,
      );
      return { notModified: false, data };
    } catch (e) {
      if (e instanceof ApiError && e.status === 304) return { notModified: true };
      throw e;
    }
  },
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
  getConfig: () => fetchJSON<Record<string, unknown>>("/api/config"),
  getModelInfo: () => fetchJSON<ModelInfoResponse>("/api/model/info"),
  getModelOptions: () => fetchJSON<ModelOptionsResponse>("/api/model/options"),
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
  revealEnvVar: (key: string) =>
    fetchJSON<{ key: string; value: string }>("/api/env/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }),

  // Web Search activation — closes the loop on the "use web search when
  // available" mandate so a user picking a provider in Settings actually
  // gets a usable backend (auto-installs the package), and a "Tavily but no
  // key" choice surfaces a warning rather than silently writing config and
  // failing later.
  getWebSearchStatus: () =>
    fetchJSON<WebSearchStatus>("/api/web-search/status"),

  getToolsets: () => fetchJSON<ToolsetInfo[]>("/api/tools/toolsets"),

  // Session search (FTS5)
  searchSessions: (q: string) =>
    fetchJSON<SessionSearchResponse>(`/api/sessions/search?q=${encodeURIComponent(q)}`),

  setTheme: (name: string) =>
    fetchJSON<{ ok: boolean; theme: string }>("/api/dashboard/theme", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  // Delete a file/dir inside a mini-repo work_dir. May throw a 409 whose body
  // carries `{ need_confirm, files, bytes }` — callers should surface that as
  // a confirmation dialog and retry with `confirm=true`.
  deleteFile: (path: string, confirm = false) =>
    fetchJSON<DeleteFileResponse>(
      `/api/files?path=${encodeURIComponent(path)}&confirm=${confirm ? "true" : "false"}`,
      { method: "DELETE" },
    ),
};

export interface DeleteFileResponse {
  ok: boolean;
  path: string;
  files_deleted: number;
  bytes_deleted: number;
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
  /** The session's mini-repo, once a preprocessing run registered one. */
  work_dir?: string | null;
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
  /** DB row id — stable across reloads, used for render keys. */
  id?: number;
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
  /** Pipeline steps as recorded (`bandpass:0.5,40`, …). */
  steps?: string[];
  qc_grade?: string;
  elapsed_s?: number;
  status?: string;
  deliverables?: string[];
  epoch_window?: [number, number];
  /** First figures under preprocessed_output/figures, with the total count. */
  figures?: string[];
  figure_count?: number;
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
  title?: string | null;
  /** Matched text, with hits wrapped in `>>>` and `<<<`. */
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

// ── File system types ─────────────────────────────────────────────────

export interface FileTreeNode {
  name: string;
  path: string;
  type: "file" | "folder";
  size?: number;
  children?: FileTreeNode[];
}

