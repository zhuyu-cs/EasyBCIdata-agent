import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { api, fetchJSON, ApiError } from "@/lib/api";
import type { SessionArtifacts } from "@/lib/api";
import type { RunEvent } from "@/lib/runsClient";
import { warn } from "@/lib/debug";
import { t } from "@/lib/i18n";

export interface FileNode {
  name: string;
  path: string;
  type: "file" | "folder";
  size?: number;
  children?: FileNode[];
  /** Set by the backend when a folder sits at the depth limit and has unshown
   *  contents (B10). */
  depth_limited?: boolean;
}

export interface TreeTruncation {
  /** Number of sibling entries dropped because a folder exceeded the cap. */
  count: number;
  /** True when at least one folder was cut off by the depth limit. */
  depth: boolean;
}

interface WorkspaceState {
  sourceDir: string | null;
  sourceFiles: FileNode[];
  sourceTrunc: TreeTruncation | null;
  outputDir: string | null;
  outputFiles: FileNode[];
  outputTrunc: TreeTruncation | null;
  sourceLoading: boolean;
  outputLoading: boolean;
  /** Either tree is loading — kept for the header spinner. */
  loading: boolean;
  /** What the session's pipeline_record says about its run, once known. */
  artifacts: SessionArtifacts | null;
  /** The first artifacts lookup for this session has finished. */
  settled: boolean;
}

interface TreeResult {
  files: FileNode[];
  trunc: TreeTruncation;
  /** False when the listing failed (e.g. the directory doesn't exist). */
  ok: boolean;
}

async function fetchFileTree(dirPath: string, maxDepth = 2): Promise<TreeResult> {
  try {
    const data = await fetchJSON<{
      files: FileNode[];
      truncated_at_count?: number;
      truncated_at_depth?: boolean;
    }>(`/api/files/tree?path=${encodeURIComponent(dirPath)}&max_depth=${maxDepth}`);
    return {
      files: data.files,
      trunc: {
        count: data.truncated_at_count ?? 0,
        depth: !!data.truncated_at_depth,
      },
      ok: true,
    };
  } catch {
    return { files: [], trunc: { count: 0, depth: false }, ok: false };
  }
}

/** If the path looks like a file (has a known extension), return its parent
 *  directory; otherwise treat it as a directory and return it as-is. */
function dirnameIfFile(p: string): string {
  const base = p.split("/").pop() ?? "";
  const ext = base.includes(".") ? base.split(".").pop()?.toLowerCase() : undefined;
  if (ext && KNOWN_EXTENSIONS.has(ext)) {
    const dir = p.replace(/\/[^/]+$/, "");
    return dir || p;
  }
  return p.replace(/\/+$/, "") || p;
}

const KNOWN_EXTENSIONS = new Set([
  "edf", "bdf", "fif", "set", "vhdr", "eeg", "cnt", "mat", "h5", "hdf5",
  "npy", "npz", "csv", "tsv", "json", "yaml", "yml", "txt", "py",
  "png", "jpg", "svg", "pdf", "md", "toml",
]);


const EMPTY_STATE: WorkspaceState = {
  sourceDir: null,
  sourceFiles: [],
  sourceTrunc: null,
  outputDir: null,
  outputFiles: [],
  outputTrunc: null,
  sourceLoading: false,
  outputLoading: false,
  loading: false,
  artifacts: null,
  settled: false,
};

// A reload starts from the last layout, so the conversation doesn't shift.
const PANEL_VISIBLE_KEY = "easybci-workspace-visible";

function withLoading(s: WorkspaceState): WorkspaceState {
  return { ...s, loading: s.sourceLoading || s.outputLoading };
}

export function useWorkspace() {
  const [state, setRawState] = useState<WorkspaceState>(EMPTY_STATE);
  const setState = useCallback(
    (fn: (prev: WorkspaceState) => WorkspaceState) => setRawState((prev) => withLoading(fn(prev))),
    [],
  );
  // Session the panel currently belongs to. Every async result checks it so a
  // slow response for a previous session can never paint into the new one.
  const sessionRef = useRef<string | null | undefined>(undefined);

  const outputDirRef = useRef<string | null>(null);
  const sourceDirRef = useRef<string | null>(null);
  // Current requested depth per directory — bumped by loadDeeper (B10).
  const sourceDepthRef = useRef(2);
  const outputDepthRef = useRef(2);
  // LIFO buffer of candidate paths extracted from tool.started, keyed by tool
  // name (matches useConversation's tool.completed pairing). The candidate is
  // only committed to source/outputDir when tool.completed fires with no error;
  // a failed tool's candidate is discarded. Without this buffer, the very first
  // tool.started — even if the tool will fail with "Directory not found" —
  // would lock the wrong path into the panel and the fallbackActive guard
  // would silently drop the agent's later corrected path.
  const pendingToolPathsRef = useRef<Array<{
    tool: string;
    sourceCandidate: string | null;
    outputCandidate: string | null;
  }>>([]);

  const setSourceDir = useCallback(async (dir: string) => {
    if (dir === sourceDirRef.current) return;
    sourceDirRef.current = dir;
    sourceDepthRef.current = 2;
    setState((prev) => ({ ...prev, sourceDir: dir, sourceLoading: true }));
    const tree = await fetchFileTree(dir, sourceDepthRef.current);
    if (sourceDirRef.current !== dir) return;
    setState((prev) => ({
      ...prev,
      sourceFiles: tree.files,
      sourceTrunc: tree.trunc,
      sourceLoading: false,
    }));
  }, [setState]);

  /** `guessed`: the path was inferred from tool args rather than reported by
   *  /artifacts — drop it again if it doesn't exist on disk. */
  const setOutputDir = useCallback(async (dir: string, opts: { guessed?: boolean } = {}) => {
    if (dir === outputDirRef.current) return;
    outputDirRef.current = dir;
    outputDepthRef.current = 2;
    setState((prev) => ({ ...prev, outputDir: dir, outputLoading: true }));
    const { files, trunc, ok } = await fetchFileTree(dir, outputDepthRef.current);
    if (outputDirRef.current !== dir) return;
    if (!ok && opts.guessed) {
      outputDirRef.current = null;
      setState((prev) => ({ ...prev, outputDir: null, outputFiles: [], outputTrunc: null, outputLoading: false }));
      return;
    }
    setState((prev) => ({ ...prev, outputFiles: files, outputTrunc: trunc, outputLoading: false }));
  }, [setState]);

  const refreshOutput = useCallback(async () => {
    const dir = outputDirRef.current;
    if (!dir) return;
    setState((prev) => ({ ...prev, outputLoading: true }));
    const { files, trunc } = await fetchFileTree(dir, outputDepthRef.current);
    if (outputDirRef.current !== dir) return;
    setState((prev) => ({ ...prev, outputFiles: files, outputTrunc: trunc, outputLoading: false }));
  }, [setState]);

  const refreshSource = useCallback(async () => {
    const dir = sourceDirRef.current;
    if (!dir) return;
    setState((prev) => ({ ...prev, sourceLoading: true }));
    const { files, trunc } = await fetchFileTree(dir, sourceDepthRef.current);
    if (sourceDirRef.current !== dir) return;
    setState((prev) => ({ ...prev, sourceFiles: files, sourceTrunc: trunc, sourceLoading: false }));
  }, [setState]);

  // Re-fetch a directory tree one level deeper — used by the "load deeper"
  // affordance when the listing was depth-limited (B10).
  const loadDeeper = useCallback(async (which: "source" | "output") => {
    const dir = which === "source" ? sourceDirRef.current : outputDirRef.current;
    if (!dir) return;
    const depthRef = which === "source" ? sourceDepthRef : outputDepthRef;
    depthRef.current = Math.min(depthRef.current + 1, 6);
    const loadingKey = which === "source" ? "sourceLoading" : "outputLoading";
    setState((prev) => ({ ...prev, [loadingKey]: true }));
    const { files, trunc } = await fetchFileTree(dir, depthRef.current);
    const current = which === "source" ? sourceDirRef.current : outputDirRef.current;
    if (current !== dir) return;
    setState((prev) =>
      which === "source"
        ? { ...prev, sourceFiles: files, sourceTrunc: trunc, sourceLoading: false }
        : { ...prev, outputFiles: files, outputTrunc: trunc, outputLoading: false },
    );
  }, [setState]);

  // Delete a file/dir inside a mini-repo work_dir, then refresh both trees so
  // the panels reflect disk. The backend rejects source-side / out-of-bounds
  // paths (403) and demands confirmation for large deletes (409). We surface
  // the 409 as a native confirm and retry with confirm=true. Returns true when
  // a delete actually happened, false when the user declined.
  const deleteFile = useCallback(async (path: string): Promise<boolean> => {
    try {
      await api.deleteFile(path);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        const d = (e.detail ?? {}) as { files?: number; bytes?: number };
        const mb = ((d.bytes ?? 0) / (1024 * 1024)).toFixed(1);
        const ok = window.confirm(
          t("Will delete {n} file(s) ({m} MB). Confirm?", { n: d.files ?? "?", m: mb }),
        );
        if (!ok) return false;
        await api.deleteFile(path, true);
      } else {
        throw e;
      }
    }
    await refreshOutput();
    return true;
  }, [refreshOutput]);


  // (e.g. activeSessionId stable but reloadCount bumped) share one request.
  const artifactsInflightRef = useRef<{ sid: string; promise: Promise<void> } | null>(null);
  // Throttle for live SSE-driven artifacts refresh during a run. Budget: ≤ ~10
  // artifacts requests per 5-minute run → leading-edge fire then a 30s minimum
  // gap, with a trailing-edge dispatch so a burst still resolves once it ends.
  const lastArtifactsAtRef = useRef(0);
  const artifactsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ARTIFACTS_THROTTLE_MS = 30_000;

  const loadFromArtifacts = useCallback(async (sessionId: string) => {
    const inflight = artifactsInflightRef.current;
    if (inflight && inflight.sid === sessionId) return inflight.promise;
    const promise = (async () => {
      try {
        const art = await api.getSessionArtifacts(sessionId);
        if (sessionRef.current !== sessionId || !art.available) return;
        setState((prev) => ({ ...prev, artifacts: art }));
        if (art.source_dir) await setSourceDir(art.source_dir);
        if (sessionRef.current !== sessionId) return;
        const outDir = art.work_dir ?? art.output_dir;
        if (outDir) await setOutputDir(outDir);
      } catch (err) {
        warn("loadFromArtifacts", "artifacts fetch failed; panels keep their last state", err);
      } finally {
        if (artifactsInflightRef.current?.sid === sessionId) {
          artifactsInflightRef.current = null;
        }
        if (sessionRef.current === sessionId) setState((prev) => ({ ...prev, settled: true }));
      }
    })();
    artifactsInflightRef.current = { sid: sessionId, promise };
    return promise;
  }, [setSourceDir, setOutputDir, setState]);

  // Throttled artifacts refresh for use during a live SSE run. Leading-edge
  // fires immediately if we haven't fetched in the last ARTIFACTS_THROTTLE_MS;
  // otherwise schedules a single trailing-edge dispatch at the deadline so the
  // last event in a burst still triggers a refresh after the run quiesces.
  const refreshFromArtifacts = useCallback((sessionId: string) => {
    const now = performance.now();
    const elapsed = now - lastArtifactsAtRef.current;
    if (elapsed < ARTIFACTS_THROTTLE_MS) {
      if (artifactsTimerRef.current) clearTimeout(artifactsTimerRef.current);
      const remaining = ARTIFACTS_THROTTLE_MS - elapsed;
      artifactsTimerRef.current = setTimeout(() => {
        lastArtifactsAtRef.current = performance.now();
        artifactsTimerRef.current = null;
        if (sessionRef.current === sessionId) loadFromArtifacts(sessionId);
      }, remaining);
      return;
    }
    lastArtifactsAtRef.current = now;
    loadFromArtifacts(sessionId);
  }, [loadFromArtifacts]);

  const reset = useCallback(() => {
    outputDirRef.current = null;
    sourceDirRef.current = null;
    sourceDepthRef.current = 2;
    outputDepthRef.current = 2;
    pendingToolPathsRef.current = [];
    if (artifactsTimerRef.current) {
      clearTimeout(artifactsTimerRef.current);
      artifactsTimerRef.current = null;
    }
    lastArtifactsAtRef.current = 0;
    artifactsInflightRef.current = null;
    setRawState(EMPTY_STATE);
  }, []);

  /** Point the panel at `sessionId`, clearing everything from the previous
   *  session first. A no-op when it already shows that session. */
  const bindSession = useCallback((sessionId: string | null) => {
    if (sessionRef.current === sessionId) return;
    sessionRef.current = sessionId;
    reset();
    if (sessionId === null) setRawState({ ...EMPTY_STATE, settled: true });
  }, [reset]);

  const handleRunEvent = useCallback(
    (ev: RunEvent, sessionId?: string | null) => {
      if (sessionId && sessionRef.current !== sessionId) return;
      // Fallback path-sniffing only fires before artifacts has bootstrapped
      // either dir — once we've got authoritative source/output paths from
      // /artifacts, regex/keyword scraping from preview text is just noise.
      const fallbackActive = sourceDirRef.current === null && outputDirRef.current === null;

      if (ev.event === "tool.started") {
        // Prefer structured path args forwarded by the gateway (B8) — these are
        // authoritative, unlike regex-scraped text. Route output-ish keys to
        // the output dir and input-ish keys to the source dir.
        //
        // Note: `directory` / `dir` are INPUT keys for tools like list_data /
        // inspect_directory (where the agent points at the raw-data folder to
        // enumerate). Routing them to outputDir made the workspace render only
        // an "Output" pane whose contents were the source dataset — exactly
        // the source/output mix-up users reported.
        //
        // The candidate paths are buffered here and only committed to state
        // when tool.completed reports success — see the tool.completed branch
        // below. Committing synchronously here meant a failed list_data with
        // a typo'd path would permanently freeze the workspace panel onto the
        // bad path even after the agent self-corrected and retried.
        let sourceCandidate: string | null = null;
        let outputCandidate: string | null = null;

        if (ev.paths && fallbackActive) {
          const OUTPUT_KEYS = ["output_path", "output_dir", "work_dir"];
          const INPUT_KEYS = [
            "path", "file", "file_path", "input_path", "data_path",
            "directory", "dir",
          ];
          for (const key of OUTPUT_KEYS) {
            const val = ev.paths[key];
            if (val) { outputCandidate = dirnameIfFile(val); break; }
          }
          for (const key of INPUT_KEYS) {
            const val = ev.paths[key];
            if (val) { sourceCandidate = dirnameIfFile(val); break; }
          }
        }

        const toolName = ev.tool ?? "";
        if (sourceCandidate !== null || outputCandidate !== null) {
          pendingToolPathsRef.current.push({
            tool: toolName,
            sourceCandidate,
            outputCandidate,
          });
        }
      }

      if (ev.event === "tool.completed") {
        // Pair with the matching tool.started entry (LIFO by tool name, mirrors
        // how a turn pairs its steps). On success, commit
        // the buffered candidate; on error, drop it so the agent's retry path
        // can flow through cleanly. The `!ref.current` checks preserve the
        // original first-write-wins semantic against artifacts-derived paths.
        const stack = pendingToolPathsRef.current;
        const idx = stack.findLastIndex((e) => e.tool === ev.tool);
        if (idx >= 0) {
          const entry = stack[idx];
          stack.splice(idx, 1);
          if (!ev.error) {
            if (entry.outputCandidate && !outputDirRef.current) {
              setOutputDir(entry.outputCandidate, { guessed: true });
            }
            if (entry.sourceCandidate && !sourceDirRef.current) {
              setSourceDir(entry.sourceCandidate);
            }
          }
        }

        // Artifacts is the authoritative source. Refresh it first
        // (throttled), then refresh the file trees we already know about.
        if (sessionId) refreshFromArtifacts(sessionId);
        // A tool just finished — its handler may have created the output
        // directory or written into it. The source side is read-only, so only
        // the output tree needs a refresh.
        if (outputDirRef.current) refreshOutput();
      }

      if (ev.event === "run.completed") {
        if (sessionId) refreshFromArtifacts(sessionId);
        refreshOutput();
      }
    },
    [setSourceDir, setOutputDir, refreshOutput, refreshFromArtifacts],
  );

  // The panel appears once there is something to show. Between two sessions it
  // holds until the new one has answered, so the layout doesn't jump twice.
  const hasContent = state.sourceFiles.length > 0 || state.outputFiles.length > 0 || !!state.artifacts;
  const [panelVisible, setPanelVisible] = useState(() => {
    try { return sessionStorage.getItem(PANEL_VISIBLE_KEY) === "1"; } catch { return false; }
  });
  if (hasContent && !panelVisible) setPanelVisible(true);
  else if (!hasContent && panelVisible && state.settled) setPanelVisible(false);
  useEffect(() => {
    try { sessionStorage.setItem(PANEL_VISIBLE_KEY, panelVisible ? "1" : "0"); } catch { /* private mode */ }
  }, [panelVisible]);

  // Stable identity between state changes so the memoized panel only
  // re-renders when the workspace itself changes.
  return useMemo(
    () => ({ ...state, panelVisible, setSourceDir, setOutputDir, reset, bindSession, handleRunEvent, loadDeeper, loadFromArtifacts, refreshFromArtifacts, refreshSource, refreshOutput, deleteFile }),
    [state, panelVisible, setSourceDir, setOutputDir, reset, bindSession, handleRunEvent, loadDeeper, loadFromArtifacts, refreshFromArtifacts, refreshSource, refreshOutput, deleteFile],
  );
}
