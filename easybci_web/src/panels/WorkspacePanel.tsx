import { useState, useMemo, useCallback, lazy, Suspense } from "react";
import type { FileNode, TreeTruncation } from "@/hooks/useWorkspace";
import { ContextMenu, type ContextMenuItem } from "@/components/ContextMenu";

const FilePreview = lazy(() => import("@/components/FilePreview").then(m => ({ default: m.FilePreview })));

interface WorkspaceData {
  sourceDir: string | null;
  sourceFiles: FileNode[];
  sourceTrunc?: TreeTruncation | null;
  outputDir: string | null;
  outputFiles: FileNode[];
  outputTrunc?: TreeTruncation | null;
  loading: boolean;
  loadDeeper?: (which: "source" | "output") => void;
  deleteFile?: (path: string) => Promise<boolean>;
  refreshSource?: () => Promise<void>;
  refreshOutput?: () => Promise<void>;
}

interface Props {
  workspace: WorkspaceData;
}

// ─── File Tree Internals ───────────────────────────────────────────────

function FileIcon({ type, name }: { type: "file" | "folder"; name: string }) {
  if (type === "folder") {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="shrink-0 text-[#c4a44e]">
        <path d="M1.5 3.5a1 1 0 011-1h3l1 1h5a1 1 0 011 1v6a1 1 0 01-1 1h-9a1 1 0 01-1-1v-7z" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="1" />
      </svg>
    );
  }
  const ext = name.split(".").pop() ?? "";
  const color: Record<string, string> = {
    py: "#3572a5", yaml: "#cb171e", yml: "#cb171e", md: "#083fa1",
    log: "#6b7280", edf: "#059669", fif: "#059669", csv: "#16a34a",
    json: "#f59e0b", txt: "#6b7280", h5: "#7c3aed", hdf5: "#7c3aed",
  };
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="shrink-0" style={{ color: color[ext] ?? "var(--text-muted)" }}>
      <path d="M3.5 1.5h5l3 3v7a1 1 0 01-1 1h-7a1 1 0 01-1-1v-9a1 1 0 011-1z" stroke="currentColor" strokeWidth="1" />
      <path d="M8.5 1.5v3h3" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}

function formatSize(bytes?: number): string {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function computeStats(nodes: FileNode[]): { count: number; totalSize: number } {
  let count = 0;
  let totalSize = 0;
  function walk(ns: FileNode[]) {
    for (const n of ns) {
      if (n.type === "file") {
        count++;
        if (n.size) totalSize += n.size;
      }
      if (n.children) walk(n.children);
    }
  }
  walk(nodes);
  return { count, totalSize };
}

const PREVIEWABLE_EXTS = new Set(["py", "yaml", "yml", "json", "md", "txt", "log", "csv", "toml", "sh", "bash", "cfg"]);
const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp"]);

function isPreviewable(name: string): boolean {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return PREVIEWABLE_EXTS.has(ext) || IMAGE_EXTS.has(ext);
}

function FileTree({ nodes, depth = 0, onFileClick, autoCollapse = false, onContext }: { nodes: FileNode[]; depth?: number; onFileClick: (path: string) => void; autoCollapse?: boolean; onContext?: (e: React.MouseEvent, node: FileNode) => void }) {
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    if (autoCollapse) return new Set<string>();
    return new Set(nodes.filter((n) => n.type === "folder").map((n) => n.name));
  });

  // Auto-expand folders that appear in newly-arrived `nodes` (while preserving
  // the user's manual collapses). Adjusted during render via the prev-value
  // pattern rather than a state-setting effect.
  const [prevNodes, setPrevNodes] = useState(nodes);
  if (nodes !== prevNodes) {
    setPrevNodes(nodes);
    if (!autoCollapse) {
      setExpanded((prev) => {
        const folderNames = nodes.filter((n) => n.type === "folder").map((n) => n.name);
        const next = new Set(prev);
        for (const name of folderNames) next.add(name);
        return next;
      });
    }
  }

  const toggle = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="flex flex-col">
      {nodes.map((node) => (
        <div key={node.path || node.name}>
          <button
            onClick={() => {
              if (node.type === "folder") toggle(node.name);
              else if (isPreviewable(node.name)) onFileClick(node.path);
            }}
            onContextMenu={onContext ? (e) => { e.preventDefault(); e.stopPropagation(); onContext(e, node); } : undefined}
            className={`group w-full flex items-center gap-1.5 px-2 py-1 text-left rounded transition-all duration-150 ${
              node.type === "file" && isPreviewable(node.name)
                ? "hover:bg-[var(--bg-active)] hover:translate-x-[1px] cursor-pointer"
                : "hover:bg-[var(--bg-hover)]"
            }`}
            style={{ paddingLeft: `${depth * 12 + 8}px` }}
          >
            {node.type === "folder" && (
              <svg
                width="10" height="10" viewBox="0 0 10 10"
                className={`shrink-0 text-[var(--text-muted)] transition-transform duration-200 ${expanded.has(node.name) ? "rotate-90" : ""}`}
              >
                <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
            {node.type === "file" && <span className="w-[10px]" />}
            <FileIcon type={node.type} name={node.name} />
            <span className="text-[12px] text-[var(--text-primary)] truncate">{node.name}</span>
            {node.type === "file" && isPreviewable(node.name) && (
              <svg
                width="12" height="12" viewBox="0 0 12 12"
                className="ml-auto shrink-0 text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity duration-150"
              >
                <path d="M1 6s2-3.5 5-3.5S11 6 11 6s-2 3.5-5 3.5S1 6 1 6z" stroke="currentColor" strokeWidth="1" fill="none" />
                <circle cx="6" cy="6" r="1.5" stroke="currentColor" strokeWidth="1" fill="none" />
              </svg>
            )}
            {node.size != null && node.type === "file" && (
              <span className="ml-auto text-[10px] text-[var(--text-faint)] shrink-0">{formatSize(node.size)}</span>
            )}
          </button>
          {node.type === "folder" && node.children && (
            <div className={`grid transition-[grid-template-rows] duration-200 ease-out ${expanded.has(node.name) ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
              <div className="overflow-hidden">
                <FileTree nodes={node.children} depth={depth + 1} onFileClick={onFileClick} autoCollapse={autoCollapse} onContext={onContext} />
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DirectorySection({
  sectionKey,
  label,
  dirPath,
  files,
  onFileClick,
  trunc,
  onLoadDeeper,
  loading,
  onContext,
}: {
  /** Stable key for the per-section sessionStorage entry. "source" / "output". */
  sectionKey: string;
  label: string;
  dirPath: string;
  files: FileNode[];
  onFileClick: (path: string) => void;
  trunc?: TreeTruncation | null;
  onLoadDeeper?: () => void;
  loading?: boolean;
  onContext?: (e: React.MouseEvent, node: FileNode) => void;
}) {
  const stats = useMemo(() => computeStats(files), [files]);
  const shouldAutoCollapse = stats.count > 20;
  const showTrunc = !!trunc && (trunc.count > 0 || trunc.depth);
  const isEmpty = files.length === 0;

  // Per-section collapse — folds the path display, file tree, and truncation
  // hint together so the user can dismiss a whole directory pane. Persisted in
  // sessionStorage so the choice survives re-renders / reloads.
  const storageKey = `directory-section-expanded:${sectionKey}`;
  const [expanded, setExpanded] = useState<boolean>(() => {
    try {
      const raw = sessionStorage.getItem(storageKey);
      // Default open; only collapsed if the user explicitly said so.
      return raw === null ? true : raw === "1";
    } catch { return true; }
  });
  const toggle = useCallback(() => {
    setExpanded((prev) => {
      const next = !prev;
      try { sessionStorage.setItem(storageKey, next ? "1" : "0"); } catch { /* private mode */ }
      return next;
    });
  }, [storageKey]);

  return (
    <div className="px-3 mb-3 animate-fade-in">
      <button
        onClick={toggle}
        className="w-full flex items-center gap-1.5 mb-1.5 text-left rounded hover:bg-[var(--bg-hover)] px-1 py-0.5 transition-colors"
        title={expanded ? "Collapse section" : "Expand section"}
      >
        <svg
          width="10" height="10" viewBox="0 0 10 10"
          className={`shrink-0 text-[var(--text-muted)] transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
        >
          <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wide">{label}</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--bg-tertiary)] text-[var(--text-muted)] tabular-nums">
          {isEmpty ? "empty" : (
            <>
              {stats.count} file{stats.count !== 1 ? "s" : ""}
              {stats.totalSize > 0 && ` · ${formatSize(stats.totalSize)}`}
            </>
          )}
        </span>
        {loading && (
          <span
            aria-hidden
            className="w-3 h-3 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin ml-1"
          />
        )}
      </button>
      <div className={`grid transition-[grid-template-rows] duration-200 ease-out ${expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden">
          <div className="text-[11px] text-[var(--text-muted)] px-2 mb-1 font-mono truncate" title={dirPath}>
            {dirPath}
          </div>
          {isEmpty ? (
            <div className="px-2 py-1.5 text-[11px] text-[var(--text-faint)] italic">
              {loading ? "Reading directory…" : "Waiting for files — output will appear here as the pipeline writes results."}
            </div>
          ) : (
            <FileTree nodes={files} onFileClick={onFileClick} autoCollapse={shouldAutoCollapse} onContext={onContext} />
          )}
          {showTrunc && (
            <div className="flex items-center gap-2 px-2 mt-1.5 text-[10px] text-[var(--text-faint)]">
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" className="shrink-0">
                <path d="M6 3.5v3M6 8.5h.01M1.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0z" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
              </svg>
              <span>
                {trunc!.count > 0 && `+${trunc!.count} more not shown`}
                {trunc!.count > 0 && trunc!.depth && " · "}
                {trunc!.depth && "some folders are deeper than shown"}
              </span>
              {trunc!.depth && onLoadDeeper && (
                <button
                  onClick={onLoadDeeper}
                  className="ml-auto text-[var(--accent-blue)] hover:underline shrink-0"
                >
                  Load deeper
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────

export function WorkspacePanel({ workspace }: Props) {
  const { sourceDir, sourceFiles, sourceTrunc, outputDir, outputFiles, outputTrunc, loading, loadDeeper, deleteFile, refreshSource, refreshOutput } = workspace;
  const hasContent = sourceDir || outputDir;
  const [previewPath, setPreviewPath] = useState<string | null>(null);

  // Right-click context menu — used both by file-tree nodes (Copy / Preview /
  // Delete) and by empty-area clicks (Refresh).
  const [menu, setMenu] = useState<{ x: number; y: number; items: ContextMenuItem[] } | null>(null);

  // File-tree right-click → copy/preview, plus Delete on the Output side only
  // (Source is read-only, honoring source_data_guard's spirit).
  const openFileMenu = (side: "source" | "output") => (e: React.MouseEvent, node: FileNode) => {
    const items: ContextMenuItem[] = [
      { label: "Copy path", action: () => navigator.clipboard.writeText(node.path) },
      { label: "Open in preview", action: () => setPreviewPath(node.path) },
    ];
    if (side === "output" && deleteFile) {
      items.push({ label: "Delete", danger: true, action: () => { void deleteFile(node.path); } });
    }
    setMenu({ x: e.clientX, y: e.clientY, items });
  };

  // Right-click on empty space inside the Files area → single Refresh action.
  // FileTree nodes call stopPropagation(), so their own menu wins for files;
  // anything outside a row (path display, empty section, gaps) bubbles here.
  const openEmptyMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    const items: ContextMenuItem[] = [
      {
        label: "Refresh",
        action: () => {
          if (refreshSource) void refreshSource();
          if (refreshOutput) void refreshOutput();
        },
      },
    ];
    setMenu({ x: e.clientX, y: e.clientY, items });
  };

  return (
    <>
      <div className="flex items-center px-4 py-3 border-b border-[var(--border-primary)]">
        <h2 className="text-heading-sm text-[var(--text-primary)]">Workspace</h2>
        {loading && (
          <span className="ml-2 w-3 h-3 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin" />
        )}
      </div>

      <div className="flex-1 overflow-y-auto py-2" onContextMenu={openEmptyMenu}>
        {!hasContent && (
          <div className="flex flex-col items-center justify-center h-full px-4 text-center animate-fade-in">
            <div className="text-[var(--text-faint)] mb-3">
              <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
                <path d="M6 10a2 2 0 012-2h10l2 2h14a2 2 0 012 2v22a2 2 0 01-2 2H8a2 2 0 01-2-2V10z" stroke="currentColor" strokeWidth="1.5" fill="currentColor" fillOpacity="0.03" />
                <path d="M11 24h3l1.5-4 2.5 8 2-5 1.5 3 1.5-2.5 1 2H30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6" />
                <circle cx="13" cy="16" r="1" fill="currentColor" opacity="0.4" />
                <circle cx="18" cy="16" r="1" fill="currentColor" opacity="0.4" />
                <circle cx="23" cy="16" r="1" fill="currentColor" opacity="0.4" />
              </svg>
            </div>
            <p className="text-[12px] text-[var(--text-muted)] mb-1">No workspace files</p>
            <p className="text-[11px] text-[var(--text-faint)] max-w-[200px]">
              Run a processing pipeline to see source data and output files here
            </p>
          </div>
        )}

        {sourceDir && (
          <DirectorySection
            sectionKey="source"
            label="Source Data"
            dirPath={sourceDir}
            files={sourceFiles}
            onFileClick={setPreviewPath}
            trunc={sourceTrunc}
            onLoadDeeper={loadDeeper ? () => loadDeeper("source") : undefined}
            loading={loading && sourceFiles.length === 0}
            onContext={openFileMenu("source")}
          />
        )}

        {sourceDir && outputDir && (
          <div className="border-t border-[var(--border-primary)] mx-3 my-2" />
        )}

        {outputDir && (
          <DirectorySection
            sectionKey="output"
            label="Output"
            dirPath={outputDir}
            files={outputFiles}
            onFileClick={setPreviewPath}
            trunc={outputTrunc}
            onLoadDeeper={loadDeeper ? () => loadDeeper("output") : undefined}
            loading={loading && outputFiles.length === 0}
            onContext={openFileMenu("output")}
          />
        )}
      </div>

      {menu && (
        <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} />
      )}

      {previewPath && (
        <Suspense fallback={null}>
          <FilePreview filePath={previewPath} onClose={() => setPreviewPath(null)} />
        </Suspense>
      )}
    </>
  );
}
