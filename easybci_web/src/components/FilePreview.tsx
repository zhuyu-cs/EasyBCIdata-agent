import { useEffect, useState } from "react";
import { fetchJSON, fileServeUrl } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  filePath: string;
  onClose: () => void;
  // Optional sibling navigation: when the preview is opened from a gallery,
  // these let the user page through related files with ←/→ (U7).
  siblings?: string[];
  onNavigate?: (path: string) => void;
}

interface FileContent {
  content: string;
  size: number;
  is_text?: boolean;
  mime_type?: string;
  truncated?: boolean;
}

/** Bytes shown by the backend before truncation (mirror of the 50KB cap in
 *  read_file_content). Used to phrase the truncation notice (B11). */
const PREVIEW_LIMIT_BYTES = 50 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp"]);

function isImageFile(path: string): boolean {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTS.has(ext);
}

function getLanguage(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const langMap: Record<string, string> = {
    py: "python", yaml: "yaml", yml: "yaml", json: "json",
    md: "markdown", sh: "bash", bash: "bash", ts: "typescript",
    tsx: "typescript", js: "javascript", css: "css", html: "html",
    log: "text", txt: "text", csv: "text", toml: "toml",
  };
  return langMap[ext] ?? "text";
}

function isMarkdown(path: string): boolean {
  return path.endsWith(".md");
}

interface NavProps {
  index: number;
  total: number;
  onPrev?: () => void;
  onNext?: () => void;
}

// Shared close + sibling navigation keyboard handling for both preview modes.
function usePreviewKeys(onClose: () => void, nav?: NavProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); return; }
      if (!nav) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); nav.onPrev?.(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); nav.onNext?.(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose, nav]);
}

function NavArrows({ nav, light }: { nav: NavProps; light?: boolean }) {
  if (nav.total <= 1) return null;
  const base = light
    ? "bg-black/50 text-white hover:bg-black/70"
    : "bg-[var(--bg-secondary)] border border-[var(--border-primary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]";
  return (
    <>
      <button
        onClick={(e) => { e.stopPropagation(); nav.onPrev?.(); }}
        disabled={!nav.onPrev}
        className={`absolute left-3 top-1/2 -translate-y-1/2 z-10 w-9 h-9 flex items-center justify-center rounded-full transition-colors disabled:opacity-30 ${base}`}
        title="Previous (←)"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <button
        onClick={(e) => { e.stopPropagation(); nav.onNext?.(); }}
        disabled={!nav.onNext}
        className={`absolute right-3 top-1/2 -translate-y-1/2 z-10 w-9 h-9 flex items-center justify-center rounded-full transition-colors disabled:opacity-30 ${base}`}
        title="Next (→)"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </>
  );
}

function ImagePreview({ filePath, onClose, nav }: { filePath: string; onClose: () => void; nav?: NavProps }) {
  const [zoom, setZoom] = useState(1);
  const fileName = filePath.split("/").pop() ?? filePath;
  const src = fileServeUrl(filePath);

  usePreviewKeys(onClose, nav);

  // Reset zoom when the previewed file changes — adjusted during render via the
  // prev-prop pattern instead of a state-setting effect.
  const [prevPath, setPrevPath] = useState(filePath);
  if (filePath !== prevPath) {
    setPrevPath(filePath);
    setZoom(1);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 animate-backdrop-in" />
      {nav && <NavArrows nav={nav} light />}
      <div
        className="relative flex flex-col items-center max-w-[92vw] max-h-[90vh] animate-dialog-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Controls */}
        <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors text-[16px]"
          >
            -
          </button>
          <span className="text-[12px] text-white/80 min-w-[40px] text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors text-[16px]"
          >
            +
          </button>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors ml-2"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Image */}
        <div className="overflow-auto max-w-[92vw] max-h-[82vh] rounded-lg">
          <img
            src={src}
            alt={fileName}
            className="transition-transform duration-200 rounded-lg shadow-2xl"
            style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
            draggable={false}
          />
        </div>

        {/* Filename + position */}
        <div className="mt-3 px-4 py-1.5 bg-black/50 rounded-full flex items-center gap-2">
          <span className="text-[12px] text-white/90 font-mono">{fileName}</span>
          {nav && nav.total > 1 && (
            <span className="text-[11px] text-white/60 tabular-nums">{nav.index + 1} / {nav.total}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function FilePreview({ filePath, onClose, siblings, onNavigate }: Props) {
  // Derive prev/next navigation from the sibling list, if provided.
  let nav: NavProps | undefined;
  if (siblings && siblings.length > 1 && onNavigate) {
    const index = siblings.indexOf(filePath);
    if (index >= 0) {
      nav = {
        index,
        total: siblings.length,
        onPrev: index > 0 ? () => onNavigate(siblings[index - 1]) : undefined,
        onNext: index < siblings.length - 1 ? () => onNavigate(siblings[index + 1]) : undefined,
      };
    }
  }

  if (isImageFile(filePath)) {
    return <ImagePreview filePath={filePath} onClose={onClose} nav={nav} />;
  }

  return <TextFilePreview filePath={filePath} onClose={onClose} nav={nav} />;
}

function TextFilePreview({ filePath, onClose, nav }: { filePath: string; onClose: () => void; nav?: NavProps }) {
  // A single keyed result so content/error/loading are always consistent with
  // the current filePath — derived during render, no setState-in-effect to reset
  // on path change.
  const [result, setResult] = useState<{
    path: string;
    content: string | null;
    error: string | null;
    size?: number;
    truncated?: boolean;
    isText?: boolean;
  }>({ path: filePath, content: null, error: null });
  const loading = result.path !== filePath || (result.content === null && result.error === null && result.isText !== false);

  const fileName = filePath.split("/").pop() ?? filePath;

  usePreviewKeys(onClose, nav);

  useEffect(() => {
    let cancelled = false;
    fetchJSON<FileContent>(`/api/files/read?path=${encodeURIComponent(filePath)}`)
      .then((data) => {
        if (!cancelled)
          setResult({
            path: filePath,
            content: data.content,
            error: null,
            size: data.size,
            truncated: data.truncated,
            isText: data.is_text,
          });
      })
      .catch((e) => {
        if (!cancelled) setResult({ path: filePath, content: null, error: e instanceof Error ? e.message : "Failed to read file" });
      });
    return () => { cancelled = true; };
  }, [filePath]);

  const current = result.path === filePath;
  const content = current ? result.content : null;
  const error = current ? result.error : null;
  const size = current ? result.size : undefined;
  const truncated = current ? result.truncated : false;
  const isBinary = current && result.isText === false && result.content === null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/40 animate-backdrop-in" />
      {nav && <NavArrows nav={nav} />}
      <div
        className="relative w-[90vw] max-w-3xl max-h-[80vh] bg-[var(--bg-secondary)] rounded-lg shadow-xl flex flex-col overflow-hidden animate-dialog-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-primary)] shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-[13px] font-medium text-[var(--text-primary)] truncate">{fileName}</span>
            <span className="text-[11px] text-[var(--text-muted)] shrink-0">{getLanguage(filePath)}</span>
            {nav && nav.total > 1 && (
              <span className="text-[11px] text-[var(--text-faint)] shrink-0 tabular-nums">{nav.index + 1} / {nav.total}</span>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Truncation notice — the backend caps preview at 50KB (B11). */}
        {truncated && size != null && (
          <div className="flex items-center gap-2 px-4 py-1.5 bg-[var(--bg-tertiary)] border-b border-[var(--border-primary)] shrink-0 text-[11px] text-[var(--text-muted)]">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="shrink-0">
              <path d="M6 3.5v3M6 8.5h.01M1.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0z" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
            </svg>
            <span>
              Showing first {formatBytes(PREVIEW_LIMIT_BYTES)} of {formatBytes(size)}
            </span>
            <span className="ml-auto text-[var(--text-faint)] font-mono truncate" title={`Open ${filePath} on this machine`}>
              open locally: {filePath}
            </span>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {loading && (
            <div className="flex items-center gap-2 text-[12px] text-[var(--text-muted)]">
              <span className="w-3 h-3 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin" />
              Loading...
            </div>
          )}

          {error && (
            <div className="text-[12px] text-[var(--text-error)]">{error}</div>
          )}

          {isBinary && !loading && !error && (
            <div className="flex flex-col items-center justify-center gap-2 py-8 text-center">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none" className="text-[var(--text-faint)]">
                <path d="M8 4h11l5 5v19a1 1 0 01-1 1H8a1 1 0 01-1-1V5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.3" />
                <path d="M19 4v5h5" stroke="currentColor" strokeWidth="1.3" />
              </svg>
              <p className="text-[12px] text-[var(--text-muted)]">
                {size != null && size > 100 * 1024
                  ? `File is ${formatBytes(size)} — too large to preview inline.`
                  : "This file isn't a previewable text type."}
              </p>
              <p className="text-[11px] text-[var(--text-faint)] font-mono break-all max-w-[80%]">
                open locally: {filePath}
              </p>
            </div>
          )}

          {content !== null && !loading && (
            isMarkdown(filePath) ? (
              <div className="prose-compact text-[13px] leading-relaxed text-[var(--text-primary)]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              </div>
            ) : (
              <pre className="text-[12px] leading-[1.7] font-mono text-[var(--text-primary)] whitespace-pre-wrap break-words">
                {content}
              </pre>
            )
          )}
        </div>

        {/* Footer path */}
        <div className="px-4 py-2 border-t border-[var(--border-primary)] shrink-0">
          <span className="text-[11px] text-[var(--text-muted)] font-mono truncate block">{filePath}</span>
        </div>
      </div>
    </div>
  );
}
