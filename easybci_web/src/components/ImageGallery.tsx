import { useState, useMemo, lazy, Suspense, memo } from "react";
import type { Turn } from "@/lib/turns";
import { fileServeUrl } from "@/lib/api";
import { t } from "@/lib/i18n";

const FilePreview = lazy(() =>
  import("@/components/FilePreview").then((m) => ({ default: m.FilePreview })),
);

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp"]);
const IMAGE_PATH_RE = /(?:^|\s)(\/(?:[^\s/]+\/)*[^\s/]+\.(?:png|jpg|jpeg|gif|svg|webp))(?:\s|$|[,.);\]}])/gi;

function extractImagePaths(turns: Turn[]): string[] {
  const seen = new Set<string>();
  const paths: string[] = [];
  const add = (p: string) => {
    if (!seen.has(p)) {
      seen.add(p);
      paths.push(p);
    }
  };

  for (const turn of turns) {
    for (const item of turn.items) {
      if (item.kind === "text") {
        const re = new RegExp(IMAGE_PATH_RE.source, IMAGE_PATH_RE.flags);
        let match: RegExpExecArray | null;
        while ((match = re.exec(item.text)) !== null) add(match[1]);
      } else if (item.kind === "step" && item.step.preview) {
        const ext = item.step.preview.split(".").pop()?.toLowerCase() ?? "";
        if (IMAGE_EXTS.has(ext)) add(item.step.preview);
      }
    }
  }
  return paths;
}

function Thumbnail({
  path,
  onClick,
}: {
  path: string;
  onClick: () => void;
}) {
  const fileName = path.split("/").pop() ?? path;
  const src = fileServeUrl(path);

  return (
    <button
      onClick={onClick}
      className="group relative aspect-square rounded-md overflow-hidden border border-[var(--border-primary)] hover:border-[var(--accent-blue)] transition-all duration-150 hover:shadow-md"
      title={fileName}
    >
      <img
        src={src}
        alt={fileName}
        className="w-full h-full object-cover"
        loading="lazy"
      />
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-150 flex items-center justify-center">
        <svg
          width="20"
          height="20"
          viewBox="0 0 20 20"
          fill="none"
          className="text-white opacity-0 group-hover:opacity-100 transition-opacity duration-150 drop-shadow-lg"
        >
          <path
            d="M3 10s3-5 7-5 7 5 7 5-3 5-7 5-7-5-7-5z"
            stroke="currentColor"
            strokeWidth="1.5"
            fill="none"
          />
          <circle cx="10" cy="10" r="2.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </div>
      <div className="absolute bottom-0 inset-x-0 px-1.5 py-1 bg-gradient-to-t from-black/50 to-transparent">
        <span className="text-[11px] text-white/90 truncate block font-mono">
          {fileName}
        </span>
      </div>
    </button>
  );
}

function DownloadButton({ path }: { path: string }) {
  const src = fileServeUrl(path);
  const fileName = path.split("/").pop() ?? "image";

  return (
    <a
      href={src}
      download={fileName}
      className="w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors"
      title={t("Download")}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
        <path d="M6 1v8M6 9L3 6M6 9l3-3M2 11h8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </a>
  );
}

interface Props {
  turns: Turn[];
  /** Skip the trailing turn — it is still streaming, and its paths may be
   *  half-written. */
  excludeLast?: boolean;
}

function ImageGalleryImpl({ turns, excludeLast = false }: Props) {
  const images = useMemo(
    () => extractImagePaths(excludeLast ? turns.slice(0, -1) : turns),
    [turns, excludeLast],
  );
  const [previewPath, setPreviewPath] = useState<string | null>(null);

  if (images.length === 0) return null;

  return (
    <>
      <div className="rounded-lg border border-[var(--border-primary)] overflow-hidden animate-fade-in">
        <div className="px-3 py-1.5 bg-[var(--bg-tertiary)] border-b border-[var(--border-primary)] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-[var(--text-muted)]">
              <rect x="1.5" y="2.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" />
              <circle cx="4.5" cy="5.5" r="1" stroke="currentColor" strokeWidth="0.8" />
              <path d="M1.5 9.5l3-3 2 2 3-3 3 3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-[11px] font-medium text-[var(--text-muted)]">
              {t("Images in Conversation")}
            </span>
            <span className="text-[11px] text-[var(--text-faint)]">
              {images.length}
            </span>
          </div>
          {images.length === 1 && <DownloadButton path={images[0]} />}
        </div>

        <div
          className="p-2 grid gap-2"
          style={{
            gridTemplateColumns: `repeat(${Math.min(images.length, 4)}, 1fr)`,
          }}
        >
          {images.map((path) => (
            <Thumbnail
              key={path}
              path={path}
              onClick={() => setPreviewPath(path)}
            />
          ))}
        </div>

        {images.length > 1 && (
          <div className="px-3 py-1.5 border-t border-[var(--border-primary)] flex items-center gap-1">
            {images.map((path) => (
              <DownloadButton key={path} path={path} />
            ))}
          </div>
        )}
      </div>

      {previewPath && (
        <Suspense fallback={null}>
          <FilePreview
            filePath={previewPath}
            siblings={images}
            onNavigate={setPreviewPath}
            onClose={() => setPreviewPath(null)}
          />
        </Suspense>
      )}
    </>
  );
}

// Streaming replaces the turns array on every flush, but only its last element
// changes; the gallery re-renders only when a settled turn does.
function settledEqual(a: Props, b: Props): boolean {
  if (a.excludeLast !== b.excludeLast) return false;
  const n = a.excludeLast ? a.turns.length - 1 : a.turns.length;
  const m = b.excludeLast ? b.turns.length - 1 : b.turns.length;
  if (n !== m) return false;
  for (let i = 0; i < n; i++) if (a.turns[i] !== b.turns[i]) return false;
  return true;
}

export const ImageGallery = memo(ImageGalleryImpl, settledEqual);
