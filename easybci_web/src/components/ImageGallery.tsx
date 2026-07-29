import { useState, useMemo, lazy, Suspense } from "react";
import type { Message } from "@/hooks/useConversation";
import { fileServeUrl } from "@/lib/api";

const FilePreview = lazy(() =>
  import("@/components/FilePreview").then((m) => ({ default: m.FilePreview })),
);

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp"]);
const IMAGE_PATH_RE = /(?:^|\s)(\/(?:[^\s/]+\/)*[^\s/]+\.(?:png|jpg|jpeg|gif|svg|webp))(?:\s|$|[,.);\]}])/gi;

function extractImagePaths(messages: Message[]): string[] {
  const seen = new Set<string>();
  const paths: string[] = [];

  for (const msg of messages) {
    if (msg.role !== "assistant") continue;

    if (msg.content) {
      let match: RegExpExecArray | null;
      const re = new RegExp(IMAGE_PATH_RE.source, IMAGE_PATH_RE.flags);
      while ((match = re.exec(msg.content)) !== null) {
        const p = match[1];
        if (!seen.has(p)) {
          seen.add(p);
          paths.push(p);
        }
      }
    }

    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        if (tc.preview) {
          const ext = tc.preview.split(".").pop()?.toLowerCase() ?? "";
          if (IMAGE_EXTS.has(ext) && !seen.has(tc.preview)) {
            seen.add(tc.preview);
            paths.push(tc.preview);
          }
        }
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
        <span className="text-[9px] text-white/90 truncate block font-mono">
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
      title="Download"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
        <path d="M6 1v8M6 9L3 6M6 9l3-3M2 11h8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </a>
  );
}

interface Props {
  messages: Message[];
}

export function ImageGallery({ messages }: Props) {
  const images = useMemo(() => extractImagePaths(messages), [messages]);
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
              Images in Conversation
            </span>
            <span className="text-[10px] text-[var(--text-faint)]">
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
