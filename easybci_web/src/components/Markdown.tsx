import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, useCallback, useMemo, lazy, Suspense, memo } from "react";
import { fileServeUrl } from "@/lib/api";
import { useHighlighter, isSupportedLang } from "@/hooks/useHighlighter";
import { useThemeStore } from "@/stores/themeStore";
import { splitMarkdownBlocks } from "@/lib/markdownBlocks";
import { isPipelineYaml } from "./pipelineYaml";
import { isProposalJson } from "./proposalJson";
import { tryParseQCMetrics } from "./qcMetrics";
import { t } from "@/lib/i18n";

const QCCard = lazy(() => import("./QCCard").then(m => ({ default: m.QCCard })));
const PipelineYamlCard = lazy(() => import("./PipelineYamlCard").then(m => ({ default: m.PipelineYamlCard })));
const ProposalCard = lazy(() => import("./ProposalCard").then(m => ({ default: m.ProposalCard })));

/** `live`: the fence is still streaming — render plain text and skip shiki
 *  until it closes, so the block never flickers between the two. */
function CodeBlockWrapper({ children, live = false }: { children: React.ReactNode; live?: boolean }) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const isDark = useThemeStore((s) => s.isDark);

  const code = extractText(children);
  const lang = codeLang(children);

  // Long code blocks collapse to a fixed height so a single snippet can't push
  // the whole conversation off-screen (U6). >30 lines is the trigger.
  const lineCount = code.split("\n").length;
  const collapsible = lineCount > 30;
  const isCollapsed = collapsible && !expanded;

  const highlightedHtml = useHighlighter(code, lang, isDark, !live);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  const bodyStyle = isCollapsed ? { maxHeight: 200, overflow: "hidden" as const } : undefined;

  return (
    <div className="group/code relative my-2 rounded-md border overflow-hidden" style={{ background: "var(--bg-code)", borderColor: "var(--border-code)" }}>
      <div className="flex items-center justify-between px-3 py-1 border-b" style={{ borderColor: "var(--border-code)", background: "var(--bg-code-header)" }}>
        <span className="text-[11px] font-mono" style={{ color: "var(--text-muted)" }}>
          {lang || "code"}
          {collapsible && <span className="ml-1.5 opacity-70 tabular-nums">{t("{n} lines", { n: lineCount })}</span>}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[11px] opacity-0 group-hover/code:opacity-100 transition-opacity duration-150"
          style={{ color: "var(--text-muted)" }}
        >
          {copied ? (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2.5 6.5l2 2 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {t("Copied")}
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <rect x="4" y="4" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1" />
                <path d="M8 4V3a1 1 0 00-1-1H3a1 1 0 00-1 1v4a1 1 0 001 1h1" stroke="currentColor" strokeWidth="1" />
              </svg>
              {t("Copy")}
            </>
          )}
        </button>
      </div>
      <div className="relative">
        {highlightedHtml && isSupportedLang(lang) ? (
          <div
            className="p-3 overflow-x-auto text-[12px] leading-[1.6] font-mono [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!m-0"
            style={bodyStyle}
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <pre className="p-3 overflow-x-auto text-[12px] leading-[1.6] font-mono" style={bodyStyle}>
            {children}
          </pre>
        )}
        {isCollapsed && (
          <div
            className="absolute bottom-0 left-0 right-0 h-12 pointer-events-none"
            style={{ background: "linear-gradient(transparent, var(--bg-code))" }}
          />
        )}
      </div>
      {collapsible && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full text-center py-1 text-[11px] font-medium border-t transition-colors hover:bg-[var(--bg-code-header)]"
          style={{ color: "var(--text-muted)", borderColor: "var(--border-code)" }}
        >
          {expanded ? t("Show less") : t("Show full ({n} lines)", { n: lineCount })}
        </button>
      )}
    </div>
  );
}

/** Language of a fenced block from the `<code>` child react-markdown hands to
 *  `pre` — read from its `language-x` class (the element's own props, not
 *  the attributes our `code` override renders). */
function codeLang(children: React.ReactNode): string {
  const el = (Array.isArray(children) ? children[0] : children) as React.ReactElement<{ className?: string }> | null;
  const cls = el && typeof el === "object" && "props" in el ? el.props.className : undefined;
  const m = cls?.match(/language-([\w+-]+)/);
  return m ? m[1].toLowerCase() : "";
}

function extractText(node: React.ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object" && "props" in node) {
    const el = node as React.ReactElement<{ children?: React.ReactNode }>;
    return extractText(el.props.children);
  }
  return "";
}

// Shared react-markdown component overrides — defined once at module scope so
// the config object is stable across renders.
const IMAGE_URL_RE = /^(https?:|data:|blob:)/i;

function MarkdownImage({ src, alt, title }: { src?: string; alt?: string; title?: string }) {
  const [failed, setFailed] = useState(false);
  const raw = (src ?? "").trim();
  if (!raw) return null;

  const isExternal = IMAGE_URL_RE.test(raw);
  const resolved = isExternal ? raw : fileServeUrl(raw);
  const fileName = raw.split(/[\\/]/).pop() ?? raw;

  if (failed) {
    return (
      <span
        className="inline-flex items-center gap-1.5 my-1 px-2 py-1 rounded-md border text-[11px] font-mono"
        style={{
          borderColor: "var(--border-primary)",
          background: "var(--bg-code)",
          color: "var(--text-muted)",
        }}
        title={t("Image not available: {f}", { f: raw })}
      >
        <svg width="11" height="11" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <rect x="1.5" y="2.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" />
          <path d="M2 11l3-3 2 2 3-3 3 3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="2" y1="2" x2="12" y2="12" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
        <span>{t("image unavailable: {f}", { f: fileName })}</span>
      </span>
    );
  }

  return (
    <img
      src={resolved}
      alt={alt ?? fileName}
      title={title ?? fileName}
      loading="lazy"
      onError={() => setFailed(true)}
      className="my-2 max-w-full h-auto rounded-md border"
      style={{ maxHeight: 480, objectFit: "contain", borderColor: "var(--border-primary)" }}
    />
  );
}

const MARKDOWN_COMPONENTS = {
  // Cards are only recognised once their fence has closed (see
  // MARKDOWN_COMPONENTS_LIVE): partial JSON/YAML can't be parsed anyway.
  pre: ({ children }: { children?: React.ReactNode }) => {
    const text = extractText(children);
    const lang = codeLang(children);

    if (lang === "json") {
      const qc = tryParseQCMetrics(text);
      if (qc) return <Suspense fallback={null}><QCCard metrics={qc} /></Suspense>;
      const proposal = isProposalJson(text);
      if (proposal) return <Suspense fallback={null}><ProposalCard proposal={proposal} /></Suspense>;
    }

    if (lang === "yaml" && isPipelineYaml(text)) {
      return <Suspense fallback={null}><PipelineYamlCard yaml={text} /></Suspense>;
    }

    return <CodeBlockWrapper>{children}</CodeBlockWrapper>;
  },
  // `node` (the mdast node) must not reach the DOM as an attribute.
  code: ({ className, children }: { className?: string; children?: React.ReactNode; node?: unknown }) => {
    if (className?.includes("language-")) return <code className={className}>{children}</code>;
    return (
      <code className="px-1 py-0.5 rounded text-[12px] font-mono" style={{ background: "var(--bg-tertiary)" }}>
        {children}
      </code>
    );
  },
  p: ({ children }: { children?: React.ReactNode }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }: { children?: React.ReactNode }) => <ul className="mb-2 pl-4 list-disc">{children}</ul>,
  ol: ({ children }: { children?: React.ReactNode }) => <ol className="mb-2 pl-4 list-decimal">{children}</ol>,
  li: ({ children }: { children?: React.ReactNode }) => <li className="mb-0.5">{children}</li>,
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="my-2 overflow-x-auto">
      <table className="text-[12px] border-collapse border" style={{ borderColor: "var(--border-secondary)" }}>{children}</table>
    </div>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="border px-2 py-1 font-medium text-left" style={{ borderColor: "var(--border-secondary)", background: "var(--bg-tertiary)" }}>{children}</th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="border px-2 py-1" style={{ borderColor: "var(--border-secondary)" }}>{children}</td>
  ),
  img: ({ src, alt, title }: { src?: string; alt?: string; title?: string }) => (
    <MarkdownImage src={src} alt={alt} title={title} />
  ),
};

const MARKDOWN_COMPONENTS_LIVE = {
  ...MARKDOWN_COMPONENTS,
  pre: ({ children }: { children?: React.ReactNode }) => <CodeBlockWrapper live>{children}</CodeBlockWrapper>,
};

const REMARK_PLUGINS = [remarkGfm];

const MarkdownBlock = memo(function MarkdownBlock({ text, closed }: { text: string; closed: boolean }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={closed ? MARKDOWN_COMPONENTS : MARKDOWN_COMPONENTS_LIVE}>
      {text}
    </ReactMarkdown>
  );
});

// Rendered block by block (lib/markdownBlocks): while streaming only the
// trailing block re-parses; every finished block keeps its key and text and
// is skipped by memo. History renders through the same path, so a reloaded
// answer looks exactly like it did live.
export const MarkdownContent = memo(function MarkdownContent({ content, streaming }: { content: string; streaming: boolean }) {
  const blocks = useMemo(() => splitMarkdownBlocks(content, !streaming), [content, streaming]);
  return (
    <>
      {blocks.map((b) => (
        <MarkdownBlock key={b.key} text={b.text} closed={b.closed} />
      ))}
    </>
  );
});
