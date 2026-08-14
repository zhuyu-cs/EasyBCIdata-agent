import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, useCallback, lazy, Suspense, memo } from "react";
import type { Message, ToolCall } from "@/hooks/useConversation";
import { fileServeUrl } from "@/lib/api";
import { useHighlighter, isSupportedLang } from "@/hooks/useHighlighter";
import { useThemeStore } from "@/stores/themeStore";
import { ThinkingBlock } from "./ThinkingBlock";
import { StepCard, type RunStatus } from "./StepCard";
import type { StageProgress } from "./StageProgressBar";
import { isPipelineYaml } from "./pipelineYaml";
import { tryParseQCMetrics } from "./qcMetrics";
import { LogoMark } from "./Logo";
import { ContextMenu, type ContextMenuItem } from "./ContextMenu";
import { MessageToolbar } from "./MessageToolbar";

const PipelineTimeline = lazy(() => import("./PipelineTimeline").then(m => ({ default: m.PipelineTimeline })));
const QCCard = lazy(() => import("./QCCard").then(m => ({ default: m.QCCard })));
const PipelineYamlCard = lazy(() => import("./PipelineYamlCard").then(m => ({ default: m.PipelineYamlCard })));
const ProposalCard = lazy(() => import("./ProposalCard").then(m => ({ default: m.ProposalCard })));
import { isProposalJson } from "./ProposalCard";

const BOOKMARK_KEY = "easybci-bookmarks";

function loadBookmarks(): Set<string> {
  try {
    const raw = localStorage.getItem(BOOKMARK_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function persistBookmark(id: string, on: boolean) {
  try {
    const set = loadBookmarks();
    if (on) set.add(id);
    else set.delete(id);
    localStorage.setItem(BOOKMARK_KEY, JSON.stringify([...set]));
  } catch {
    /* ignore */
  }
}

// Messages longer than this can be collapsed from the hover toolbar.
const COLLAPSE_THRESHOLD = 900;

interface Props {
  message: Message;
  onResend?: (text: string) => void;
  streaming?: boolean;
  /** Run lifecycle status for false-fail filtering — forwarded to StepCard /
   *  PipelineTimeline so only a genuinely-failed run shows terminal red. */
  runStatus?: RunStatus;
  /** Live StageProgress payload (T1.6) — only populated for the currently-
   *  streaming bubble; rendered as a four-stage progress ribbon at the top
   *  of the StepCard / PipelineTimeline. */
  progress?: StageProgress | null;
}

function UserAvatar() {
  return (
    <div
      className="w-6 h-6 rounded-full flex items-center justify-center shrink-0"
      style={{
        background:
          "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)",
        boxShadow:
          "0 1px 3px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.18)",
      }}
      aria-label="You"
      role="img"
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="#ffffff"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    </div>
  );
}

function AgentAvatar() {
  return (
    <div className="w-6 h-6 rounded-full border flex items-center justify-center shrink-0" style={{ background: "var(--bg-user-bubble)", borderColor: "var(--border-secondary)" }}>
      <LogoMark size={14} className="text-[var(--text-primary)]" />
    </div>
  );
}

function CodeBlockWrapper({ children }: { children: React.ReactNode }) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  const code = extractText(children);
  const childEl = children as React.ReactElement<{ "data-lang"?: string }> | null;
  const lang = childEl?.props?.["data-lang"] ?? "";

  // Long code blocks collapse to a fixed height so a single snippet can't push
  // the whole conversation off-screen (U6). >30 lines is the trigger.
  const lineCount = code.split("\n").length;
  const collapsible = lineCount > 30;
  const isCollapsed = collapsible && !expanded;

  const highlightedHtml = useHighlighter(code, lang, isDark);

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
        <span className="text-[10px] font-mono" style={{ color: "var(--text-muted)" }}>
          {lang || "code"}
          {collapsible && <span className="ml-1.5 opacity-70 tabular-nums">{lineCount} lines</span>}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[10px] opacity-0 group-hover/code:opacity-100 transition-opacity duration-150"
          style={{ color: "var(--text-muted)" }}
        >
          {copied ? (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2.5 6.5l2 2 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <rect x="4" y="4" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1" />
                <path d="M8 4V3a1 1 0 00-1-1H3a1 1 0 00-1 1v4a1 1 0 001 1h1" stroke="currentColor" strokeWidth="1" />
              </svg>
              Copy
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
          className="w-full text-center py-1 text-[10px] font-medium border-t transition-colors hover:bg-[var(--bg-code-header)]"
          style={{ color: "var(--text-muted)", borderColor: "var(--border-code)" }}
        >
          {expanded ? "Show less" : `Show full (${lineCount} lines)`}
        </button>
      )}
    </div>
  );
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
        title={`Image not available: ${raw}`}
      >
        <svg width="11" height="11" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <rect x="1.5" y="2.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.1" />
          <path d="M2 11l3-3 2 2 3-3 3 3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="2" y1="2" x2="12" y2="12" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
        <span>image unavailable: {fileName}</span>
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
  pre: ({ children }: { children?: React.ReactNode }) => {
    const text = extractText(children);
    const childEl = children as React.ReactElement<{ "data-lang"?: string }> | null;
    const lang = childEl?.props?.["data-lang"] ?? "";

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
  code: ({ className, children, ...props }: { className?: string; children?: React.ReactNode }) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      const lang = className?.replace("language-", "") ?? "";
      return <code data-lang={lang} {...props}>{children}</code>;
    }
    return (
      <code className="px-1 py-0.5 rounded text-[12px] font-mono" style={{ background: "var(--bg-tertiary)" }} {...props}>
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

const REMARK_PLUGINS = [remarkGfm];

// Memoized markdown renderer. react-markdown re-parses on every render, which is
// wasteful when a virtualized list mounts/unmounts history messages repeatedly.
// Memoizing on the content string skips re-parse for unchanged, settled
// messages (U11). Streaming bubbles change content each frame, so memoization is
// effectively a no-op there — correct, not a regression.
const MarkdownContent = memo(function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={MARKDOWN_COMPONENTS}>
      {content}
    </ReactMarkdown>
  );
});

function MessageBubbleImpl({ message, onResend, streaming, runStatus = "unknown", progress = null }: Props) {
  const isUser = message.role === "user";
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [copied, setCopied] = useState(false);
  const [bookmarked, setBookmarked] = useState(() => loadBookmarks().has(message.id));
  const [collapsed, setCollapsed] = useState(false);



  const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;
  const hasOrphanThinking = message.thinking && !hasToolCalls;
  // Reloaded history: an assistant message can carry top-level `thinking` AND
  // tool calls. During live streaming that reasoning threads into each step's
  // per-call reasoning, but on reload there is no per-step reasoning to thread
  // into, so the thinking would vanish (only copyable via context menu).
  // Surface it as a block above the steps when we're NOT streaming.
  const hasReloadedThinkingWithTools = !streaming && message.thinking && hasToolCalls;
  const collapsible = !streaming && (message.content?.length ?? 0) > COLLAPSE_THRESHOLD;

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content || "").then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [message.content]);

  const toggleBookmark = useCallback(() => {
    setBookmarked((prev) => {
      const next = !prev;
      persistBookmark(message.id, next);
      return next;
    });
  }, [message.id]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const contextMenuItems: ContextMenuItem[] = [
    {
      label: "Copy text",
      action: () => {
        navigator.clipboard.writeText(message.content || "");
      },
    },
    {
      label: "Copy as Markdown",
      action: () => {
        const parts: string[] = [];
        if (message.thinking) parts.push(`<details><summary>Thinking</summary>\n\n${message.thinking}\n\n</details>\n`);
        if (message.content) parts.push(message.content);
        navigator.clipboard.writeText(parts.join("\n"));
      },
    },
  ];

  if (isUser && onResend) {
    contextMenuItems.push({
      label: "Re-send",
      action: () => onResend(message.content),
    });
  }

  contextMenuItems.push({
    label: bookmarked ? "Remove bookmark" : "Bookmark",
    action: toggleBookmark,
  });

  return (
    <div
      className={`flex gap-2.5 group/msg ${isUser ? "flex-row-reverse" : "flex-row"} ${message.isNew ? "animate-fade-in" : ""}`}
      onContextMenu={handleContextMenu}
    >
      {isUser ? <UserAvatar /> : <AgentAvatar />}
      <div className={`flex flex-col ${isUser ? "items-end" : "items-start"} min-w-0 flex-1 max-w-[85%]`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>
            {isUser ? "You" : "EasyBCI Agent"}
          </span>
          {/* Reasoning phase: thinking but no visible content yet (U12) */}
          {streaming && !message.content && (
            <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--text-faint)" }}>
              <span className="w-[5px] h-[5px] rounded-full reasoning-dot" style={{ background: "var(--accent-green)" }} />
              reasoning
            </span>
          )}
          <span className="text-[11px]" style={{ color: "var(--text-faint)" }}>
            {new Date(message.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
          <MessageToolbar
            onCopy={handleCopy}
            copied={copied}
            onResend={isUser && onResend ? () => onResend(message.content) : undefined}
            collapsible={collapsible}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((c) => !c)}
          />
        </div>

      {hasOrphanThinking && <ThinkingBlock thinking={message.thinking!} />}
      {hasReloadedThinkingWithTools && <ThinkingBlock thinking={message.thinking!} />}

      {hasToolCalls && (message.toolCalls!.length >= 5
        ? <Suspense fallback={null}><PipelineTimeline toolCalls={message.toolCalls!} runStatus={runStatus} messageId={message.id} progress={progress} /></Suspense>
        : <StepCard toolCalls={message.toolCalls!} runStatus={runStatus} messageId={message.id} progress={progress} />
      )}

      {message.content && (
        <div className="relative w-full">
          <div
            className={`px-4 py-3 text-[13.5px] leading-relaxed ${
              isUser
                ? ""
                : `border rounded-lg ${streaming ? "animate-accent-pulse" : ""}`
            }`}
            style={{
              ...(isUser
                ? { background: "var(--bg-user-bubble)", color: "var(--text-primary)", borderRadius: "16px 16px 4px 16px" }
                : {
                    background: "var(--bg-agent-bubble)",
                    color: "var(--text-primary)",
                    borderColor: "var(--border-primary)",
                    borderLeftWidth: "2px",
                    borderLeftColor: "var(--accent-green)",
                    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.03)",
                  }),
              ...(collapsed ? { maxHeight: 200, overflow: "hidden" } : {}),
            }}
          >
            {isUser ? (
              <span className="whitespace-pre-wrap">{message.content}</span>
            ) : (
              <div className="prose-compact">
                <MarkdownContent content={message.content} />
              </div>
            )}
          </div>
          {collapsed && (
            <button
              onClick={() => setCollapsed(false)}
              className="absolute inset-x-0 bottom-0 flex items-end justify-center pb-1.5 pt-8 text-[11px] font-medium rounded-b-lg"
              style={{
                color: "var(--text-secondary)",
                background: `linear-gradient(to bottom, transparent, ${isUser ? "var(--bg-user-bubble)" : "var(--bg-agent-bubble)"} 70%)`,
              }}
            >
              Show more
            </button>
          )}
        </div>
      )}
      </div>

      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          items={contextMenuItems}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  );
}

function toolCallsEqual(a?: ToolCall[], b?: ToolCall[]): boolean {
  if (a === b) return true;
  if (!a || !b) return a === b;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x.tool !== y.tool || x.status !== y.status || x.preview !== y.preview ||
        x.duration !== y.duration || x.reasoning !== y.reasoning) {
      return false;
    }
  }
  return true;
}

function arePropsEqual(prev: Props, next: Props): boolean {
  // Streaming bubble must always re-render so deltas paint.
  if (prev.streaming || next.streaming) return false;
  if (prev.onResend !== next.onResend) return false;
  // runStatus drives false-fail colouring; a change (e.g. running→failed on the
  // last assistant bubble) must repaint even when the message is identical.
  if (prev.runStatus !== next.runStatus) return false;
  const a = prev.message, b = next.message;
  if (a === b) return true;
  return (
    a.id === b.id &&
    a.role === b.role &&
    a.content === b.content &&
    a.thinking === b.thinking &&
    a.timestamp === b.timestamp &&
    a.isNew === b.isNew &&
    toolCallsEqual(a.toolCalls, b.toolCalls)
  );
}

export const MessageBubble = memo(MessageBubbleImpl, arePropsEqual);
