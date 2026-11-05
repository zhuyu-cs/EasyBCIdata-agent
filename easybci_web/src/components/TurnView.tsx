import { memo, useCallback, useMemo, useState } from "react";
import { splitTurn, type Turn } from "@/lib/turns";
import { formatDuration } from "@/lib/utils";
import { conversationActions } from "@/hooks/useConversation";
import { ContextMenu, type ContextMenuItem } from "./ContextMenu";
import { LogoMark } from "./Logo";
import { MarkdownContent } from "./Markdown";
import { MessageToolbar } from "./MessageToolbar";
import { StepList } from "./StepList";
import { t } from "@/lib/i18n";

// Text longer than this can be collapsed from the hover toolbar.
const COLLAPSE_THRESHOLD = 900;

function UserAvatar() {
  return (
    <div
      className="w-6 h-6 rounded-full flex items-center justify-center shrink-0"
      style={{ background: "var(--bg-active)", color: "var(--text-secondary)" }}
      aria-label={t("You")}
      role="img"
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
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

/** HH:MM for today, MM-DD HH:MM otherwise (older sessions need the date). */
function formatTurnTime(tsSec: number): string {
  const d = new Date(tsSec * 1000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return time;
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const prefix = d.getFullYear() === now.getFullYear() ? `${mm}-${dd}` : `${d.getFullYear()}-${mm}-${dd}`;
  return `${prefix} ${time}`;
}

function useCopied(text: string) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);
  return { copied, copy };
}

const STATUS_BADGE: Partial<Record<Turn["status"], { label: string; color: string }>> = {
  running: { label: "Running", color: "var(--accent-green)" },
  failed: { label: "Failed", color: "var(--text-error)" },
  cancelled: { label: "Stopped", color: "var(--text-muted)" },
  incomplete: { label: "No reply", color: "var(--text-muted)" },
};

function UserHead({ turn }: { turn: Turn }) {
  const content = turn.user?.content ?? "";
  const { copied, copy } = useCopied(content);
  const [collapsed, setCollapsed] = useState(false);
  const collapsible = content.length > COLLAPSE_THRESHOLD;
  const badge = STATUS_BADGE[turn.status];
  const ts = turn.user?.ts ?? 0;
  const duration = formatDuration(turn.durationSec);

  return (
    <div className="flex gap-2.5 group/msg mb-3">
      <UserAvatar />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 mb-1 min-h-6">
          <span className="text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>{t("You")}</span>
          {ts > 0 && (
            <span className="text-[11px]" style={{ color: "var(--text-faint)" }} title={new Date(ts * 1000).toLocaleString()}>
              {formatTurnTime(ts)}
            </span>
          )}
          {duration && <span className="text-[11px] tabular-nums" style={{ color: "var(--text-faint)" }}>{duration}</span>}
          {badge && (
            <span className="flex items-center gap-1 text-[11px]" style={{ color: badge.color }}>
              <span
                className={`w-1.5 h-1.5 rounded-full ${turn.status === "running" ? "animate-pulse" : ""}`}
                style={{ background: badge.color }}
              />
              {t(badge.label)}
            </span>
          )}
          <MessageToolbar
            onCopy={copy}
            copied={copied}
            onResend={content ? () => conversationActions.send(content) : undefined}
            collapsible={collapsible}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((c) => !c)}
          />
        </div>
        <div
          className="px-3.5 py-2.5 text-[13.5px] leading-relaxed whitespace-pre-wrap break-words"
          style={{
            background: "var(--bg-user-bubble)",
            color: "var(--text-primary)",
            borderRadius: "4px 16px 16px 16px",
            ...(collapsed ? { maxHeight: 120, overflow: "hidden" } : {}),
          }}
        >
          {content}
        </div>
      </div>
    </div>
  );
}

function Answer({ text, streaming }: { text: string; streaming: boolean }) {
  const { copied, copy } = useCopied(text);
  const [collapsed, setCollapsed] = useState(false);
  const collapsible = !streaming && text.length > COLLAPSE_THRESHOLD;
  return (
    <div className="relative w-full group/msg">
      <div className="absolute -top-1 right-0 z-[1]">
        <MessageToolbar
          onCopy={copy}
          copied={copied}
          collapsible={collapsible}
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed((c) => !c)}
        />
      </div>
      <div
        className="text-[13.5px] leading-relaxed"
        style={{
          color: "var(--text-primary)",
          ...(collapsed ? { maxHeight: 200, overflow: "hidden" } : {}),
        }}
      >
        <div className="prose-compact">
          <MarkdownContent content={text} streaming={streaming} />
        </div>
      </div>
      {collapsed && (
        <button
          onClick={() => setCollapsed(false)}
          className="absolute inset-x-0 bottom-0 flex items-end justify-center pb-1.5 pt-8 text-[11px] font-medium rounded-b-lg"
          style={{
            color: "var(--text-secondary)",
            background: "linear-gradient(to bottom, transparent, var(--bg-secondary) 70%)",
          }}
        >
          {t("Show more")}
        </button>
      )}
    </div>
  );
}

function Outcome({ turn }: { turn: Turn }) {
  if (turn.status !== "failed" && turn.status !== "cancelled") return null;
  const failed = turn.status === "failed";
  const color = failed ? "var(--text-error)" : "var(--text-muted)";
  return (
    <div className="mt-1.5 flex items-center gap-1.5 text-[11px]" style={{ color }} role="status">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {failed ? t("Run failed") : t("Stopped by you")}
      {turn.error && <span className="opacity-80 truncate max-w-[480px]">— {turn.error}</span>}
    </div>
  );
}

function turnMarkdown(turn: Turn, answer: string): string {
  const thinking = turn.items.flatMap((it) => (it.kind === "thinking" ? [it.text] : [])).join("\n\n");
  const parts: string[] = [];
  if (thinking) parts.push(`<details><summary>Thinking</summary>\n\n${thinking}\n\n</details>\n`);
  if (answer) parts.push(answer);
  return parts.join("\n");
}

interface Props {
  turn: Turn;
  /** The newest turn keeps its process area open by default; older ones fold
   *  to their one-line summary with the answer still visible. */
  isLatest: boolean;
}

/** One round of the conversation: what was asked, what the agent did, what it
 *  replied. The single renderer for both live and reopened turns. */
export const TurnView = memo(function TurnView({ turn, isLatest }: Props) {
  const { process, answer } = useMemo(() => splitTurn(turn), [turn]);
  const running = turn.status === "running";
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const onContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const menuItems: ContextMenuItem[] = [
    { label: t("Copy text"), action: () => void navigator.clipboard.writeText(answer || turn.user?.content || "") },
    { label: t("Copy as Markdown"), action: () => void navigator.clipboard.writeText(turnMarkdown(turn, answer)) },
  ];
  if (turn.user) {
    const content = turn.user.content;
    menuItems.push({ label: t("Re-send"), action: () => conversationActions.send(content) });
  }

  return (
    <article className={turn.live ? "animate-fade-in" : undefined} onContextMenu={onContextMenu}>
      {turn.user && <UserHead turn={turn} />}
      {(process.length > 0 || answer) && (
        <div className="flex gap-2.5">
          <AgentAvatar />
          <div className="min-w-0 flex-1">
            <StepList items={process} status={turn.status} expandKey={turn.id} defaultExpanded={isLatest || running} />
            {answer && <Answer text={answer} streaming={running} />}
            <Outcome turn={turn} />
          </div>
        </div>
      )}
      {process.length === 0 && !answer && (
        <div className="pl-[34px]"><Outcome turn={turn} /></div>
      )}
      {menu && <ContextMenu x={menu.x} y={menu.y} items={menuItems} onClose={() => setMenu(null)} />}
    </article>
  );
});
