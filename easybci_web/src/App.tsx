import { useEffect, useCallback, useMemo } from "react";
import { WorkspaceLayout } from "@/layouts/WorkspaceLayout";
import { SessionPanel } from "@/panels/SessionPanel";
import { ConversationPanel } from "@/panels/ConversationPanel";
import { WorkspacePanel } from "@/panels/WorkspacePanel";
import { ConnectionBanner, GatewayRestartBanner } from "@/components/ConnectionBanner";
import { ApprovalDialog } from "@/components/ApprovalDialog";
import { ToastContainer } from "@/components/Toast";
import { SettingsPanel } from "@/components/SettingsPanel";
import { CommandPalette, type PaletteAction } from "@/components/CommandPalette";
import { ShortcutsOverlay } from "@/components/ShortcutsOverlay";
import { useConversation } from "@/hooks/useConversation";
import { useWorkspace } from "@/hooks/useWorkspace";
import { useConnectionStatus } from "@/hooks/useConnectionStatus";
import { useKeyboard } from "@/hooks/useKeyboard";
import { prewarmHighlighter } from "@/hooks/useHighlighter";
import { useSessionList } from "@/hooks/useSessionList";
import { useSessionStore } from "@/stores/sessionStore";
import { useUIStore } from "@/stores/uiStore";
import { useThemeStore } from "@/stores/themeStore";
import { exportToMarkdown, downloadMarkdown } from "@/lib/export";
import { BCI_PROMPT_TEMPLATES } from "@/lib/prompts";

export default function App() {
  const conversation = useConversation();
  const workspace = useWorkspace();
  const connection = useConnectionStatus();
  const { sessions } = useSessionList();
  const triggerRefresh = useSessionStore((s) => s.triggerRefresh);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const setActiveSessionId = useSessionStore((s) => s.setActiveSessionId);
  const setTheme = useThemeStore((s) => s.setTheme);
  const theme = useThemeStore((s) => s.theme);

  const ui = useUIStore();

  const gatewayDisabled = !connection.gatewayOk && connection.state !== "checking";

  // Source / output directories are derived from the mini-repo on disk via
  // /api/sessions/{id}/artifacts. Re-fetched whenever activeSessionId changes
  // or reloadSession bumps the reload counter (post external-update or
  // post-stream correction). isStreaming is in the deps so when a run finishes
  // we refetch (pipeline_record.json + dirs may have just been written).
  const { loadFromArtifacts: workspaceLoadFromArtifacts, reset: workspaceReset } = workspace;
  const reloadCount = conversation.reloadCount;
  const isStreaming = conversation.isStreaming;
  useEffect(() => {
    if (!activeSessionId) {
      workspaceReset();
      return;
    }
    workspaceLoadFromArtifacts(activeSessionId);
  }, [activeSessionId, reloadCount, isStreaming, workspaceLoadFromArtifacts, workspaceReset]);

  // ── Shared action handlers ──────────────────────────────────────────
  const exportConversation = useCallback(() => {
    if (conversation.messages.length === 0) return;
    const md = exportToMarkdown(conversation.messages);
    downloadMarkdown(md, `easybci-session-${new Date().toISOString().slice(0, 10)}.md`);
  }, [conversation.messages]);

  const navigateSession = useCallback(
    (dir: -1 | 1) => {
      if (sessions.length === 0) return;
      const idx = sessions.findIndex((s) => s.id === activeSessionId);
      // From "new session" (no active id) ↑ enters the most recent.
      const nextIdx = idx === -1 ? 0 : Math.min(sessions.length - 1, Math.max(0, idx + dir));
      const next = sessions[nextIdx];
      if (next) setActiveSessionId(next.id);
    },
    [sessions, activeSessionId, setActiveSessionId],
  );

  // ── Keyboard shortcuts ──────────────────────────────────────────────
  useKeyboard({
    onCloseOverlay: useCallback(() => {
      if (conversation.approvalRequest) {
        conversation.dismissApproval();
        return;
      }
      ui.closeAll();
    }, [conversation, ui]),
    onCommandPalette: useCallback(() => ui.togglePalette(), [ui]),
    onShowShortcuts: useCallback(() => ui.toggleShortcuts(), [ui]),
    onExport: exportConversation,
    onInterrupt: conversation.interrupt,
    onPrevSession: useCallback(() => navigateSession(-1), [navigateSession]),
    onNextSession: useCallback(() => navigateSession(1), [navigateSession]),
  });

  // Stable setters from the conversation hook (useCallback with [] deps) — pull
  // them out so the effects below can depend on the functions directly.
  const { setWorkspaceCallback, setOnRunComplete } = conversation;

  useEffect(() => {
    // Wrap so handleRunEvent receives the current activeSessionId — it uses
    // it to call workspace.refreshFromArtifacts(sid) on tool.completed /
    // run.completed (live artifacts refresh).
    setWorkspaceCallback((ev) => workspace.handleRunEvent(ev, activeSessionId));
  }, [setWorkspaceCallback, workspace, activeSessionId]);

  useEffect(() => {
    setOnRunComplete(triggerRefresh);
  }, [setOnRunComplete, triggerRefresh]);

  // Warm shiki during idle so the first code block paints without a flash (U13).
  useEffect(() => {
    prewarmHighlighter();
  }, []);

  // ── Command palette actions ─────────────────────────────────────────
  const paletteActions: PaletteAction[] = useMemo(() => {
    const isDark =
      theme === "dark" ||
      (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const actions: PaletteAction[] = [
      {
        id: "new-session",
        label: "New session",
        hint: "⌘N",
        keywords: "create start fresh",
        run: () => setActiveSessionId(null),
      },
      {
        id: "open-settings",
        label: "Open settings",
        keywords: "config model theme preferences",
        run: () => ui.openSettings(),
      },
      {
        id: "toggle-theme",
        label: isDark ? "Switch to light mode" : "Switch to dark mode",
        keywords: "theme dark light appearance",
        run: () => setTheme(isDark ? "light" : "dark"),
      },
      {
        id: "show-shortcuts",
        label: "Keyboard shortcuts",
        hint: "?",
        keywords: "help keys bindings",
        run: () => ui.toggleShortcuts(),
      },
    ];
    if (conversation.messages.length > 0) {
      actions.push({
        id: "export",
        label: "Export conversation",
        hint: "⌘E",
        keywords: "download markdown save",
        run: exportConversation,
      });
    }
    // Launch a BCI task directly from the palette.
    for (const t of BCI_PROMPT_TEMPLATES) {
      actions.push({
        id: `task-${t.id}`,
        label: t.label,
        keywords: `task template ${t.prompt}`,
        run: () => {
          if (!gatewayDisabled) conversation.send(t.prompt);
        },
      });
    }
    return actions;
  }, [theme, conversation, ui, setActiveSessionId, setTheme, exportConversation, gatewayDisabled]);

  return (
    <div className="flex flex-col h-screen">
      <ConnectionBanner state={connection.state} onRetry={connection.retry} />
      <GatewayRestartBanner
        show={connection.gatewayRestarted}
        onDismiss={connection.acknowledgeGatewayRestart}
      />
      <div className="flex-1 min-h-0">
        <WorkspaceLayout
          sessionPanel={<SessionPanel onOpenSettings={ui.openSettings} />}
          conversationPanel={
            <ConversationPanel
              messages={conversation.messages}
              isStreaming={conversation.isStreaming}
              runStatus={conversation.runStatus}
              progress={conversation.progress}
              latestTurnEta={conversation.latestTurnEta}
              turnStartedAtMs={conversation.turnStartedAtMs}
              error={conversation.error}
              onSend={conversation.send}
              onInterrupt={conversation.interrupt}
              disabled={gatewayDisabled}
              streamStatus={conversation.streamStatus}
              onDismissStreamError={conversation.dismissStreamError}
              onCopyStreamedOutput={conversation.copyStreamedOutput}
              externalUpdateAvailable={conversation.externalUpdateAvailable}
              onReloadSession={conversation.reloadSession}
              pendingCount={conversation.pendingMessages.length}
              onFlushPending={conversation.flushPending}
              onClearPending={conversation.clearPending}
            />
          }
          workspacePanel={<WorkspacePanel workspace={workspace} />}
        />
      </div>

      {conversation.approvalRequest && (
        <ApprovalDialog
          request={conversation.approvalRequest}
          onResolved={conversation.dismissApproval}
        />
      )}

      <SettingsPanel open={ui.settingsOpen} onClose={ui.closeSettings} />
      {ui.paletteOpen && (
        <CommandPalette
          actions={paletteActions}
          onClose={ui.closePalette}
          onSelectSession={(id) => setActiveSessionId(id)}
        />
      )}
      {ui.shortcutsOpen && <ShortcutsOverlay onClose={ui.closeShortcuts} />}

      <ToastContainer />
    </div>
  );
}
