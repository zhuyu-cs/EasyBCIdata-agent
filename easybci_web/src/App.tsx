import { lazy, Suspense, useEffect, useCallback, useMemo, useState } from "react";
import { WorkspaceLayout } from "@/layouts/WorkspaceLayout";
import { SessionPanel } from "@/panels/SessionPanel";
import { ConversationPanel } from "@/panels/ConversationPanel";
import { WorkspacePanel } from "@/panels/WorkspacePanel";
import { ConnectionBanner, GatewayRestartBanner } from "@/components/ConnectionBanner";
import { ToastContainer } from "@/components/Toast";
import type { PaletteAction } from "@/components/CommandPalette";
import { conversationActions, useActiveSlot, useConversationSync, useHasMessages, useVisibleApproval } from "@/hooks/useConversation";
import { useWorkspace } from "@/hooks/useWorkspace";
import { useConnectionStatus } from "@/hooks/useConnectionStatus";
import { useKeyboard } from "@/hooks/useKeyboard";
import { prewarmHighlighter } from "@/hooks/useHighlighter";
import { useSessionListStore } from "@/stores/sessionListStore";
import { useSessionStore } from "@/stores/sessionStore";
import { getSlot } from "@/stores/conversationStore";
import { useUIStore } from "@/stores/uiStore";
import { useThemeStore } from "@/stores/themeStore";
import { exportToMarkdown, downloadMarkdown } from "@/lib/export";
import { visibleSessionOrder } from "@/lib/sessionOrder";
import { BCI_PROMPT_TEMPLATES } from "@/lib/prompts";
import { setRunCompleteTap, setRunEventTap } from "@/lib/runManager";
import { t } from "@/lib/i18n";

// Overlays load on first use, so they don't weigh on first paint.
const ApprovalDialog = lazy(() => import("@/components/ApprovalDialog").then((m) => ({ default: m.ApprovalDialog })));
const SettingsPanel = lazy(() => import("@/components/SettingsPanel").then((m) => ({ default: m.SettingsPanel })));
const CommandPalette = lazy(() => import("@/components/CommandPalette").then((m) => ({ default: m.CommandPalette })));
const ShortcutsOverlay = lazy(() => import("@/components/ShortcutsOverlay").then((m) => ({ default: m.ShortcutsOverlay })));

function exportActiveConversation() {
  const { turns } = getSlot(useSessionStore.getState().activeSessionId);
  if (turns.length === 0) return;
  downloadMarkdown(exportToMarkdown(turns), `easybci-session-${new Date().toISOString().slice(0, 10)}.md`);
}

function navigateSession(dir: -1 | 1) {
  const { activeSessionId, setActiveSessionId, pinnedIds } = useSessionStore.getState();
  const sessions = visibleSessionOrder(useSessionListStore.getState().sessions, pinnedIds);
  if (sessions.length === 0) return;
  const idx = sessions.findIndex((s) => s.id === activeSessionId);
  // From "new session" (no active id) ↑ enters the most recent.
  const nextIdx = idx === -1 ? 0 : Math.min(sessions.length - 1, Math.max(0, idx + dir));
  const next = sessions[nextIdx];
  if (next) setActiveSessionId(next.id);
}

const KEYBOARD_ACTIONS = {
  onCommandPalette: () => useUIStore.getState().togglePalette(),
  onShowShortcuts: () => useUIStore.getState().toggleShortcuts(),
  onExport: exportActiveConversation,
  onInterrupt: conversationActions.interrupt,
  onPrevSession: () => navigateSession(-1),
  onNextSession: () => navigateSession(1),
};

// App subscribes only to what it renders itself; the conversation panel reads
// its own state from the store, so streamed tokens never re-render App, the
// session list or the workspace panel.
export default function App() {
  useConversationSync();
  const workspace = useWorkspace();
  const connection = useConnectionStatus();
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const setActiveSessionId = useSessionStore((s) => s.setActiveSessionId);
  const setTheme = useThemeStore((s) => s.setTheme);
  const isDark = useThemeStore((s) => s.isDark);
  const reloadCount = useActiveSlot("reloadCount");
  const hasMessages = useHasMessages();
  const approvalRequest = useVisibleApproval();

  const settingsOpen = useUIStore((s) => s.settingsOpen);
  // Stays mounted after the first open, so the drawer can slide closed.
  const [settingsLoaded, setSettingsLoaded] = useState(settingsOpen);
  if (settingsOpen && !settingsLoaded) setSettingsLoaded(true);
  const paletteOpen = useUIStore((s) => s.paletteOpen);
  const shortcutsOpen = useUIStore((s) => s.shortcutsOpen);
  const openSettings = useUIStore((s) => s.openSettings);
  const closeSettings = useUIStore((s) => s.closeSettings);
  const closePalette = useUIStore((s) => s.closePalette);
  const toggleShortcuts = useUIStore((s) => s.toggleShortcuts);
  const closeShortcuts = useUIStore((s) => s.closeShortcuts);

  const gatewayDisabled = !connection.gatewayOk && connection.state !== "checking";

  // Source / output directories are derived from the mini-repo on disk via
  // /api/sessions/{id}/artifacts. The panel is cleared the moment the session
  // changes, then re-fetched on switch, on reload, and when a run in this
  // session ends (pipeline_record.json + dirs may have just been written).
  const {
    loadFromArtifacts: workspaceLoadFromArtifacts,
    bindSession: workspaceBindSession,
    handleRunEvent: workspaceHandleRunEvent,
  } = workspace;
  useEffect(() => {
    workspaceBindSession(activeSessionId);
    if (activeSessionId) workspaceLoadFromArtifacts(activeSessionId);
  }, [activeSessionId, reloadCount, workspaceLoadFromArtifacts, workspaceBindSession]);

  useEffect(() => {
    setRunEventTap((sid, ev) => workspaceHandleRunEvent(ev, sid));
    setRunCompleteTap((sid) => {
      if (sid === useSessionStore.getState().activeSessionId) workspaceLoadFromArtifacts(sid);
    });
    return () => {
      setRunEventTap(null);
      setRunCompleteTap(null);
    };
  }, [workspaceHandleRunEvent, workspaceLoadFromArtifacts]);

  useKeyboard(KEYBOARD_ACTIONS);

  // Warm shiki during idle so the first code block paints without a flash.
  useEffect(() => {
    prewarmHighlighter();
  }, []);

  // ── Command palette actions ─────────────────────────────────────────
  const paletteActions: PaletteAction[] = useMemo(() => {
    const actions: PaletteAction[] = [
      {
        id: "new-session",
        label: t("New session"),
        hint: "⌘N",
        keywords: "create start fresh",
        run: () => setActiveSessionId(null),
      },
      {
        id: "open-settings",
        label: t("Open settings"),
        keywords: "config model theme preferences",
        run: openSettings,
      },
      {
        id: "toggle-theme",
        label: isDark ? t("Switch to light mode") : t("Switch to dark mode"),
        keywords: "theme dark light appearance",
        run: () => setTheme(isDark ? "light" : "dark"),
      },
      {
        id: "show-shortcuts",
        label: t("Keyboard shortcuts"),
        hint: "?",
        keywords: "help keys bindings",
        run: toggleShortcuts,
      },
    ];
    if (hasMessages) {
      actions.push({
        id: "export",
        label: t("Export conversation"),
        hint: "⌘E",
        keywords: "download markdown save",
        run: exportActiveConversation,
      });
    }
    // Launch a BCI task directly from the palette.
    for (const tpl of BCI_PROMPT_TEMPLATES) {
      actions.push({
        id: `task-${tpl.id}`,
        label: t(tpl.label),
        keywords: `task template ${t(tpl.prompt)}`,
        run: () => {
          if (!gatewayDisabled) conversationActions.send(t(tpl.prompt));
        },
      });
    }
    return actions;
  }, [isDark, hasMessages, gatewayDisabled, setActiveSessionId, setTheme, openSettings, toggleShortcuts]);

  const onSelectSession = useCallback((id: string) => setActiveSessionId(id), [setActiveSessionId]);

  return (
    <div className="flex flex-col h-dvh">
      <ConnectionBanner state={connection.state} onRetry={connection.retry} />
      <GatewayRestartBanner
        show={connection.gatewayRestarted}
        onDismiss={connection.acknowledgeGatewayRestart}
      />
      <div className="flex-1 min-h-0">
        <WorkspaceLayout
          sessionPanel={<SessionPanel onOpenSettings={openSettings} />}
          conversationPanel={<ConversationPanel disabled={gatewayDisabled} />}
          workspacePanel={<WorkspacePanel workspace={workspace} />}
          workspaceVisible={workspace.panelVisible}
        />
      </div>

      <Suspense fallback={null}>
        {approvalRequest && (
          <ApprovalDialog
            request={approvalRequest}
            onResolved={conversationActions.resolveApproval}
            onHide={conversationActions.hideApproval}
          />
        )}

        {settingsLoaded && <SettingsPanel open={settingsOpen} onClose={closeSettings} />}
        {paletteOpen && (
          <CommandPalette
            actions={paletteActions}
            onClose={closePalette}
            onSelectSession={onSelectSession}
          />
        )}
        {shortcutsOpen && <ShortcutsOverlay onClose={closeShortcuts} />}
      </Suspense>

      <ToastContainer />
    </div>
  );
}
