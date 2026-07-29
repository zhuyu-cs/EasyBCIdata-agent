import { create } from "zustand";

// Centralizes the open/closed state of the global overlays (Settings drawer,
// Command Palette, Shortcuts help) so they can be triggered from anywhere —
// the SessionPanel gear, keyboard shortcuts, or Command Palette actions —
// without threading callbacks through the panel tree.
interface UIStore {
  settingsOpen: boolean;
  paletteOpen: boolean;
  shortcutsOpen: boolean;
  openSettings: () => void;
  closeSettings: () => void;
  togglePalette: () => void;
  closePalette: () => void;
  toggleShortcuts: () => void;
  closeShortcuts: () => void;
  closeAll: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  settingsOpen: false,
  paletteOpen: false,
  shortcutsOpen: false,
  openSettings: () => set({ settingsOpen: true, paletteOpen: false }),
  closeSettings: () => set({ settingsOpen: false }),
  togglePalette: () => set((s) => ({ paletteOpen: !s.paletteOpen, shortcutsOpen: false })),
  closePalette: () => set({ paletteOpen: false }),
  toggleShortcuts: () => set((s) => ({ shortcutsOpen: !s.shortcutsOpen, paletteOpen: false })),
  closeShortcuts: () => set({ shortcutsOpen: false }),
  closeAll: () => set({ settingsOpen: false, paletteOpen: false, shortcutsOpen: false }),
}));
