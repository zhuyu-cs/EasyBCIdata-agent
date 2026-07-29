import { useToastStore } from "@/stores/toastStore";

const ICONS = {
  success: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="6" stroke="#2d8a4e" strokeWidth="1.5" />
      <path d="M4.5 7l2 2 3.5-3.5" stroke="#2d8a4e" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  error: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="6" stroke="#d1242f" strokeWidth="1.5" />
      <path d="M5 5l4 4M9 5l-4 4" stroke="#d1242f" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  info: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-[var(--text-secondary)]">
      <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7 6v4M7 4.5v0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
};

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const remove = useToastStore((s) => s.remove);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="pointer-events-auto flex items-center gap-2 px-4 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-lg shadow-lg animate-fade-in cursor-pointer"
          onClick={() => remove(toast.id)}
        >
          {ICONS[toast.type]}
          <span className="text-[13px] text-[var(--text-primary)]">{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
