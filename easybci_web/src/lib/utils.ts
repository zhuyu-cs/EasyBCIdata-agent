import { twMerge } from "tailwind-merge";
import { t } from "@/lib/i18n";

export function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(inputs.filter(Boolean).join(" "));
}

/** Human duration from seconds (`850ms`, `12.4s`, `2m 13s`). */
export function formatDuration(sec: number | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "";
  if (sec < 1) return `${Math.max(1, Math.round(sec * 1000))}ms`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  return `${m}m ${Math.round(sec - m * 60)}s`;
}

/** Relative time from a Unix epoch timestamp (seconds). */
export function timeAgo(ts: number): string {
  const delta = Date.now() / 1000 - ts;
  if (delta < 60) return t("just now");
  if (delta < 3600) return t("{n}m ago", { n: Math.floor(delta / 60) });
  if (delta < 86400) return t("{n}h ago", { n: Math.floor(delta / 3600) });
  if (delta < 172800) return t("yesterday");
  return t("{n}d ago", { n: Math.floor(delta / 86400) });
}

/** Relative time from an ISO-8601 timestamp string. */
export function isoTimeAgo(iso: string): string {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (delta < 0 || Number.isNaN(delta)) return "unknown";
  if (delta < 60) return t("just now");
  if (delta < 3600) return t("{n}m ago", { n: Math.floor(delta / 60) });
  if (delta < 86400) return t("{n}h ago", { n: Math.floor(delta / 3600) });
  return t("{n}d ago", { n: Math.floor(delta / 86400) });
}
