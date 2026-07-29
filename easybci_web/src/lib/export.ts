import type { Message } from "@/hooks/useConversation";

function formatToolCalls(toolCalls: Message["toolCalls"]): string {
  if (!toolCalls?.length) return "";
  const lines = toolCalls.map((tc) => {
    const status = tc.status === "error" ? " (failed)" : "";
    const duration = tc.duration ? ` — ${(tc.duration / 1000).toFixed(1)}s` : "";
    return `- \`${tc.tool}\`${status}${duration}${tc.preview ? `: ${tc.preview}` : ""}`;
  });
  return `\n\n<details><summary>Tool calls (${toolCalls.length})</summary>\n\n${lines.join("\n")}\n\n</details>`;
}

function formatThinking(thinking: string | undefined): string {
  if (!thinking) return "";
  return `\n\n<details><summary>Thinking</summary>\n\n${thinking.trim()}\n\n</details>`;
}

export function exportToMarkdown(messages: Message[], sessionTitle?: string): string {
  const title = sessionTitle || "EasyBCI Conversation";
  const timestamp = new Date().toISOString().split("T")[0];
  const lines: string[] = [`# ${title}`, ``, `*Exported: ${timestamp}*`, ``];

  for (const msg of messages) {
    const roleLabel = msg.role === "user" ? "**User**" : "**Assistant**";
    lines.push(`## ${roleLabel}`);
    lines.push("");
    if (msg.content) lines.push(msg.content);
    lines.push(formatThinking(msg.thinking));
    lines.push(formatToolCalls(msg.toolCalls));
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  return lines.join("\n");
}

export function downloadMarkdown(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
