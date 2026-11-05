import { splitTurn, turnSteps, type Turn } from "@/lib/turns";

function formatSteps(turn: Turn): string {
  const steps = turnSteps(turn);
  if (!steps.length) return "";
  const lines = steps.map((s) => {
    const status = s.status === "error" ? " (failed)" : "";
    const duration = s.durationSec ? ` — ${s.durationSec.toFixed(1)}s` : "";
    return `- \`${s.tool}\`${status}${duration}${s.preview ? `: ${s.preview}` : ""}`;
  });
  return `<details><summary>Tool calls (${steps.length})</summary>\n\n${lines.join("\n")}\n\n</details>`;
}

function formatThinking(turn: Turn): string {
  const thinking = turn.items.flatMap((it) => (it.kind === "thinking" ? [it.text.trim()] : [])).join("\n\n");
  return thinking ? `<details><summary>Thinking</summary>\n\n${thinking}\n\n</details>` : "";
}

export function exportToMarkdown(turns: Turn[], sessionTitle?: string): string {
  const title = sessionTitle || "EasyBCI Conversation";
  const timestamp = new Date().toISOString().split("T")[0];
  const lines: string[] = [`# ${title}`, ``, `*Exported: ${timestamp}*`, ``];

  for (const turn of turns) {
    if (turn.user) lines.push(`## **User**`, "", turn.user.content, "");
    const { answer } = splitTurn(turn);
    lines.push(`## **Assistant**`, "");
    if (answer) lines.push(answer, "");
    for (const block of [formatThinking(turn), formatSteps(turn)]) if (block) lines.push(block, "");
    lines.push("---", "");
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
