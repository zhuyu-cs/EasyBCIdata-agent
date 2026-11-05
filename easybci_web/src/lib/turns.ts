import type { SessionMessage } from "@/lib/api";

/**
 * The conversation is rendered from one model: a list of turns, each a user
 * input and everything that happened in response. Live runs (run events) and
 * reopened sessions (stored rows) both produce it, so a reload shows the same
 * view that was on screen while the run was live.
 *
 * No React in this file — pure types and functions.
 */

/** `incomplete`: the stored transcript ends without a final reply. */
export type TurnStatus = "running" | "completed" | "failed" | "cancelled" | "incomplete";

export interface Step {
  /** Unique within the turn; the render key. */
  id: string;
  /** Provider tool_call id used to pair a call with its result. */
  callId?: string;
  tool: string;
  /** One line naming the call's primary argument. */
  preview?: string;
  status: "running" | "done" | "error";
  durationSec?: number;
  /** Full text from history; only the head while a run is live. */
  result?: string;
  /** Original length of the result. */
  resultChars?: number;
}

export type TurnItem =
  | { kind: "thinking"; text: string }
  | { kind: "text"; text: string }
  | { kind: "step"; step: Step };

export interface Turn {
  id: string;
  /** null: assistant rows with no user row before them, or a run re-attached
   *  after a page reload whose input isn't known yet. */
  user: { content: string; ts: number } | null;
  /** In the order they happened. */
  items: TurnItem[];
  status: TurnStatus;
  /** Wall-clock length. Only known for turns seen live: stored rows are all
   *  written together when the turn ends, so their timestamps can't give it. */
  durationSec?: number;
  error?: string;
  /** Started in this page and not yet confirmed by the stored transcript. */
  live?: boolean;
}

// ── ids ────────────────────────────────────────────────────────────────

function hashText(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/**
 * Derived from the user text and how often it occurred before, so a live turn
 * and its stored version share an id — even when an earlier turn never reached
 * the database, which would shift a position-based id.
 */
function turnId(sid: string, content: string | null, occurrence: number): string {
  return `${sid}:${content === null ? "orphan" : hashText(content.trim())}:${occurrence}`;
}

function occurrenceOf(turns: Turn[], sid: string, content: string | null): number {
  const prefix = turnId(sid, content, 0).slice(0, -1);
  return turns.reduce((n, t) => (t.id.startsWith(prefix) ? n + 1 : n), 0);
}

/** A new running turn following `existing`. */
export function newLiveTurn(sid: string, existing: Turn[], content: string | null, tsSec: number): Turn {
  return {
    id: turnId(sid, content, occurrenceOf(existing, sid, content)),
    user: content === null ? null : { content, ts: tsSec },
    items: [],
    status: "running",
    live: true,
  };
}

// ── derived views ──────────────────────────────────────────────────────

export function turnSteps(turn: Pick<Turn, "items">): Step[] {
  return turn.items.flatMap((it) => (it.kind === "step" ? [it.step] : []));
}

/**
 * Text after the last step is the answer; everything before it (thinking,
 * narration between steps, the steps) is the process. Computed rather than
 * stored so a streaming reply already sits in its final place and only moves
 * if a further tool call turns it into narration.
 */
export function splitTurn(turn: Turn): { process: TurnItem[]; answer: string } {
  let lastStep = -1;
  turn.items.forEach((it, i) => {
    if (it.kind === "step") lastStep = i;
  });
  const process: TurnItem[] = [];
  const answer: string[] = [];
  turn.items.forEach((it, i) => {
    if (it.kind === "text" && i > lastStep) answer.push(it.text);
    else process.push(it);
  });
  return { process, answer: answer.join("\n\n") };
}

// ── live events ────────────────────────────────────────────────────────

export function appendText(turn: Turn, text: string): Turn {
  const items = turn.items.slice();
  const last = items[items.length - 1];
  if (last?.kind === "text") items[items.length - 1] = { kind: "text", text: last.text + text };
  else items.push({ kind: "text", text });
  return { ...turn, items };
}

function uniqueStepId(turn: Turn, wanted: string | undefined): string {
  const taken = new Set(turnSteps(turn).map((s) => s.id));
  const base = wanted || `s${turn.items.length}`;
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(`${base}#${n}`)) n++;
  return `${base}#${n}`;
}

export function startStep(turn: Turn, ev: { callId?: string; tool: string; preview?: string }): Turn {
  const step: Step = {
    id: uniqueStepId(turn, ev.callId),
    callId: ev.callId,
    tool: ev.tool,
    preview: ev.preview,
    status: "running",
  };
  return { ...turn, items: [...turn.items, { kind: "step", step }] };
}

function findLastIndex<T>(list: T[], pred: (x: T) => boolean): number {
  for (let i = list.length - 1; i >= 0; i--) if (pred(list[i])) return i;
  return -1;
}

/** Pairs by tool_call_id; events without one match the newest running step of
 *  that tool. */
export function completeStep(
  turn: Turn,
  ev: { callId?: string; tool?: string; error: boolean; durationSec?: number; result?: string; resultChars?: number },
): Turn {
  const idx = findLastIndex(turn.items, (it) => {
    if (it.kind !== "step" || it.step.status !== "running") return false;
    return ev.callId ? it.step.callId === ev.callId : it.step.tool === ev.tool;
  });
  if (idx < 0) return turn;
  const items = turn.items.slice();
  const prev = items[idx] as Extract<TurnItem, { kind: "step" }>;
  items[idx] = {
    kind: "step",
    step: {
      ...prev.step,
      status: ev.error ? "error" : "done",
      durationSec: ev.durationSec,
      result: ev.result,
      resultChars: ev.resultChars,
    },
  };
  return { ...turn, items };
}

/** Close a running turn. Output only fills in a reply that never streamed
 *  (non-streaming providers); a turn already closed is left as it is. */
export function closeTurn(
  turn: Turn,
  status: Exclude<TurnStatus, "running" | "incomplete">,
  opts: { error?: string; output?: string; durationSec?: number } = {},
): Turn {
  if (turn.status !== "running") return turn;
  const base = opts.output && !splitTurn(turn).answer ? appendText(turn, opts.output) : turn;
  // Steps cut off by the end of the run never report back.
  const items = base.items.map((it) =>
    it.kind === "step" && it.step.status === "running"
      ? { kind: "step" as const, step: { ...it.step, status: "error" as const } }
      : it,
  );
  return { ...base, items, status, error: opts.error, durationSec: opts.durationSec };
}

// ── stored rows ────────────────────────────────────────────────────────

/**
 * Best-effort text extraction from a provider-specific `reasoning_details`
 * payload. Some providers (OpenAI Responses API, OpenRouter details-only) put
 * the thinking ONLY here, leaving `reasoning`/`reasoning_content` null. Handles
 * a plain string, an array of `{text|summary|content}` blocks, or a single
 * such object. Returns undefined when nothing textual is found (never throws).
 */
function extractReasoningDetails(details: unknown): string | undefined {
  if (!details) return undefined;
  if (typeof details === "string") return details.trim() || undefined;
  const pluck = (o: unknown): string | undefined => {
    if (!o) return undefined;
    if (typeof o === "string") return o;
    if (typeof o === "object") {
      const r = o as Record<string, unknown>;
      const v = r.text ?? r.summary ?? r.content ?? r.reasoning;
      if (typeof v === "string") return v;
      // Nested content arrays (e.g. Responses items with {content:[{text}]}).
      if (Array.isArray(v)) {
        const inner = v.map(pluck).filter(Boolean).join("\n").trim();
        return inner || undefined;
      }
    }
    return undefined;
  };
  if (Array.isArray(details)) {
    const joined = details.map(pluck).filter(Boolean).join("\n").trim();
    return joined || undefined;
  }
  return pluck(details);
}

// Same primary-argument table as the gateway's live preview
// (easybci_agent/display.py:build_tool_preview), so a stored step reads like
// the live one; neural tools name their input instead of carrying a preview.
const PRIMARY_ARG: Record<string, string> = {
  terminal: "command", web_search: "query", web_extract: "urls",
  read_file: "path", write_file: "path", patch: "path",
  search_files: "pattern", browser_navigate: "url",
  browser_click: "ref", browser_type: "text",
  vision_analyze: "question", mixture_of_agents: "user_prompt",
  skill_view: "name", skills_list: "category",
  execute_code: "code", delegate_task: "goal",
  clarify: "question", skill_manage: "name",
};
const FALLBACK_ARGS = ["query", "text", "command", "path", "name", "prompt", "code", "goal", "data_path", "work_dir", "output_dir"];
const PREVIEW_MAX = 120;

function oneLine(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

export function summarizeArgs(tool: string, rawArgs: string | undefined): string | undefined {
  if (!rawArgs) return undefined;
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(rawArgs);
  } catch {
    return undefined;
  }
  if (!args || typeof args !== "object") return undefined;
  const key = PRIMARY_ARG[tool] ?? FALLBACK_ARGS.find((k) => k in args);
  if (!key || !(key in args)) return undefined;
  const raw = args[key];
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (value == null) return undefined;
  const line = oneLine(String(value));
  if (!line) return undefined;
  return line.length > PREVIEW_MAX ? `${line.slice(0, PREVIEW_MAX - 3)}...` : line;
}

function rowReasoning(row: SessionMessage): string | undefined {
  return row.reasoning ?? row.reasoning_content ?? extractReasoningDetails(row.reasoning_details) ?? undefined;
}

/** Group stored rows into turns: each user row opens one. `system` rows are
 *  prompt scaffolding and stay hidden; a tool row whose call isn't in the turn
 *  still shows as a step, so real tool output is never discarded. */
export function historyToTurns(sid: string, rows: SessionMessage[]): Turn[] {
  const turns: Turn[] = [];
  const seen = new Map<string, number>();

  const open = (content: string | null, ts: number): Turn => {
    const probe = turnId(sid, content, 0);
    const n = seen.get(probe) ?? 0;
    seen.set(probe, n + 1);
    const turn: Turn = {
      id: turnId(sid, content, n),
      user: content === null ? null : { content, ts },
      items: [],
      status: "incomplete",
    };
    turns.push(turn);
    return turn;
  };
  const current = (row: SessionMessage): Turn => turns[turns.length - 1] ?? open(null, row.timestamp ?? 0);

  for (const row of rows) {
    if (row.role === "user") {
      open(row.content ?? "", row.timestamp ?? 0);
    } else if (row.role === "assistant") {
      const turn = current(row);
      const thinking = rowReasoning(row);
      if (thinking) turn.items.push({ kind: "thinking", text: thinking });
      if (row.content) turn.items.push({ kind: "text", text: row.content });
      for (const tc of row.tool_calls ?? []) {
        turn.items.push({
          kind: "step",
          step: {
            id: uniqueStepId(turn, tc.id),
            callId: tc.id,
            tool: tc.function.name,
            preview: summarizeArgs(tc.function.name, tc.function.arguments),
            // Legacy tool rows have no status; a call with no tool row stays
            // "done" too — nothing recorded that it failed.
            status: "done",
          },
        });
      }
    } else if (row.role === "tool") {
      const turn = current(row);
      const status: Step["status"] = row.tool_status === "error" ? "error" : "done";
      const fill = {
        status,
        durationSec: row.tool_duration ?? undefined,
        result: row.content ?? undefined,
        resultChars: row.content?.length,
      };
      // Oldest unfilled call first: results are stored in call order, which
      // matters when a provider repeats ids within a turn.
      const idx = turn.items.findIndex(
        (it) => it.kind === "step" && it.step.callId === row.tool_call_id && it.step.result === undefined,
      );
      if (idx >= 0) {
        const prev = turn.items[idx] as Extract<TurnItem, { kind: "step" }>;
        turn.items[idx] = { kind: "step", step: { ...prev.step, ...fill } };
      } else {
        turn.items.push({
          kind: "step",
          step: { id: uniqueStepId(turn, row.tool_call_id), tool: row.tool_name || "tool", ...fill },
        });
      }
    }
  }
  for (const t of turns) if (splitTurn(t).answer) t.status = "completed";
  return turns;
}

// ── reconciling ────────────────────────────────────────────────────────

const KEPT_STATUS: TurnStatus[] = ["failed", "cancelled"];

/** What only the page saw and the stored rows can't say. */
function overlay(stored: Turn, seen: Turn): Turn {
  const merged: Turn = {
    ...stored,
    user: seen.user ?? stored.user, // the send time, not the end-of-turn write
    durationSec: seen.durationSec ?? stored.durationSec,
  };
  if (KEPT_STATUS.includes(seen.status)) {
    merged.status = seen.status;
    merged.error = seen.error;
    // A stopped run keeps only the user row, so what streamed before the stop
    // exists only here.
    if (seen.items.length > stored.items.length) merged.items = seen.items;
  }
  return merged;
}

function sameTurn(a: Turn, b: Turn): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Stored rows win, matched by id so nothing remounts. An unchanged turn keeps
 * its object, so a revalidation doesn't re-render settled turns. A live turn
 * with no stored row survives only if the run ended early and may never have
 * been written; a completed one is in the rows under whatever id they gave it.
 */
export function mergeTurns(stored: Turn[], prev: Turn[]): Turn[] {
  const prevById = new Map(prev.map((t) => [t.id, t]));
  const storedIds = new Set(stored.map((t) => t.id));
  const merged = stored.map((s) => {
    const seen = prevById.get(s.id);
    if (!seen) return s;
    const next = overlay(s, seen);
    return sameTurn(next, seen) ? seen : next;
  });
  prev.forEach((p, i) => {
    if (!p.live || storedIds.has(p.id) || !p.user) return;
    if (!KEPT_STATUS.includes(p.status) && p.status !== "running") return;
    const before = i === 0 ? -1 : merged.findIndex((t) => t.id === prev[i - 1].id);
    merged.splice(i > 0 && before < 0 ? merged.length : before + 1, 0, p);
  });
  return merged;
}
