/**
 * Split markdown into top-level blocks so a streaming message only re-parses
 * the block still being written; every earlier block keeps its text (and its
 * key), so it renders once and stays put.
 *
 * Boundaries are conservative — a split must never change how the text
 * renders:
 *   - a blank line ends a block only when the next line starts at column 0
 *     (indented lines continue a list item or code block);
 *   - a fence that opens at column 0 starts its own block and the block ends
 *     at the matching closing fence; blank lines inside a fence never split.
 */

export interface MdBlock {
  key: string;
  text: string;
  /** False only for the trailing block of a message that is still
   *  streaming — e.g. a code fence whose closing line hasn't arrived. */
  closed: boolean;
  /** Info string of the fence when the block is a code fence ("" otherwise). */
  fenceLang: string;
}

const FENCE_RE = /^( {0,3})(`{3,}|~{3,})\s*([^\s`]*)/;

interface OpenFence {
  char: string;
  len: number;
  topLevel: boolean;
}

export function splitMarkdownBlocks(src: string, final: boolean): MdBlock[] {
  const lines = src.split("\n");
  const blocks: MdBlock[] = [];
  let buf: string[] = [];
  let bufLang = "";
  let fence: OpenFence | null = null;
  let blankPending = false;

  const push = (closed: boolean) => {
    // Drop trailing blank lines; they carry no content.
    while (buf.length && buf[buf.length - 1].trim() === "") buf.pop();
    if (buf.length) {
      blocks.push({ key: `b${blocks.length}`, text: buf.join("\n"), closed, fenceLang: bufLang });
    }
    buf = [];
    bufLang = "";
  };

  for (const line of lines) {
    if (fence) {
      buf.push(line);
      const m = line.match(FENCE_RE);
      if (m && m[2][0] === fence.char && m[2].length >= fence.len && line.slice(m[0].length).trim() === "" && !m[3]) {
        const wasTopLevel = fence.topLevel;
        fence = null;
        if (wasTopLevel) push(true);
      }
      continue;
    }

    if (line.trim() === "") {
      if (buf.length) blankPending = true;
      continue;
    }

    const m = line.match(FENCE_RE);
    const startsAtColumn0 = !/^\s/.test(line);

    if (blankPending) {
      blankPending = false;
      if (startsAtColumn0) push(true);
      else buf.push("");
    }

    if (m) {
      const topLevel = m[1] === "";
      if (topLevel) {
        push(true);
        bufLang = m[3] ?? "";
      }
      fence = { char: m[2][0], len: m[2].length, topLevel };
    }
    buf.push(line);
  }

  // The trailing block (including an unclosed fence) is still being written
  // unless the message is final.
  push(final);
  // A closing fence on the very last line may itself be incomplete
  // ("```" can still grow into "```py"), so keep that block open too.
  if (!final && blocks.length && !src.endsWith("\n")) blocks[blocks.length - 1].closed = false;
  return blocks;
}
