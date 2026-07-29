"""Unified streaming tag filter for reasoning/tool-call XML suppression.

Single source of truth for tag definitions and filtering logic, shared by
cli.py (streaming + final display) and run_agent.py (final content cleanup).
"""

import re
from typing import Callable, Optional

# ─── Tag Definitions ────────────────────────────────────────────────────────

REASONING_TAGS = (
    "REASONING_SCRATCHPAD",
    "think",
    "thinking",
    "reasoning",
    "thought",
)

TOOL_CALL_TAGS = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)

_OPEN_TAGS = tuple(f"<{t}>" for t in REASONING_TAGS)
_CLOSE_TAGS = tuple(f"</{t}>" for t in REASONING_TAGS)


# ─── Static Stripping (Final Content) ──────────────────────────────────────

def strip_reasoning_tags(text: str) -> str:
    """Remove reasoning/thinking and tool-call XML blocks from text.

    Handles:
      * Closed pairs ``<tag>...</tag>`` (case-insensitive, multi-line).
      * Unterminated open tags at block boundaries (start or after newline).
      * Stray orphan close tags.
      * Tool-call XML blocks (tool_call, function_calls, etc.).
      * Gemma-style ``<function name="...">...</function>`` (boundary-gated).

    This is the canonical implementation used by both cli.py and run_agent.py.
    """
    if not text:
        return ""
    content = text

    # 1. Closed tag pairs — case-insensitive
    for tag in REASONING_TAGS:
        content = re.sub(
            rf"<{tag}>.*?</{tag}>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # 1b. Tool-call XML blocks
    for tc_tag in TOOL_CALL_TAGS:
        content = re.sub(
            rf"<{tc_tag}\b[^>]*>.*?</{tc_tag}>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # 1c. <function name="...">...</function> — boundary + attribute gated
    content = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Unterminated reasoning block — open tag at block boundary with no close
    content = re.sub(
        r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Stray orphan open/close reasoning tags
    content = re.sub(
        r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )

    # 3b. Stray tool-call close tags
    content = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )

    return content.strip()


# ─── Streaming Filter (Token-by-Token) ─────────────────────────────────────

class StreamTagFilter:
    """Stateful streaming filter that suppresses reasoning/tool-call blocks.

    Feed streaming tokens via ``feed()``; returns text safe to emit.
    Reasoning content is either discarded or routed to an optional callback.

    States:
      NORMAL — passing tokens through (with partial-tag buffering)
      IN_REASONING — suppressing content until close tag found
    """

    def __init__(self, on_reasoning: Optional[Callable[[str], None]] = None):
        self._on_reasoning = on_reasoning
        self.reset()

    def reset(self) -> None:
        """Reset state for a new stream."""
        self._buffer = ""
        self._in_reasoning = False
        self._last_was_newline = True

    def feed(self, text: str) -> str:
        """Feed streaming tokens. Returns text to emit (may be empty)."""
        self._buffer += text
        return self._drain()

    def flush(self) -> str:
        """Flush any buffered content at end of stream.

        If still in a reasoning block at stream end, it was likely a false
        positive (model mentioned a tag in prose but never closed it).
        Recover the buffered content as regular text.
        """
        result = self._buffer
        self._buffer = ""
        self._in_reasoning = False
        return result

    def _drain(self) -> str:
        """Process buffer, potentially transitioning between states."""
        output = ""
        while self._buffer:
            cur_state = self._in_reasoning
            if not self._in_reasoning:
                chunk = self._process_normal()
                if chunk:
                    output += chunk
                elif self._in_reasoning != cur_state:
                    continue
                else:
                    break
            else:
                chunk = self._process_reasoning()
                if chunk:
                    output += chunk
                elif self._in_reasoning != cur_state:
                    continue
                else:
                    break
        return output

    def _process_normal(self) -> str:
        """Process buffer in NORMAL state. Look for reasoning open tags."""
        for i, tag in enumerate(_OPEN_TAGS):
            search_start = 0
            while True:
                idx = self._buffer.find(tag, search_start)
                if idx == -1:
                    break
                if self._is_block_boundary(idx):
                    preceding = self._buffer[:idx]
                    self._in_reasoning = True
                    self._buffer = self._buffer[idx + len(tag):]
                    if preceding:
                        self._last_was_newline = preceding.endswith("\n")
                    return preceding
                search_start = idx + 1

        # Check for partial tag at the end — hold it back
        safe = self._buffer
        for tag in _OPEN_TAGS:
            for j in range(1, len(tag)):
                if self._buffer.endswith(tag[:j]):
                    safe = self._buffer[:-j]
                    break
            if len(safe) < len(self._buffer):
                break

        if safe:
            self._last_was_newline = safe.endswith("\n")
            self._buffer = self._buffer[len(safe):]
            return safe
        return ""

    def _process_reasoning(self) -> str:
        """Process buffer in IN_REASONING state. Look for close tags."""
        for tag in _CLOSE_TAGS:
            idx = self._buffer.find(tag)
            if idx != -1:
                self._in_reasoning = False
                if self._on_reasoning:
                    inner = self._buffer[:idx]
                    if inner:
                        self._on_reasoning(inner)
                self._buffer = self._buffer[idx + len(tag):]
                return ""

        # No close tag yet — stream reasoning content if callback set,
        # keeping tail that could be partial close tag
        max_tag_len = max(len(t) for t in _CLOSE_TAGS)
        if len(self._buffer) > max_tag_len:
            if self._on_reasoning:
                safe_reasoning = self._buffer[:-max_tag_len]
                self._on_reasoning(safe_reasoning)
            self._buffer = self._buffer[-max_tag_len:]
        return ""

    def _is_block_boundary(self, idx: int) -> bool:
        """Check if position idx in buffer is at a block boundary."""
        if idx == 0:
            return self._last_was_newline
        preceding = self._buffer[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            return self._last_was_newline and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""
