"""Marker-block helpers for KnowledgeInternalizer write-backs.

A marker block looks like:

    <!-- easybci-internalization id="ABC12345" date="2026-06-15" source="tavily:doi:..." confidence="high" -->
    ...body...
    <!-- /easybci-internalization id="ABC12345" -->

The ID is a SHA256(skill_path|source_url|content) truncated to 8 hex chars —
stable across replays so a re-internalisation of the same evidence doesn't
duplicate the block.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Tuple


def compute_internalization_id(*, skill_path: str, source_url: str, content: str) -> str:
    raw = f"{skill_path}|{source_url}|{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def wrap_with_marker(
    *,
    body: str,
    internalization_id: str,
    date_iso: str,
    source: str,
    confidence: str,
) -> str:
    open_tag = (
        f'<!-- easybci-internalization id="{internalization_id}" '
        f'date="{date_iso}" source="{source}" confidence="{confidence}" -->'
    )
    close_tag = f'<!-- /easybci-internalization id="{internalization_id}" -->'
    return f"{open_tag}\n{body.rstrip()}\n{close_tag}"


def _block_re(internalization_id: str) -> re.Pattern[str]:
    esc = re.escape(internalization_id)
    pattern = (
        r'<!--\s*easybci-internalization\s+id="' + esc + r'"[^>]*-->'
        r'(?P<body>.*?)'
        r'<!--\s*/easybci-internalization\s+id="' + esc + r'"\s*-->'
    )
    return re.compile(pattern, re.DOTALL)


def find_marker_block(doc: str, *, internalization_id: str) -> Optional[str]:
    m = _block_re(internalization_id).search(doc)
    if m is None:
        return None
    return m.group("body").strip()


def strip_marker_block(doc: str, *, internalization_id: str) -> Tuple[str, int]:
    """Return (new_doc, removed_lines_count)."""
    m = _block_re(internalization_id).search(doc)
    if m is None:
        return doc, 0
    removed = m.group(0).count("\n") + 1
    new_doc = doc[: m.start()] + doc[m.end() :]
    while "\n\n\n" in new_doc:
        new_doc = new_doc.replace("\n\n\n", "\n\n")
    return new_doc, removed
