"""~/.easybci/internalization_audit.jsonl — append-only event log.

Coexists with `internalized_knowledge.json` (active-entries snapshot).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    event: str  # "internalize" | "revoke" | "auto_flag"
    internalization_id: str
    skill_path: str
    source_url: str
    content_excerpt: str
    timestamp_iso: str
    confidence: str
    revoke_reason: Optional[str] = None
    auto_flag_reason: Optional[str] = None


class InternalizationAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("internalization_audit append failed: %s", exc)

    def read_all(self) -> List[AuditEvent]:
        out: List[AuditEvent] = []
        if not self.path.exists():
            return out
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(AuditEvent(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("skipping corrupted audit line: %s", exc)
        except OSError as exc:
            logger.warning("internalization_audit read failed: %s", exc)
        return out
