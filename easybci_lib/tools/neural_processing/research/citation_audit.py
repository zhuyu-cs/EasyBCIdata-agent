"""Event-log jsonl for citation checks.

Mirrors the schema vocabulary of ``internalization_audit.jsonl``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

from .citation_checker import CitationCheckResult

logger = logging.getLogger(__name__)


class CitationAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: CitationCheckResult) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("citation_audit append failed: %s", exc)

    def read_all(self) -> List[CitationCheckResult]:
        out: List[CitationCheckResult] = []
        if not self.path.exists():
            return out
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(CitationCheckResult(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("skipping corrupted audit line: %s", exc)
        except OSError as exc:
            logger.warning("citation_audit read failed: %s", exc)
        return out
