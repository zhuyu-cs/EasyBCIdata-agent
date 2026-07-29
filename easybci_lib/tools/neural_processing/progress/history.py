"""~/.easybci/progress_history.jsonl — per-operator elapsed-time samples.

Append-only jsonl; per-operator LRU cap (default 200) enforced lazily on query
to avoid sync rewrite latency on every append. Compaction happens when
`query()` reads more than the cap for one operator — rewrites the file with
only the newest `cap` entries per operator.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LRU_CAP_PER_OPERATOR = 200


@dataclass
class ProgressHistoryEntry:
    stage: str
    operator: str
    fingerprint_hash: str
    elapsed_s: float
    n_channels: int
    duration_s: float
    timestamp: str  # ISO 8601 UTC
    intent_kind: Optional[str] = None


class ProgressHistoryStore:
    def __init__(self, path: Path, *, lru_cap_per_operator: int = _DEFAULT_LRU_CAP_PER_OPERATOR) -> None:
        self.path = path
        self._cap = lru_cap_per_operator
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: ProgressHistoryEntry) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("progress_history append failed: %s", exc)

    def query(
        self,
        *,
        operator: Optional[str] = None,
        fingerprint_hash: Optional[str] = None,
        intent_kind: Optional[str] = None,
    ) -> Iterator[ProgressHistoryEntry]:
        rows = self._read_all()
        if operator is not None:
            rows = [r for r in rows if r.operator == operator]
        if fingerprint_hash is not None:
            rows = [r for r in rows if r.fingerprint_hash == fingerprint_hash]
        if intent_kind is not None:
            rows = [r for r in rows if r.intent_kind == intent_kind]
        if operator is not None and len(rows) > self._cap:
            rows = rows[-self._cap:]
            self._compact_if_needed(operator)
        for r in rows:
            yield r

    def _read_all(self) -> List[ProgressHistoryEntry]:
        if not self.path.exists():
            return []
        out: List[ProgressHistoryEntry] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        known = {
                            "stage", "operator", "fingerprint_hash", "elapsed_s",
                            "n_channels", "duration_s", "timestamp", "intent_kind",
                        }
                        d_filtered = {k: v for k, v in d.items() if k in known}
                        out.append(ProgressHistoryEntry(**d_filtered))
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("skipping corrupted progress_history line: %s", exc)
                        continue
        except OSError as exc:
            logger.warning("progress_history read failed: %s", exc)
        return out

    def _compact_if_needed(self, operator: str) -> None:
        all_rows = self._read_all()
        by_op: dict[str, List[ProgressHistoryEntry]] = {}
        for r in all_rows:
            by_op.setdefault(r.operator, []).append(r)
        compacted: List[ProgressHistoryEntry] = []
        for op_rows in by_op.values():
            compacted.extend(op_rows[-self._cap:])
        compacted.sort(key=lambda r: r.timestamp)
        try:
            tmp = self.path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in compacted:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("progress_history compaction failed: %s", exc)


def record_intent_elapsed(
    *,
    kind: Optional[str],
    elapsed_s: float,
    fingerprint: Optional[str] = None,
    operator: Optional[str] = None,
) -> None:
    """Append one turn-scope elapsed sample to ~/.easybci/progress_history.jsonl.

    Failures are non-fatal — this is observability, not correctness. Empty or
    None ``kind`` is a no-op. ``operator`` mirrors ``kind`` when omitted so
    legacy code that filters by operator still finds turn-scope samples.
    """
    if not kind:
        return
    try:
        import datetime
        from easybci_lib.constants import get_easybci_home

        store = ProgressHistoryStore(path=get_easybci_home() / "progress_history.jsonl")
        store.append(ProgressHistoryEntry(
            stage="turn",
            operator=operator or kind,
            fingerprint_hash=fingerprint or "",
            elapsed_s=float(elapsed_s),
            n_channels=0,
            duration_s=0.0,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            intent_kind=kind,
        ))
    except Exception:
        logger.debug("record_intent_elapsed failed", exc_info=True)
