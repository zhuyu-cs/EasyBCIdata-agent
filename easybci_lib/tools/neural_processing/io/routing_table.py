"""Routing table — multi-input single source of truth.

Lives at ``<work_dir>/middle_process/inputs_routing.json``. Every input file
processed in this work_dir has exactly one entry here, with its
``(subject_id, session_id, stem_safe, file_id)`` coordinate. Downstream
codegen / dispatcher / contract_check read from this file — no stage may
re-derive ``(subject_id, session_id)`` from ``Path(raw).stem``.

The table is written by :func:`deep_inspect` and updated via
:func:`upsert_routing_entry` (idempotent by ``file_id``). Single-input
work_dirs may omit the routing table entirely — downstream tools fall back
to legacy single-file behavior when the file is missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


ROUTING_TABLE_FILENAME = "inputs_routing.json"
ROUTING_TABLE_SCHEMA_VERSION = "1"


class RoutingConflictError(ValueError):
    """Raised when an upsert would create two entries sharing
    ``(subject_id, session_id, stem_safe)`` under different ``file_id``s.

    That triple is the routing coordinate downstream stages use to write
    files — duplicates would silently clobber each other.
    """


@dataclass
class RoutingEntry:
    """One input file's routing coordinates.

    All path-like fields use forward slashes; ``inspection_report_path`` is
    relative to ``<work_dir>/`` so the table remains portable when the
    work_dir is moved.
    """

    data_path: str
    stem_safe: str
    sha256_1mb: str
    file_id: str
    subject_id: str
    session_id: str
    identity_source: str
    identity_confidence: float
    inspection_report_path: str
    events_path: Optional[str] = None
    override_script: Optional[str] = None
    # Estimated peak-processing footprint (MB) from the recipe-aware model in
    # memory_strategy.estimate_peak_mb. Recorded once at routing time so the
    # batch scheduler and the cross-instance memory gate reuse it without
    # re-reading the inspection report. None when metadata was too thin.
    peak_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoutingEntry":
        return cls(
            data_path=str(d["data_path"]),
            stem_safe=str(d["stem_safe"]),
            sha256_1mb=str(d.get("sha256_1mb") or ""),
            file_id=str(d["file_id"]),
            subject_id=str(d["subject_id"]),
            session_id=str(d["session_id"]),
            identity_source=str(d.get("identity_source") or "fallback"),
            identity_confidence=float(d.get("identity_confidence") or 0.0),
            inspection_report_path=str(d.get("inspection_report_path") or ""),
            events_path=(str(d["events_path"]) if d.get("events_path") else None),
            override_script=(
                str(d["override_script"]) if d.get("override_script") else None
            ),
            peak_mb=(float(d["peak_mb"]) if d.get("peak_mb") is not None else None),
        )


@dataclass
class RoutingTable:
    schema_version: str = ROUTING_TABLE_SCHEMA_VERSION
    generated_at: str = ""
    inputs: list[RoutingEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "inputs": [e.to_dict() for e in self.inputs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoutingTable":
        return cls(
            schema_version=str(d.get("schema_version") or ROUTING_TABLE_SCHEMA_VERSION),
            generated_at=str(d.get("generated_at") or ""),
            inputs=[RoutingEntry.from_dict(e) for e in d.get("inputs") or []],
        )

    def find_by_file_id(self, file_id: str) -> Optional[RoutingEntry]:
        for e in self.inputs:
            if e.file_id == file_id:
                return e
        return None


def routing_table_path(work_dir: Path | str) -> Path:
    """Return the absolute path of the routing table inside ``work_dir``."""
    return Path(work_dir) / "middle_process" / ROUTING_TABLE_FILENAME


def stem_safe(raw_path: Path | str) -> str:
    """Canonical filename stem normalization.

    Single source of truth for downstream filename derivation: every
    ``*_preprocessed.nwb`` / ``*_epochs.pkl`` / figure name uses this. The
    rule is exactly ``Path(p).stem.replace(" ", "_")`` — minimalist on
    purpose so a human can predict the output name from the raw filename.
    """
    return Path(raw_path).stem.replace(" ", "_")


def load_routing_table(work_dir: Path | str) -> Optional[RoutingTable]:
    """Read ``inputs_routing.json`` from ``work_dir/middle_process/``.

    Returns ``None`` when the file does not exist. Raises ``ValueError``
    when the file exists but is unparseable — that's a hard error the
    caller should surface, not silently swallow.
    """
    p = routing_table_path(work_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"routing table at {p} is not valid JSON: {exc}") from exc
    return RoutingTable.from_dict(data)


def save_routing_table(work_dir: Path | str, table: RoutingTable) -> Path:
    """Write the routing table atomically to ``work_dir/middle_process/``."""
    import datetime as _dt

    p = routing_table_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not table.generated_at:
        table.generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(table.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def upsert_routing_entry(
    work_dir: Path | str,
    entry: RoutingEntry,
) -> RoutingTable:
    """Insert or replace ``entry`` (matched by ``file_id``) in the routing table.

    Idempotency: re-running ``deep_inspect`` on the same file replaces its
    existing entry rather than appending a duplicate.

    Conflict rule: if the new entry's ``(subject_id, session_id, stem_safe)``
    triple matches an existing entry with a DIFFERENT ``file_id``, raises
    ``RoutingConflictError`` — that triple is the routing coordinate
    downstream stages use to write files, so duplicates would clobber.

    Returns the updated table after persistence.
    """
    try:
        table = load_routing_table(work_dir) or RoutingTable()
    except ValueError as exc:
        logger.warning(
            "routing table at %s is corrupt — rebuilding from empty: %s",
            routing_table_path(work_dir), exc,
        )
        table = RoutingTable()

    triple = (entry.subject_id, entry.session_id, entry.stem_safe)
    for existing in table.inputs:
        if existing.file_id == entry.file_id:
            continue
        if (existing.subject_id, existing.session_id, existing.stem_safe) == triple:
            raise RoutingConflictError(
                f"routing conflict: file_id={entry.file_id!r} and "
                f"file_id={existing.file_id!r} both map to "
                f"sub={triple[0]!r} ses={triple[1]!r} stem={triple[2]!r}. "
                "Two different files cannot share the same output coordinate."
            )

    table.inputs = [e for e in table.inputs if e.file_id != entry.file_id]
    table.inputs.append(entry)
    save_routing_table(work_dir, table)
    return table
