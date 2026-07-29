"""Inspection report schema — deep_inspect → planner hand-off artifact.

Written by deep_inspect, read by propose_pipeline / plan_pipeline /
generate_code. Lives at `<work_dir>/middle_process/inspection_report.json`
(legacy single-input shape) or `<work_dir>/middle_process/inspect/<file_id>/
inspection_report.json` (multi-input shape — one report per input).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from easybci_lib.tools.neural_processing.profile.identity_resolver import (
    RecordingIdentity,
)

logger = logging.getLogger(__name__)

# v1 → v2 added optional ``identity`` field carrying RecordingIdentity.
# v2 → v3 added optional ``file_id`` (sha256-of-first-1MB[:8]) so multi-input
# work_dirs can address per-file reports under ``middle_process/inspect/<file_id>/``.
# Older v1/v2 reports remain readable — identity / file_id are allowed to be
# None / absent.
SCHEMA_VERSION = "3"
_COMPATIBLE_SCHEMA_VERSIONS = {"1", "2", "3"}


@dataclass
class ChannelStat:
    name: str
    category: str
    variance: float
    mean: float
    std: float
    nan_pct: float
    inf_pct: float
    flat_pct: float
    spike_count: int


@dataclass
class Fingerprint:
    format: str
    modality: str
    n_channels: int
    sampling_freq_hz: float
    duration_s: float
    n_events: int
    event_types: list[str]


@dataclass
class ChannelSummary:
    must_drop: list[str]
    suggest_drop: list[str]
    bad_candidates_high_variance: list[str]
    bad_candidates_flat: list[str]
    bad_candidates_spike: list[str]


@dataclass
class PsdSummary:
    power_line_peak_hz: float | None
    power_line_peak_db_above_floor: float | None
    harmonics_detected_hz: list[float]
    low_freq_drift_below_1hz_present: bool
    high_freq_noise_above_40hz_present: bool


@dataclass
class ArtifactSummary:
    sample_pct: float
    blink_rate_per_min: float
    muscle_artifact_pct: float
    saturation_pct: float


@dataclass
class MemoryEstimate:
    preload_full_mb: float
    peak_processing_mb_estimate: float


@dataclass
class InspectionReport:
    generated_at: str
    data_path: str
    fingerprint: Fingerprint
    channel_stats: list[ChannelStat]
    channel_summary: ChannelSummary
    psd_summary: PsdSummary
    artifact_summary: ArtifactSummary
    memory_estimate: MemoryEstimate
    degraded: bool = False
    degraded_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    identity: Optional[RecordingIdentity] = None
    file_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "data_path": self.data_path,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "fingerprint": asdict(self.fingerprint),
            "channel_stats": [asdict(c) for c in self.channel_stats],
            "channel_summary": asdict(self.channel_summary),
            "psd_summary": asdict(self.psd_summary),
            "artifact_summary": asdict(self.artifact_summary),
            "memory_estimate": asdict(self.memory_estimate),
            "warnings": list(self.warnings),
            "identity": self.identity.to_dict() if self.identity else None,
            "file_id": self.file_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InspectionReport:
        version = d.get("schema_version")
        if version not in _COMPATIBLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"inspection_report schema_version={version!r} is "
                f"incompatible (expected one of {sorted(_COMPATIBLE_SCHEMA_VERSIONS)}). "
                "Re-run deep_inspect to regenerate."
            )
        identity_d = d.get("identity")
        identity = (
            RecordingIdentity.from_dict(identity_d)
            if isinstance(identity_d, dict) else None
        )
        return cls(
            schema_version=version,
            generated_at=d["generated_at"],
            data_path=d["data_path"],
            degraded=bool(d.get("degraded", False)),
            degraded_reason=d.get("degraded_reason"),
            fingerprint=Fingerprint(**d["fingerprint"]),
            channel_stats=[ChannelStat(**c) for c in d.get("channel_stats", [])],
            channel_summary=ChannelSummary(**d["channel_summary"]),
            psd_summary=PsdSummary(**d["psd_summary"]),
            artifact_summary=ArtifactSummary(**d["artifact_summary"]),
            memory_estimate=MemoryEstimate(**d["memory_estimate"]),
            warnings=list(d.get("warnings", [])),
            identity=identity,
            file_id=str(d.get("file_id") or ""),
        )


def save_inspection_report(report: InspectionReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_inspection_report(path: Path) -> InspectionReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InspectionReport.from_dict(data)


_FILE_ID_BYTES = 1 << 20  # 1 MiB head


def compute_file_id(path: Path | str) -> str:
    """sha256 of the first 1 MiB of ``path``, truncated to 8 hex chars.

    Stable across runs for the same file, cheap on large recordings, and
    short enough to use as a directory segment under ``middle_process/inspect/``.
    Returns ``""`` and logs at DEBUG when the file is unreadable — the caller
    is expected to fall back to another id (e.g. an md5 of the path string).
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(_FILE_ID_BYTES)
    except OSError as exc:
        logger.debug("compute_file_id(%s) failed: %s", p, exc)
        return ""
    return hashlib.sha256(head).hexdigest()[:8]
