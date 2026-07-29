"""StageProgress + StagedProgressTracker."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .estimator import EtaEstimator

logger = logging.getLogger(__name__)

StageName = Literal["plan", "codegen", "preprocess", "qc"]
ConfidenceLevel = Literal["high", "medium", "low", "unknown"]

_VALID_STAGES: tuple[str, ...] = ("plan", "codegen", "preprocess", "qc")
_VALID_CONFIDENCE: tuple[str, ...] = ("high", "medium", "low", "unknown")


@dataclass(frozen=True)
class StageProgress:
    """Immutable per-tick snapshot of stage progress.

    Frozen because consumers (SSE serialiser, WebUI hook) treat it as a value
    object — any change creates a new instance via `tracker.tick(...)`.
    """
    stage: StageName
    stage_index: int
    stage_total: int
    sub_step: str
    sub_index: int
    sub_total: int
    percent: Optional[float]
    eta_seconds: Optional[int]
    confidence: ConfidenceLevel
    elapsed_seconds: int
    heartbeat_ts: float

    def __post_init__(self) -> None:
        if self.stage not in _VALID_STAGES:
            raise ValueError(f"invalid stage: {self.stage!r} (expected one of {_VALID_STAGES})")
        if self.confidence not in _VALID_CONFIDENCE:
            raise ValueError(f"invalid confidence: {self.confidence!r}")
        if self.percent is not None and not (0.0 <= self.percent <= 100.0):
            raise ValueError(f"percent out of [0,100]: {self.percent}")

    def to_payload(self) -> Dict[str, Any]:
        """JSON-serialisable dict for SSE / status_callback.

        Includes ``scope: "stage"`` so frontends can dispatch symmetrically
        with turn-scope progress events (which use ``scope: "turn"``).
        """
        return {
            "scope": "stage",
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "sub_step": self.sub_step,
            "sub_index": self.sub_index,
            "sub_total": self.sub_total,
            "percent": self.percent,
            "eta_seconds": self.eta_seconds,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "heartbeat_ts": self.heartbeat_ts,
        }


_STAGE_ORDER: Sequence[StageName] = ("plan", "codegen", "preprocess", "qc")


class StagedProgressTracker:
    """Owns the four-stage progress lifecycle for one session.

    Construction-time invariants:
    - exactly one `on_progress` callback; called synchronously on every emit
    - stages can be started in any order; each start_stage replaces active stage
    """

    def __init__(
        self,
        on_progress: Callable[[StageProgress], None],
        *,
        clock: Callable[[], float] = time.time,
        estimator: Optional["EtaEstimator"] = None,
    ) -> None:
        self._on_progress = on_progress
        self._clock = clock
        self._estimator = estimator
        self._active_stage: Optional[StageName] = None
        self._stage_started_at: float = 0.0
        self._sub_step: str = ""
        self._sub_index: int = 0
        self._sub_total: int = 0
        self._last_eta: Optional[int] = None
        self._last_confidence: ConfidenceLevel = "unknown"
        self._last_percent: Optional[float] = None

    def start_stage(
        self,
        stage: StageName,
        *,
        sub_total: Optional[int] = None,
        operator: Optional[str] = None,
        fingerprint_hash: Optional[str] = None,
    ) -> None:
        if stage not in _STAGE_ORDER:
            raise ValueError(f"invalid stage: {stage!r}")
        self._active_stage = stage
        self._stage_started_at = self._clock()
        self._sub_step = ""
        self._sub_index = 0
        self._sub_total = sub_total or 0
        self._last_percent = None
        self._last_eta = None
        self._last_confidence = "unknown"

        if self._estimator is not None and operator and fingerprint_hash:
            from .estimator import EstimationIntent  # noqa: PLC0415

            intent = EstimationIntent(
                kind="preprocess_step",
                operator=operator,
                fingerprint=fingerprint_hash,
            )
            try:
                est = self._estimator.estimate(intent)
            except Exception as exc:
                logger.debug("estimator.estimate failed: %s", exc)
                est = None
            if est is not None:
                self._last_eta = est.seconds
                self._last_confidence = est.confidence

        self._emit()

    def tick(
        self,
        *,
        sub_step: Optional[str] = None,
        sub_index: Optional[int] = None,
        percent: Optional[float] = None,
        eta_seconds: Optional[int] = None,
        confidence: Optional[ConfidenceLevel] = None,
    ) -> StageProgress:
        if self._active_stage is None:
            raise RuntimeError("no active stage; call start_stage() first")
        if sub_step is not None:
            self._sub_step = sub_step
        if sub_index is not None:
            self._sub_index = sub_index
        if percent is not None:
            self._last_percent = percent
        if eta_seconds is not None:
            self._last_eta = eta_seconds
        if confidence is not None:
            self._last_confidence = confidence
        return self._emit()

    def end_stage(self) -> None:
        self._active_stage = None

    @property
    def active_stage(self) -> Optional[StageName]:
        """Public read-only access for HeartbeatDaemon."""
        return self._active_stage

    def _emit(self) -> StageProgress:
        assert self._active_stage is not None
        sp = StageProgress(
            stage=self._active_stage,
            stage_index=_STAGE_ORDER.index(self._active_stage),
            stage_total=len(_STAGE_ORDER),
            sub_step=self._sub_step,
            sub_index=self._sub_index,
            sub_total=self._sub_total,
            percent=self._last_percent,
            eta_seconds=self._last_eta,
            confidence=self._last_confidence,
            elapsed_seconds=int(self._clock() - self._stage_started_at),
            heartbeat_ts=self._clock(),
        )
        self._on_progress(sp)
        return sp
