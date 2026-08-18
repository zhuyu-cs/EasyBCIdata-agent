"""Typed I/O contract for operator steps.

Every EasyBCI operator function should consume / produce a dict shaped
exactly like ``OperatorIO``.  The dataclass is the canonical reference;
real call sites (step_cache, record_step_elapsed, pipeline.py) keep their
``Dict[str, Any]`` plumbing for backwards compatibility, but the field
names are pinned here so a typo like ``freq`` vs ``frequency`` shows up
at the codegen lint stage (``code_standard_check.py``).

Schema:

| Field        | Type                       | Required | Notes                                               |
|--------------|----------------------------|----------|-----------------------------------------------------|
| ``data``     | ``np.ndarray (C, T)``      | yes      | float32 or float64 only.                            |
| ``channels`` | ``List[str]``              | yes      | ``len(channels) == data.shape[0]``                  |
| ``frequency``| ``float``                  | yes      | sampling rate in Hz.                                |
| ``duration`` | ``float``                  | yes      | seconds; cached because ``data.shape[1]/frequency`` |
|              |                            |          | is a hot computation.                               |
| ``meta``     | ``Dict[str, Any]``         | yes      | operator-step state (cumulative).                   |
| ``elapsed_s``| ``Optional[float]``        | no       | filled by ``record_step_elapsed``.                  |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# Canonical key list — used by ``code_standard_check.py`` to flag typos.
OPERATOR_IO_REQUIRED_KEYS: tuple[str, ...] = (
    "data",
    "channels",
    "frequency",
    "duration",
    "meta",
)
OPERATOR_IO_OPTIONAL_KEYS: tuple[str, ...] = (
    "elapsed_s",
)
OPERATOR_IO_ALL_KEYS: frozenset[str] = frozenset(
    OPERATOR_IO_REQUIRED_KEYS + OPERATOR_IO_OPTIONAL_KEYS
)


@dataclass
class OperatorIO:
    """Strict schema for operator step input / output.

    The dataclass exists so type-checkers / IDEs catch typos.  Production
    operators still accept plain dicts (because ``step_cache`` /
    ``preprocess`` predate this contract) — see ``OPERATOR_IO_ALL_KEYS``
    for the runtime-pinned key set.
    """
    data: np.ndarray
    channels: List[str]
    frequency: float
    duration: float
    meta: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "data": self.data,
            "channels": list(self.channels),
            "frequency": float(self.frequency),
            "duration": float(self.duration),
            "meta": dict(self.meta),
        }
        if self.elapsed_s is not None:
            out["elapsed_s"] = float(self.elapsed_s)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperatorIO":
        return cls(
            data=d["data"],
            channels=list(d.get("channels") or []),
            frequency=float(d.get("frequency", 0.0)),
            duration=float(
                d.get("duration", 0.0)
                or (d["data"].shape[-1] / float(d.get("frequency") or 1.0))
            ),
            meta=dict(d.get("meta") or {}),
            elapsed_s=d.get("elapsed_s"),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if the contract is violated.

        Operators don't have to call this — it exists for tests and for
        the codegen lint to invoke on the entry / exit boundaries.
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("OperatorIO.data must be np.ndarray")
        if self.data.ndim != 2:
            raise ValueError(
                f"OperatorIO.data must be 2-D (n_channels, n_times); "
                f"got shape {self.data.shape}"
            )
        if self.data.dtype.kind != "f":
            raise ValueError(
                f"OperatorIO.data dtype must be floating; got {self.data.dtype}"
            )
        if len(self.channels) != self.data.shape[0]:
            raise ValueError(
                f"len(channels) {len(self.channels)} != n_channels "
                f"{self.data.shape[0]}"
            )
        if self.frequency <= 0:
            raise ValueError(f"OperatorIO.frequency must be > 0; got {self.frequency}")
        if self.duration < 0:
            raise ValueError(f"OperatorIO.duration must be >= 0; got {self.duration}")
