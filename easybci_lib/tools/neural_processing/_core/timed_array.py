"""Time-aligned array container — the one core abstraction worth keeping.

Why this exists: neural data needs frequency + start time attached to the
array for correct time-slicing and multi-source alignment. Without it,
every function needs (data, freq, start) triplets passed around.

Design: pure numpy, no torch, no pydantic. Just a plain class.
"""

from typing import Any, Optional, Literal, TypeVar, overload
import numpy as np

_TA = TypeVar("_TA", bound="TimedArray")


class Frequency(float):
    """Sampling rate in Hz. Provides seconds ↔ sample index conversion.

    >>> freq = Frequency(256.0)
    >>> freq.to_ind(1.0)   # 1 second → 256 samples
    256
    >>> freq.to_sec(128)   # 128 samples → 0.5 seconds
    0.5
    """

    @overload
    def to_ind(self, seconds: float) -> int: ...
    @overload
    def to_ind(self, seconds: np.ndarray) -> np.ndarray: ...

    def to_ind(self, seconds):
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @overload
    def to_sec(self, index: int) -> float: ...
    @overload
    def to_sec(self, index: np.ndarray) -> np.ndarray: ...

    def to_sec(self, index):
        return index / self


class TimedArray:
    """Numpy array annotated with sampling frequency and start time.

    Time is the LAST dimension when frequency > 0.
    Supports time-aligned overlap extraction and incremental addition (+=).

    This is neuralset's best abstraction — it makes time slicing trivial
    across heterogeneous recordings.
    """

    __slots__ = ("frequency", "start", "duration", "data", "header",
                 "aggregation", "_overlap_counts")

    def __init__(
        self,
        data: np.ndarray,
        frequency: float,
        start: float = 0.0,
        header: Optional[dict] = None,
    ):
        self.frequency = Frequency(frequency)
        self.start = start
        self.data = data
        self.header = header or {}
        self.aggregation = "sum"
        self._overlap_counts: Optional[np.ndarray] = None

        if frequency > 0:
            self.duration = self.frequency.to_sec(data.shape[-1])
        else:
            self.duration = 0.0

    @classmethod
    def empty(
        cls,
        frequency: float,
        start: float,
        duration: float,
        n_channels: int = 0,
        aggregation: Literal["sum", "mean"] = "sum",
    ) -> "TimedArray":
        """Create an empty TimedArray for incremental accumulation."""
        freq = Frequency(frequency)
        n_samples = max(1, freq.to_ind(duration))
        shape = (n_channels, n_samples) if n_channels else (n_samples,)
        ta = cls(data=np.zeros(shape, dtype=np.float32), frequency=frequency, start=start)
        ta.aggregation = aggregation
        if aggregation == "mean":
            ta._overlap_counts = np.zeros(n_samples, dtype=int)
        return ta

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1] if self.frequency > 0 else 0

    def overlap(self: _TA, start: float, duration: float) -> _TA:
        """Extract sub-array overlapping [start, start+duration].

        Returns zero-length array if no overlap.
        """
        if not self.frequency:
            # Static array — return full data as-is since time-slicing is meaningless
            return self

        overlap_start = max(start, self.start)
        overlap_stop = min(start + duration, self.end)

        if overlap_stop <= overlap_start:
            cls = type(self)
            return cls(
                data=self.data[..., 0:0],
                frequency=float(self.frequency),
                start=start,
                header=self.header,
            )

        start_ind = self.frequency.to_ind(overlap_start - self.start)
        n_ind = self.frequency.to_ind(overlap_stop - overlap_start)
        n_ind = max(1, n_ind)
        start_ind = min(start_ind, self.data.shape[-1] - n_ind)
        start_ind = max(0, start_ind)

        cls = type(self)
        return cls(
            data=self.data[..., start_ind:start_ind + n_ind],
            frequency=float(self.frequency),
            start=self.frequency.to_sec(start_ind) + self.start,
            header=self.header,
        )

    def with_start(self: _TA, start: float) -> _TA:
        """Lightweight copy with a new start time (shares data memory)."""
        cls = type(self)
        return cls(
            data=self.data,
            frequency=float(self.frequency),
            start=start,
            header=self.header,
        )

    def __repr__(self) -> str:
        return (
            f"TimedArray(freq={float(self.frequency)}, start={self.start:.3f}, "
            f"dur={self.duration:.3f}, shape={self.data.shape})"
        )
