"""Pipeline event transform tracking — maintain event alignment through processing steps.

When a pipeline applies transforms that change the time axis (resample, crop, trim),
external event tables must be updated accordingly. This module:
- Tracks all time-axis transforms applied during pipeline execution
- Provides rescale_events() to update external event timestamps post-pipeline
- Records the transform log in pipeline output for reproducibility

Solves the problem: keeping external event timestamps aligned with the data after resample.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimeTransform:
    """A single time-axis transformation."""
    step_name: str
    transform_type: str  # "resample", "crop", "trim", "offset"
    params: Dict[str, Any] = field(default_factory=dict)

    # For resample
    original_freq: float = 0.0
    target_freq: float = 0.0

    # For crop/trim
    time_offset: float = 0.0  # seconds removed from start
    new_duration: float = 0.0

    # For sample-based events
    sample_scale_factor: float = 1.0


@dataclass
class EventTransformLog:
    """Accumulated time transforms from a pipeline run.

    Usage:
        log = EventTransformLog()
        # During pipeline:
        log.record_resample(original_freq=512.0, target_freq=128.0)
        log.record_crop(start_sec=10.0, end_sec=60.0)
        # After pipeline:
        corrected_events = log.apply_to_events(original_events)
    """
    transforms: List[TimeTransform] = field(default_factory=list)
    original_frequency: float = 0.0
    current_frequency: float = 0.0
    cumulative_time_offset: float = 0.0  # total seconds removed from start
    cumulative_sample_scale: float = 1.0  # total sample index scale factor

    def record_resample(self, original_freq: float, target_freq: float, step_name: str = "resample") -> None:
        """Record a resampling operation."""
        scale = target_freq / original_freq
        self.transforms.append(TimeTransform(
            step_name=step_name,
            transform_type="resample",
            original_freq=original_freq,
            target_freq=target_freq,
            sample_scale_factor=scale,
        ))
        if self.original_frequency == 0.0:
            self.original_frequency = original_freq
        self.current_frequency = target_freq
        self.cumulative_sample_scale *= scale

    def record_crop(self, start_sec: float, end_sec: float, step_name: str = "crop") -> None:
        """Record a crop/trim operation (removes data from start/end)."""
        duration = end_sec - start_sec
        self.transforms.append(TimeTransform(
            step_name=step_name,
            transform_type="crop",
            time_offset=start_sec,
            new_duration=duration,
            params={"start_sec": start_sec, "end_sec": end_sec},
        ))
        self.cumulative_time_offset += start_sec

    def record_trim(self, start_sec: float = 0.0, end_sec: float = 0.0, step_name: str = "trim") -> None:
        """Record trimming from start and/or end."""
        self.transforms.append(TimeTransform(
            step_name=step_name,
            transform_type="trim",
            time_offset=start_sec,
            params={"trim_start": start_sec, "trim_end": end_sec},
        ))
        self.cumulative_time_offset += start_sec

    def record_offset(self, offset_sec: float, step_name: str = "offset") -> None:
        """Record a time offset adjustment."""
        self.transforms.append(TimeTransform(
            step_name=step_name,
            transform_type="offset",
            time_offset=offset_sec,
        ))
        self.cumulative_time_offset += offset_sec

    def apply_to_events(
        self,
        events: List[Dict[str, Any]],
        time_unit: str = "seconds",
    ) -> List[Dict[str, Any]]:
        """Apply accumulated transforms to an external event list.

        Parameters
        ----------
        events : list of dict
            Each dict must have "onset" (float). Optional: "duration", "start", "end".
        time_unit : str
            Unit of event timestamps: "seconds" or "samples".

        Returns
        -------
        Corrected events list (new list, original not modified).
        """
        corrected = []
        for event in events:
            new_event = dict(event)

            if time_unit == "samples":
                # Convert sample-based timestamps
                onset = float(new_event.get("onset", new_event.get("start", 0)))
                onset = (onset * self.cumulative_sample_scale) - (
                    self.cumulative_time_offset * self.current_frequency
                )
                new_event["onset"] = onset
                if "start" in new_event:
                    new_event["start"] = onset

                if "end" in new_event:
                    end = float(new_event["end"])
                    end = (end * self.cumulative_sample_scale) - (
                        self.cumulative_time_offset * self.current_frequency
                    )
                    new_event["end"] = end

                if "duration" in new_event and "onset" in event:
                    dur = float(new_event["duration"])
                    new_event["duration"] = dur * self.cumulative_sample_scale

            else:  # seconds
                onset = float(new_event.get("onset", new_event.get("start", 0)))
                onset = onset - self.cumulative_time_offset
                new_event["onset"] = onset
                if "start" in new_event:
                    new_event["start"] = onset

                if "end" in new_event:
                    end = float(new_event["end"])
                    new_event["end"] = end - self.cumulative_time_offset

                # Duration is unaffected by crop/resample (it's still in seconds)

            corrected.append(new_event)

        return corrected

    def apply_to_timestamps(
        self,
        timestamps: np.ndarray,
        time_unit: str = "seconds",
    ) -> np.ndarray:
        """Apply transforms to a numpy array of timestamps.

        Parameters
        ----------
        timestamps : ndarray of float
        time_unit : "seconds" or "samples"

        Returns
        -------
        Corrected timestamps array.
        """
        result = timestamps.copy().astype(np.float64)

        if time_unit == "samples":
            result = result * self.cumulative_sample_scale
            result = result - (self.cumulative_time_offset * self.current_frequency)
        else:
            result = result - self.cumulative_time_offset

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the transform log for storage in pipeline output."""
        return {
            "original_frequency": self.original_frequency,
            "current_frequency": self.current_frequency,
            "cumulative_time_offset_s": self.cumulative_time_offset,
            "cumulative_sample_scale": self.cumulative_sample_scale,
            "n_transforms": len(self.transforms),
            "transforms": [
                {
                    "step": t.step_name,
                    "type": t.transform_type,
                    "params": t.params,
                    "time_offset": t.time_offset,
                    "sample_scale": t.sample_scale_factor,
                }
                for t in self.transforms
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventTransformLog":
        """Reconstruct from serialized dict."""
        log = cls()
        log.original_frequency = data.get("original_frequency", 0.0)
        log.current_frequency = data.get("current_frequency", 0.0)
        log.cumulative_time_offset = data.get("cumulative_time_offset_s", 0.0)
        log.cumulative_sample_scale = data.get("cumulative_sample_scale", 1.0)

        for t_data in data.get("transforms", []):
            transform = TimeTransform(
                step_name=t_data.get("step", "unknown"),
                transform_type=t_data.get("type", "unknown"),
                params=t_data.get("params", {}),
                time_offset=t_data.get("time_offset", 0.0),
                sample_scale_factor=t_data.get("sample_scale", 1.0),
            )
            log.transforms.append(transform)

        return log


def rescale_events(
    events: List[Dict[str, Any]],
    original_freq: float,
    target_freq: float,
    time_unit: str = "seconds",
    crop_start: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convenience function to rescale events after resample + optional crop.

    Parameters
    ----------
    events : list of event dicts
    original_freq : float — original sampling rate
    target_freq : float — new sampling rate after resample
    time_unit : str — "seconds" or "samples"
    crop_start : float — seconds removed from start (0 if no crop)

    Returns
    -------
    New list of events with corrected timestamps.
    """
    log = EventTransformLog()
    if original_freq != target_freq:
        log.record_resample(original_freq, target_freq)
    if crop_start > 0:
        log.record_crop(crop_start, crop_start + 1e9)  # end doesn't matter for offset

    return log.apply_to_events(events, time_unit=time_unit)
