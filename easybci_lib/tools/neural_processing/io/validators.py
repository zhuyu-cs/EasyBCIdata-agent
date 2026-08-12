"""Post-load data validation — catches structural problems immediately.

Called automatically after load_neural() to ensure the returned data dict
is well-formed before any processing begins. Catches issues that would
otherwise surface as cryptic errors mid-pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when loaded data has critical structural problems."""
    pass


@dataclass
class ValidationResult:
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_loaded_data(data_dict: Dict[str, Any]) -> ValidationResult:
    """Validate a loaded neural data dict for structural correctness.

    Checks:
    - Required keys present (data, frequency, channels)
    - Data is ndarray with valid dtype
    - Shape consistency: channels list matches data dimension
    - Frequency in valid range (0.1–100000 Hz)
    - Duration consistency with shape and frequency
    - No all-NaN channels
    - Data not entirely zero

    Returns ValidationResult. Raises DataValidationError only on
    critical issues that make the data completely unusable.
    """
    result = ValidationResult()

    # Required keys
    for key in ("data", "frequency", "channels"):
        if key not in data_dict:
            result.valid = False
            result.issues.append(f"Missing required key: '{key}'")

    if not result.valid:
        logger.warning("Data missing required keys: %s", result.issues)
        return result

    data = data_dict["data"]
    frequency = data_dict["frequency"]
    channels = data_dict["channels"]

    # Data type check
    if not isinstance(data, np.ndarray):
        if isinstance(data, list):
            # Spike data: list of arrays (valid for spike modality)
            result.warnings.append("Data is list (spike times) — skipping array checks")
            return result
        result.valid = False
        result.issues.append(f"Data must be ndarray, got {type(data).__name__}")
        logger.warning("Data must be ndarray, got %s", type(data).__name__)
        return result

    # Dtype check
    if data.dtype not in (np.float32, np.float64, np.float16):
        result.warnings.append(
            f"Data dtype is {data.dtype}, expected float32/64. May cause precision issues."
        )

    # Shape check
    if data.ndim < 1:
        result.valid = False
        result.issues.append("Data has 0 dimensions")
        logger.warning("Data has 0 dimensions")
        return result

    if data.ndim == 2:
        n_channels_data = data.shape[0]
        n_samples = data.shape[1]
    elif data.ndim == 1:
        n_channels_data = 1
        n_samples = data.shape[0]
    else:
        result.warnings.append(f"Unexpected data ndim={data.ndim}, expected 2")
        n_channels_data = data.shape[0]
        n_samples = data.shape[-1]

    # Channels list consistency
    if len(channels) != n_channels_data:
        result.valid = False
        result.issues.append(
            f"Channel count mismatch: {len(channels)} names vs {n_channels_data} in data"
        )

    # Frequency validation
    if not isinstance(frequency, (int, float)):
        result.valid = False
        result.issues.append(f"Frequency must be numeric, got {type(frequency).__name__}")
    elif frequency <= 0:
        result.valid = False
        result.issues.append(f"Frequency must be positive, got {frequency}")
    elif frequency < 0.1:
        result.warnings.append(f"Very low frequency: {frequency} Hz")
    elif frequency > 100000:
        result.warnings.append(f"Very high frequency: {frequency} Hz")

    # Duration consistency
    if isinstance(frequency, (int, float)) and frequency > 0 and data.ndim >= 1:
        # inspect-only loads return a tiny stub (e.g. first 1 s) while `duration`
        # reports the file's TRUE length, so the check would always "mismatch"
        # by ~orders of magnitude and flood logs once per file during batch
        # inspection. Skip it for inspect stubs — the full-load path validates
        # the real signal.
        _meta = data_dict.get("meta") if isinstance(data_dict, dict) else None
        _is_stub = bool(isinstance(_meta, dict) and _meta.get("inspect_only"))
        expected_duration = n_samples / frequency
        reported_duration = data_dict.get("duration")
        if not _is_stub and reported_duration is not None:
            ratio = abs(expected_duration - reported_duration) / max(expected_duration, 0.001)
            if ratio > 0.1:
                result.warnings.append(
                    f"Duration mismatch: reported={reported_duration:.2f}s vs "
                    f"computed={expected_duration:.2f}s (diff={ratio*100:.1f}%)"
                )

    # All-NaN channels
    if data.ndim == 2:
        nan_channels = np.all(np.isnan(data), axis=1)
        n_nan = int(nan_channels.sum())
        if n_nan > 0:
            result.warnings.append(f"{n_nan} channel(s) are entirely NaN")
            if n_nan == data.shape[0]:
                result.valid = False
                result.issues.append("All channels are NaN — data is empty")

    # All-zero check
    if data.ndim >= 1 and data.size > 0:
        if np.all(data == 0):
            result.warnings.append("Data is entirely zero — may be uninitialized")

    # Empty data
    if data.size == 0:
        result.valid = False
        result.issues.append("Data array is empty (size=0)")

    # Log results
    if result.issues:
        logger.warning("Data validation failed: %s", "; ".join(result.issues))
    if result.warnings:
        logger.info("Data validation warnings: %s", "; ".join(result.warnings))

    if not result.valid:
        logger.warning("Data validation failed: %s", "; ".join(result.issues))
        return result

    return result
