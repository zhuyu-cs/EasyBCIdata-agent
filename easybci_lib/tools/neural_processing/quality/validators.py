"""Signal validators — detect problems before they propagate downstream.

Fast checks an agent can run after loading or preprocessing to decide
whether to proceed, retry, or flag for human review.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_signal(
    data: np.ndarray,
    frequency: Optional[float] = None,
    checks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a battery of signal quality checks.

    Parameters
    ----------
    data : ndarray
        Shape (n_channels, n_samples) or (n_segments, n_channels, n_samples).
    frequency : float or None
        Sampling rate (required for some checks).
    checks : list of str or None
        Which checks to run. Default: all available.
        Options: "nan", "inf", "flat", "amplitude", "variance"

    Returns
    -------
    dict with:
        passed : bool — True if all checks pass
        issues : list of dict — each issue has "check", "severity", "detail"
        stats : dict — per-channel statistics
    """
    if checks is None:
        checks = ["nan", "inf", "flat", "amplitude", "variance"]

    # Normalize to 2D (channels × samples)
    if data.size == 0:
        logger.warning("Empty data array (size=0) passed to validate_signal — skipping checks.")
        return {
            "passed": False,
            "issues": [{"check": "empty", "severity": "error", "detail": "Data array is empty (size=0)"}],
            "stats": {"n_channels": 0, "n_samples": 0},
        }

    if data.ndim == 3:
        n_seg, n_ch, n_t = data.shape
        data_2d = data.reshape(n_seg * n_ch, n_t)
    elif data.ndim == 2:
        data_2d = data
    elif data.ndim == 1:
        data_2d = data[np.newaxis, :]
    else:
        logger.warning("Expected 1-3D array, got shape %s — returning passing validation result", data.shape)
        return {
            "passed": True,
            "issues": [],
            "stats": {},
        }

    if data_2d.shape[1] == 0:
        logger.warning("Data has channels but zero samples — skipping checks.")
        return {
            "passed": False,
            "issues": [{"check": "empty", "severity": "error", "detail": "Data has 0 samples per channel"}],
            "stats": {"n_channels": data_2d.shape[0], "n_samples": 0},
        }

    issues = []
    stats = {}

    if "nan" in checks:
        nan_mask = np.isnan(data_2d)
        nan_count = nan_mask.sum()
        nan_per_channel = nan_mask.sum(axis=-1)
        stats["nan_count"] = int(nan_count)
        stats["nan_channels"] = int((nan_per_channel > 0).sum())
        if nan_count > 0:
            pct = nan_count / data_2d.size * 100
            issues.append({
                "check": "nan",
                "severity": "error" if pct > 5 else "warning",
                "detail": f"{nan_count} NaN values ({pct:.2f}% of data) in {stats['nan_channels']} channels",
            })

    if "inf" in checks:
        inf_mask = np.isinf(data_2d)
        inf_count = inf_mask.sum()
        stats["inf_count"] = int(inf_count)
        if inf_count > 0:
            issues.append({
                "check": "inf",
                "severity": "error",
                "detail": f"{inf_count} Inf values detected",
            })

    if "flat" in checks:
        # Channels with zero variance (flatlined)
        with np.errstate(invalid="ignore"):
            ch_var = np.nanvar(data_2d, axis=-1)
        flat_channels = int((ch_var == 0).sum())
        stats["flat_channels"] = flat_channels
        if flat_channels > 0:
            issues.append({
                "check": "flat",
                "severity": "warning",
                "detail": f"{flat_channels} flat (zero-variance) channels detected",
            })

    if "amplitude" in checks:
        with np.errstate(invalid="ignore"):
            ch_max = np.nanmax(np.abs(data_2d), axis=-1)
        extreme_channels = int((ch_max > 1e6).sum())
        stats["max_amplitude"] = float(np.nanmax(ch_max)) if ch_max.size > 0 else 0.0
        stats["extreme_channels"] = extreme_channels
        if extreme_channels > 0:
            issues.append({
                "check": "amplitude",
                "severity": "warning",
                "detail": f"{extreme_channels} channels with amplitude > 1e6 (possible artifact or wrong scale)",
            })

    if "variance" in checks:
        with np.errstate(invalid="ignore"):
            ch_var = np.nanvar(data_2d, axis=-1)
            valid_var = ch_var[np.isfinite(ch_var) & (ch_var > 0)]
        if valid_var.size > 1:
            median_var = np.median(valid_var)
            high_var = int((valid_var > median_var * 100).sum())
            low_var = int((valid_var < median_var * 0.01).sum())
            stats["median_variance"] = float(median_var)
            stats["high_variance_channels"] = high_var
            stats["low_variance_channels"] = low_var
            if high_var > 0:
                issues.append({
                    "check": "variance",
                    "severity": "warning",
                    "detail": f"{high_var} channels with variance >100x median (possible artifact)",
                })

    passed = all(i["severity"] != "error" for i in issues)

    return {
        "passed": passed,
        "issues": issues,
        "stats": stats,
    }


def check_channels(
    channels: List[str],
    expected: Optional[List[str]] = None,
    min_count: int = 1,
) -> Dict[str, Any]:
    """Validate channel list.

    Parameters
    ----------
    channels : list of str
        Actual channel names.
    expected : list of str or None
        If provided, check that all expected channels are present.
    min_count : int
        Minimum acceptable number of channels.

    Returns
    -------
    dict with: passed, missing, extra, n_channels
    """
    result = {
        "passed": True,
        "n_channels": len(channels),
        "missing": [],
        "extra": [],
        "issues": [],
    }

    if len(channels) < min_count:
        result["passed"] = False
        result["issues"].append(
            f"Only {len(channels)} channels, expected at least {min_count}"
        )

    if expected:
        ch_set = set(channels)
        exp_set = set(expected)
        result["missing"] = sorted(exp_set - ch_set)
        result["extra"] = sorted(ch_set - exp_set)
        if result["missing"]:
            result["passed"] = False
            result["issues"].append(
                f"{len(result['missing'])} expected channels missing"
            )

    # Check for duplicates
    if len(channels) != len(set(channels)):
        from collections import Counter
        dupes = [ch for ch, count in Counter(channels).items() if count > 1]
        result["duplicates"] = dupes
        result["issues"].append(f"Duplicate channel names: {dupes}")

    return result


def check_sampling_rate(
    frequency: float,
    expected: Optional[float] = None,
    min_hz: float = 1.0,
    max_hz: float = 100000.0,
) -> Dict[str, Any]:
    """Validate sampling rate.

    Parameters
    ----------
    frequency : float
        Actual sampling rate.
    expected : float or None
        Expected rate (checks within 1% tolerance).
    min_hz, max_hz : float
        Acceptable range.

    Returns
    -------
    dict with: passed, issues
    """
    issues = []

    if frequency <= 0:
        issues.append(f"Invalid sampling rate: {frequency} Hz")
    elif frequency < min_hz:
        issues.append(f"Sampling rate {frequency} Hz below minimum {min_hz} Hz")
    elif frequency > max_hz:
        issues.append(f"Sampling rate {frequency} Hz above maximum {max_hz} Hz")

    if expected is not None and frequency > 0:
        pct_diff = abs(frequency - expected) / expected * 100
        if pct_diff > 1.0:
            issues.append(
                f"Sampling rate {frequency} Hz differs from expected {expected} Hz by {pct_diff:.1f}%"
            )

    return {
        "passed": len(issues) == 0,
        "frequency": frequency,
        "issues": issues,
    }
