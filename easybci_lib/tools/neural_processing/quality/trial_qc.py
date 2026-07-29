"""Trial-level quality control — per-trial artifact assessment and rejection.

Evaluates each trial/epoch for:
- Artifact ratio (proportion of samples exceeding threshold)
- Peak-to-peak amplitude (channel-wise)
- Flat-line detection (zero-variance segments)
- Behavioral validity (response time, missing responses)
- Event timing sanity (onset within data range, reasonable ISI)

Outputs per-trial status: accepted / rejected_artifact / rejected_behavior / out_of_range.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def trial_qc(
    segments: np.ndarray,
    frequency: float,
    events: Optional[List[Dict[str, Any]]] = None,
    amplitude_threshold: float = 100e-6,
    flat_threshold: float = 1e-8,
    max_artifact_ratio: float = 0.3,
    expected_n_trials: Optional[int] = None,
    data_duration: Optional[float] = None,
    behavioral_data: Optional[List[Dict[str, Any]]] = None,
    rt_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Perform trial-level quality assessment.

    Parameters
    ----------
    segments : ndarray shape (n_trials, n_channels, n_samples)
        Epoched data.
    frequency : float
        Sampling rate in Hz.
    events : list of dict, optional
        Event info per trial. Each dict may have "onset", "type", "response_time".
    amplitude_threshold : float
        Threshold for artifact detection (in data units, default 100µV).
    flat_threshold : float
        Std below this is considered flat/dead channel.
    max_artifact_ratio : float
        Max proportion of artifact samples for a trial to pass (0-1).
    expected_n_trials : int, optional
        Expected number of trials. Reports missing if actual < expected.
    data_duration : float, optional
        Total data duration in seconds (for out-of-range detection).
    behavioral_data : list of dict, optional
        Per-trial behavioral data (response_time, accuracy, etc.).
    rt_range : tuple (min_rt, max_rt), optional
        Valid response time range in seconds. Trials outside are flagged.

    Returns
    -------
    Dict with:
        trial_status : list[str] — status per trial
        trial_metrics : list[dict] — per-trial quality metrics
        summary : dict — aggregate statistics
        rejection_report : dict — counts and reasons
    """
    n_trials, n_channels, n_samples = segments.shape

    trial_status = []
    trial_metrics = []

    for i in range(n_trials):
        trial = segments[i]
        metrics: Dict[str, Any] = {"trial_idx": i}

        # Amplitude analysis
        peak_to_peak = np.ptp(trial, axis=-1)
        max_ptp = float(np.max(peak_to_peak))
        mean_ptp = float(np.mean(peak_to_peak))
        metrics["peak_to_peak_max"] = max_ptp
        metrics["peak_to_peak_mean"] = mean_ptp

        # Artifact ratio: samples exceeding threshold
        exceeds = np.abs(trial) > amplitude_threshold
        artifact_ratio = float(np.mean(exceeds))
        metrics["artifact_ratio"] = artifact_ratio

        # Flat channel detection
        channel_stds = np.std(trial, axis=-1)
        n_flat = int(np.sum(channel_stds < flat_threshold))
        metrics["n_flat_channels"] = n_flat
        metrics["flat_ratio"] = n_flat / max(n_channels, 1)

        # RMS per channel
        rms = np.sqrt(np.mean(trial**2, axis=-1))
        metrics["rms_mean"] = float(np.mean(rms))
        metrics["rms_std"] = float(np.std(rms))

        # Determine status
        status = "accepted"
        rejection_reason = None

        if artifact_ratio > max_artifact_ratio:
            status = "rejected_artifact"
            rejection_reason = f"artifact_ratio={artifact_ratio:.3f} > {max_artifact_ratio}"
        elif n_flat > n_channels * 0.5:
            status = "rejected_artifact"
            rejection_reason = f"flat_channels={n_flat}/{n_channels}"
        elif max_ptp > amplitude_threshold * 20:
            status = "rejected_artifact"
            rejection_reason = f"extreme_amplitude={max_ptp:.6f}"

        # Behavioral validity check
        if status == "accepted" and behavioral_data and i < len(behavioral_data):
            beh = behavioral_data[i]
            rt = beh.get("response_time", beh.get("rt"))
            if rt is not None:
                rt = float(rt)
                metrics["response_time"] = rt
                if rt < 0 or rt == -1:
                    status = "rejected_behavior"
                    rejection_reason = "no_response"
                elif rt_range and (rt < rt_range[0] or rt > rt_range[1]):
                    status = "rejected_behavior"
                    rejection_reason = f"rt={rt:.3f}s outside [{rt_range[0]}, {rt_range[1]}]"

        # Event timing check
        if events and i < len(events):
            event = events[i]
            onset = event.get("onset", event.get("start"))
            if onset is not None and data_duration is not None:
                if float(onset) < 0 or float(onset) > data_duration:
                    status = "out_of_range"
                    rejection_reason = f"onset={onset}s outside [0, {data_duration}]"
                metrics["onset"] = float(onset)

        metrics["status"] = status
        if rejection_reason:
            metrics["rejection_reason"] = rejection_reason

        trial_status.append(status)
        trial_metrics.append(metrics)

    # Summary statistics
    n_accepted = trial_status.count("accepted")
    n_rejected_artifact = trial_status.count("rejected_artifact")
    n_rejected_behavior = trial_status.count("rejected_behavior")
    n_out_of_range = trial_status.count("out_of_range")

    # Missing trials
    missing_trials = 0
    if expected_n_trials is not None and expected_n_trials > n_trials:
        missing_trials = expected_n_trials - n_trials

    # ISI analysis (if events available)
    isi_stats = None
    if events and len(events) >= 2:
        isi_stats = _compute_isi_stats(events, frequency)

    summary = {
        "n_trials": n_trials,
        "n_accepted": n_accepted,
        "n_rejected": n_trials - n_accepted,
        "acceptance_rate": n_accepted / max(n_trials, 1),
        "rejection_breakdown": {
            "artifact": n_rejected_artifact,
            "behavior": n_rejected_behavior,
            "out_of_range": n_out_of_range,
        },
        "missing_trials": missing_trials,
        "expected_trials": expected_n_trials,
        "amplitude_threshold": amplitude_threshold,
        "max_artifact_ratio": max_artifact_ratio,
    }
    if isi_stats:
        summary["isi_stats"] = isi_stats

    # Artifact distribution across channels
    if n_trials > 0:
        per_channel_artifact = np.mean(
            np.abs(segments) > amplitude_threshold, axis=(0, 2)
        )
        worst_channels = np.argsort(per_channel_artifact)[-5:][::-1]
        summary["worst_channels"] = {
            "indices": worst_channels.tolist(),
            "artifact_rates": per_channel_artifact[worst_channels].tolist(),
        }

    return {
        "trial_status": trial_status,
        "trial_metrics": trial_metrics,
        "summary": summary,
        "accepted_indices": [i for i, s in enumerate(trial_status) if s == "accepted"],
        "rejected_indices": [i for i, s in enumerate(trial_status) if s != "accepted"],
    }


def _compute_isi_stats(events: List[Dict[str, Any]], frequency: float) -> Dict[str, Any]:
    """Compute inter-stimulus interval statistics."""
    onsets = []
    for ev in events:
        onset = ev.get("onset", ev.get("start"))
        if onset is not None:
            onsets.append(float(onset))

    if len(onsets) < 2:
        return {}

    onsets_sorted = sorted(onsets)
    isis = np.diff(onsets_sorted)

    stats: Dict[str, Any] = {
        "mean_s": float(np.mean(isis)),
        "std_s": float(np.std(isis)),
        "min_s": float(np.min(isis)),
        "max_s": float(np.max(isis)),
        "cv": float(np.std(isis) / max(np.mean(isis), 1e-10)),
    }

    # Detect anomalies
    if len(isis) > 5:
        median_isi = float(np.median(isis))
        mad = float(np.median(np.abs(isis - median_isi)))
        threshold = median_isi + 5 * max(mad, 1e-6)
        anomalous_idx = np.where(isis > threshold)[0]
        stats["n_anomalous_intervals"] = int(len(anomalous_idx))
        if len(anomalous_idx) > 0 and len(anomalous_idx) <= 10:
            stats["anomalous_positions"] = anomalous_idx.tolist()

    # Check for zero/negative intervals
    n_zero = int(np.sum(isis <= 0))
    n_very_short = int(np.sum(isis < 0.01))
    if n_zero > 0:
        stats["warning_zero_intervals"] = n_zero
    if n_very_short > n_zero:
        stats["warning_very_short_intervals"] = n_very_short

    return stats


def filter_trials(
    segments: np.ndarray,
    trial_status: List[str],
    keep_statuses: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Filter segments based on trial QC status.

    Parameters
    ----------
    segments : ndarray (n_trials, n_channels, n_samples)
    trial_status : list of status strings from trial_qc()
    keep_statuses : list of statuses to keep. Default: ["accepted"]

    Returns
    -------
    filtered_segments : ndarray — only accepted trials
    kept_indices : list[int] — original indices of kept trials
    """
    if keep_statuses is None:
        keep_statuses = ["accepted"]

    kept_indices = [i for i, s in enumerate(trial_status) if s in keep_statuses]

    if not kept_indices:
        n_channels = segments.shape[1] if segments.ndim >= 2 else 1
        n_samples = segments.shape[2] if segments.ndim >= 3 else 0
        return np.zeros((0, n_channels, n_samples), dtype=segments.dtype), []

    return segments[kept_indices], kept_indices
