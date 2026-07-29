"""Conditional pipeline routing — data-driven step selection.

Given a DataProfile and target modality/paradigm, produces an adapted pipeline
that skips unnecessary steps, inserts required ones, and adjusts parameters
based on observed signal characteristics.

This replaces the static PIPELINE_RECOMMENDATIONS lookup with intelligent
routing while keeping the static recommendations as fallback defaults.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from easybci_lib.tools.neural_processing.profile.data_profile import DataProfile

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """A single step inclusion/exclusion decision with reasoning."""
    step: str
    action: str  # "include", "skip", "modify"
    reason: str
    original_step: str = ""  # if modified, what was the original


@dataclass
class SegmentationStrategy:
    """Segmentation approach derived from label type."""
    method: str  # "event_locked", "interval", "sliding_window", "hierarchical"
    label_type: str = ""  # L1-L5 or "none"
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "label_type": self.label_type,
            "params": self.params,
            "reason": self.reason,
        }


@dataclass
class RoutedPipeline:
    """Result of adaptive routing: steps + per-step reasoning."""
    steps: List[str]
    decisions: List[RoutingDecision]
    profile_summary: str = ""
    adaptations_made: int = 0
    segmentation: Optional["SegmentationStrategy"] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "steps": self.steps,
            "decisions": [
                {"step": d.step, "action": d.action, "reason": d.reason}
                for d in self.decisions
            ],
            "profile_summary": self.profile_summary,
            "adaptations_made": self.adaptations_made,
        }
        if self.segmentation:
            result["segmentation"] = self.segmentation.to_dict()
        return result


# --- Default pipelines (baseline before routing) ---

_BASELINE_PIPELINES: Dict[str, Dict[str, List[str]]] = {
    "eeg": {
        "motor_imagery": ["drop_bads", "notch:50", "bandpass:0.5,40", "resample:256", "scale:robust"],
        "erp": ["drop_bads", "notch:50", "bandpass:0.1,30", "resample:256", "scale:standard"],
        "p300": ["drop_bads", "notch:50", "bandpass:0.1,30", "resample:256", "scale:standard"],
        "ssvep": ["drop_bads", "notch:50", "bandpass:5,45", "resample:256", "scale:robust"],
        "emotion": ["drop_bads", "notch:50", "bandpass:0.5,45", "resample:256", "scale:robust"],
        "sleep": ["drop_bads", "notch:50", "bandpass:0.3,35", "resample:256", "scale:robust"],
        "default": ["drop_bads", "notch:50", "bandpass:0.5,40", "resample:256", "scale:robust"],
    },
    "seeg": {
        "epilepsy": ["drop_bads", "bipolar_ref", "notch:50", "bandpass:0.5,200", "scale:robust"],
        "default": ["drop_bads", "bipolar_ref", "notch:50", "bandpass:1,200", "scale:robust"],
    },
    "ecog": {
        "default": ["drop_bads", "car", "notch:50", "bandpass:1,200", "scale:robust"],
    },
    "meg": {
        "default": ["drop_bads", "notch:50", "bandpass:1,100", "resample:500", "scale:robust"],
    },
    "spike": {
        "default": ["drop_bads", "scale:standard"],
    },
    "fnirs": {
        "default": ["bandpass:0.01,0.2", "scale:standard"],
    },
}


def route_pipeline(
    profile: DataProfile,
    modality: str,
    paradigm: str = "default",
    label_type: Optional[str] = None,
    label_info: Optional[Dict[str, Any]] = None,
    n_events: int = 0,
) -> RoutedPipeline:
    """Produce an adapted pipeline based on data characteristics.

    Parameters
    ----------
    profile : DataProfile
        Quantified data characteristics from compute_profile().
    modality : str
        Neural data modality (eeg, seeg, ecog, meg, spike, fnirs).
    paradigm : str
        Processing paradigm (motor_imagery, erp, ssvep, etc.).
    label_type : str, optional
        Detected label type (e.g. "L1_event", "L2_segment") for segmentation routing.
    label_info : dict, optional
        Full label classification result from classify_label_type().
    n_events : int
        Number of detected events/intervals.

    Returns
    -------
    RoutedPipeline with adapted steps, per-decision reasoning, and segmentation strategy.
    """
    mod_pipelines = _BASELINE_PIPELINES.get(modality, _BASELINE_PIPELINES.get("eeg", {}))
    baseline = list(mod_pipelines.get(paradigm, mod_pipelines.get("default", [])))

    decisions: List[RoutingDecision] = []
    final_steps: List[str] = []

    for step in baseline:
        step_name = step.split(":")[0]
        decision = _evaluate_step(step, step_name, profile, modality, paradigm)
        decisions.append(decision)

        if decision.action == "include":
            final_steps.append(decision.step)
        elif decision.action == "modify":
            final_steps.append(decision.step)

    # Check for additional steps the profile suggests but baseline doesn't include
    extra_decisions = _suggest_extra_steps(final_steps, profile, modality)
    for extra_d in extra_decisions:
        decisions.append(extra_d)
        if extra_d.action == "include":
            final_steps = _insert_step(final_steps, extra_d.step)

    adaptations = sum(1 for d in decisions if d.action in ("skip", "modify") or
                      (d.action == "include" and d.step not in baseline))

    summary = _build_profile_summary(profile)

    # Route segmentation strategy based on label type
    seg_strategy = route_segmentation(
        label_type=label_type,
        paradigm=paradigm,
        data_duration=profile.duration_s,
        frequency=profile.sampling_rate,
        n_events=n_events,
        label_info=label_info,
    )

    return RoutedPipeline(
        steps=final_steps,
        decisions=decisions,
        profile_summary=summary,
        adaptations_made=adaptations,
        segmentation=seg_strategy,
    )


def _evaluate_step(
    step: str, step_name: str, profile: DataProfile, modality: str, paradigm: str
) -> RoutingDecision:
    """Decide whether to include, skip, or modify a baseline step."""

    # --- NOTCH filter ---
    if step_name == "notch":
        if not profile.powerline_present:
            return RoutingDecision(
                step=step, action="skip",
                reason=f"No power line interference detected (peak prominence {profile.powerline_amplitude_db:.1f} dB < 6 dB threshold). "
                       f"Skipping notch filter preserves spectral integrity near {profile.powerline_freq or 50} Hz.",
            )
        # Adapt to detected powerline frequency
        detected_freq = profile.powerline_freq
        step_freq = _parse_notch_freq(step)
        if detected_freq > 0 and abs(detected_freq - step_freq) > 5:
            new_step = f"notch:{int(detected_freq)}"
            return RoutingDecision(
                step=new_step, action="modify",
                reason=f"Power line detected at {detected_freq} Hz (not {step_freq} Hz). "
                       f"Adapting notch frequency to match actual interference.",
                original_step=step,
            )
        return RoutingDecision(
            step=step, action="include",
            reason=f"Power line interference confirmed at {profile.powerline_amplitude_db:.1f} dB prominence. Notch filter required.",
        )

    # --- BANDPASS filter ---
    if step_name == "bandpass":
        low, high = _parse_bandpass_params(step)
        nyquist = profile.sampling_rate / 2.0 if profile.sampling_rate > 0 else 500

        # Adjust high cutoff if effective bandwidth is much lower
        if profile.effective_bandwidth > 0 and high > 0:
            # If 95% of signal power is below effective_bandwidth, consider tightening
            # But don't tighten below paradigm-minimum (e.g., gamma for some paradigms)
            suggested_high = min(high, max(profile.effective_bandwidth * 1.2, _min_high_freq(paradigm)))
            if suggested_high < high * 0.7:
                new_step = f"bandpass:{low},{suggested_high:.0f}"
                return RoutingDecision(
                    step=new_step, action="modify",
                    reason=f"Effective signal bandwidth is {profile.effective_bandwidth:.0f} Hz. "
                           f"Tightening high cutoff from {high} Hz to {suggested_high:.0f} Hz reduces noise without information loss.",
                    original_step=step,
                )

        # Adjust for significant drift
        if profile.has_significant_drift and low is not None and low < 0.5:
            new_low = 0.5
            new_step = f"bandpass:{new_low},{high}" if high else f"bandpass:{new_low},"
            return RoutingDecision(
                step=new_step, action="modify",
                reason=f"Significant drift detected (severity={profile.drift_severity:.2f}). "
                       f"Raising high-pass from {low} Hz to {new_low} Hz for more aggressive drift removal.",
                original_step=step,
            )

        return RoutingDecision(
            step=step, action="include",
            reason=f"Bandpass {low}-{high} Hz appropriate for {paradigm} paradigm with current signal characteristics.",
        )

    # --- RESAMPLE ---
    if step_name == "resample":
        target_freq = _parse_resample_freq(step)
        if profile.sampling_rate > 0 and target_freq >= profile.sampling_rate:
            return RoutingDecision(
                step=step, action="skip",
                reason=f"Current sampling rate ({profile.sampling_rate:.0f} Hz) is already at or below target ({target_freq} Hz). Resampling not needed.",
            )
        # If sampling rate is very high and effective bandwidth is low, suggest more aggressive downsampling
        if profile.effective_bandwidth > 0 and profile.sampling_rate > target_freq * 4:
            min_rate = max(target_freq, int(profile.effective_bandwidth * 2.5))
            if min_rate < target_freq:
                new_step = f"resample:{min_rate}"
                return RoutingDecision(
                    step=new_step, action="modify",
                    reason=f"Sampling rate ({profile.sampling_rate:.0f} Hz) is very high relative to effective bandwidth ({profile.effective_bandwidth:.0f} Hz). "
                           f"Downsampling to {min_rate} Hz (>2.5x effective bandwidth) is safe and reduces data volume more.",
                    original_step=step,
                )
        return RoutingDecision(
            step=step, action="include",
            reason=f"Downsampling from {profile.sampling_rate:.0f} Hz to {target_freq} Hz. Satisfies Nyquist criterion for bandpass upper bound.",
        )

    # --- DROP_BADS ---
    if step_name == "drop_bads":
        if profile.n_bad_channels == 0 and profile.flat_channel_ratio == 0:
            return RoutingDecision(
                step=step, action="skip",
                reason="No bad or flat channels detected. All channels show consistent variance. Skipping channel rejection.",
            )
        return RoutingDecision(
            step=step, action="include",
            reason=f"{profile.n_bad_channels} bad channels detected (flat ratio: {profile.flat_channel_ratio:.1%}). Channel rejection needed.",
        )

    # --- SCALE ---
    if step_name == "scale":
        if profile.has_extreme_amplitudes:
            return RoutingDecision(
                step="scale:robust", action="modify" if "robust" not in step else "include",
                reason="Extreme amplitudes present. Robust scaling (median/IQR) preferred over standard z-score to resist outlier influence.",
                original_step=step if "robust" not in step else "",
            )
        return RoutingDecision(
            step=step, action="include",
            reason=f"Amplitude normalization for consistent feature scaling. Dynamic range: {profile.dynamic_range_db:.0f} dB.",
        )

    # --- CAR / BIPOLAR_REF ---
    if step_name in ("car", "bipolar_ref"):
        return RoutingDecision(
            step=step, action="include",
            reason=f"Re-referencing ({step_name}) standard for {modality} data.",
        )

    # --- INTERPOLATE_BADS ---
    if step_name == "interpolate_bads":
        if profile.n_bad_channels == 0:
            return RoutingDecision(
                step=step, action="skip",
                reason="No bad channels to interpolate.",
            )
        return RoutingDecision(
            step=step, action="include",
            reason=f"{profile.n_bad_channels} channels need interpolation.",
        )

    # --- Default: include as-is ---
    return RoutingDecision(
        step=step, action="include",
        reason=f"Standard step for {modality}/{paradigm} pipeline.",
    )


def _suggest_extra_steps(
    current_steps: List[str],
    profile: DataProfile,
    modality: str,
) -> List[RoutingDecision]:
    """Suggest additional steps not in baseline but indicated by profile."""
    extra: List[RoutingDecision] = []

    # Suggest fill_nan if NaNs present
    if profile.has_nans and not any(s.startswith("fill_nan") for s in current_steps):
        extra.append(RoutingDecision(
            step="fill_nan:0",
            action="include",
            reason=f"Data contains NaN values ({profile.nan_ratio:.2%} of samples). Inserting fill_nan to prevent downstream errors.",
        ))

    # Suggest clip if extreme amplitudes and no clip already
    if profile.has_extreme_amplitudes and profile.artifact_ratio > 0.1:
        if not any(s.startswith("clip") for s in current_steps):
            extra.append(RoutingDecision(
                step="clip:500",
                action="include",
                reason=f"High artifact contamination ({profile.artifact_ratio:.0%} of windows). "
                       f"Adding amplitude clipping to limit extreme outliers before scaling.",
            ))

    # Suggest ICA if artifact ratio is moderate (not extreme) and modality supports it
    if 0.05 < profile.artifact_ratio < 0.4 and modality in ("eeg", "meg"):
        if not any(s.startswith("ica") for s in current_steps):
            if profile.n_channels >= 16 and profile.duration_s >= 30:
                extra.append(RoutingDecision(
                    step="ica:eog",
                    action="include",
                    reason=f"Moderate artifact contamination ({profile.artifact_ratio:.0%}) with sufficient channels ({profile.n_channels}) "
                           f"and duration ({profile.duration_s:.0f}s) for reliable ICA decomposition. Suggesting ICA for artifact removal.",
                ))

    # Suggest interpolate_bads if bad channels detected but not in pipeline
    if profile.n_bad_channels > 0 and profile.n_bad_channels <= profile.n_channels * 0.15:
        if not any(s.startswith("interpolate_bads") for s in current_steps):
            if any(s.startswith("drop_bads") for s in current_steps):
                pass  # drop_bads handles it
            else:
                extra.append(RoutingDecision(
                    step="interpolate_bads",
                    action="include",
                    reason=f"{profile.n_bad_channels} bad channels detected (<15% of total). "
                           f"Interpolation preserves channel count for spatial methods (CSP, source localization).",
                ))

    return extra


def _insert_step(steps: List[str], new_step: str) -> List[str]:
    """Insert a step at the appropriate position in the pipeline."""
    step_name = new_step.split(":")[0]

    # Priority ordering for insertion
    _ORDER = [
        "fill_nan", "pick_channels", "drop_bads", "interpolate_bads",
        "car", "bipolar_ref", "notch", "bandpass", "hilbert",
        "ica", "resample", "clip", "scale",
    ]

    new_priority = _ORDER.index(step_name) if step_name in _ORDER else len(_ORDER)

    for i, existing in enumerate(steps):
        existing_name = existing.split(":")[0]
        existing_priority = _ORDER.index(existing_name) if existing_name in _ORDER else len(_ORDER)
        if existing_priority > new_priority:
            return steps[:i] + [new_step] + steps[i:]

    return steps + [new_step]


def _build_profile_summary(profile: DataProfile) -> str:
    """One-paragraph summary of data characteristics for user display."""
    parts = []

    if profile.powerline_present:
        parts.append(f"power line at {profile.powerline_freq:.0f} Hz ({profile.powerline_amplitude_db:.0f} dB)")
    else:
        parts.append("no power line interference")

    if profile.has_significant_drift:
        parts.append(f"significant drift (severity {profile.drift_severity:.2f})")

    if profile.n_bad_channels > 0:
        parts.append(f"{profile.n_bad_channels} suspect channels")

    if profile.has_extreme_amplitudes:
        parts.append(f"artifact contamination in {profile.artifact_ratio:.0%} of windows")

    parts.append(f"quality score {profile.quality_score:.0%}")

    return f"Data profile: {', '.join(parts)}."


# --- Helper parsers ---

def _parse_notch_freq(step: str) -> float:
    parts = step.split(":")
    if len(parts) > 1:
        try:
            return float(parts[1].split(",")[0])
        except ValueError:
            pass
    return 50.0


def _parse_bandpass_params(step: str) -> Tuple[Optional[float], Optional[float]]:
    parts = step.split(":")
    if len(parts) > 1:
        params = parts[1].split(",")
        low = float(params[0]) if params[0] else None
        high = float(params[1]) if len(params) > 1 and params[1] else None
        return low, high
    return None, None


def _parse_resample_freq(step: str) -> float:
    parts = step.split(":")
    if len(parts) > 1:
        try:
            return float(parts[1])
        except ValueError:
            pass
    return 256.0


def _min_high_freq(paradigm: str) -> float:
    """Minimum high-cutoff frequency for a paradigm (preserves essential bands)."""
    _PARADIGM_MIN_HIGH = {
        "motor_imagery": 35.0,
        "erp": 25.0,
        "p300": 25.0,
        "ssvep": 45.0,
        "emotion": 45.0,
        "sleep": 35.0,
        "default": 40.0,
    }
    return _PARADIGM_MIN_HIGH.get(paradigm, 40.0)


# --- Segmentation strategy routing ---

def route_segmentation(
    label_type: Optional[str] = None,
    paradigm: str = "default",
    data_duration: float = 0.0,
    frequency: float = 256.0,
    n_events: int = 0,
    label_info: Optional[Dict[str, Any]] = None,
) -> SegmentationStrategy:
    """Determine segmentation strategy based on label type classification.

    Parameters
    ----------
    label_type : str or None
        Label type classification: "L1_event", "L2_segment", "L3_continuous",
        "L4_session", "L5_hierarchical", or None (no labels detected).
    paradigm : str
        Processing paradigm for default parameter selection.
    data_duration : float
        Total data duration in seconds (for window sizing).
    frequency : float
        Sampling rate after preprocessing.
    n_events : int
        Number of events/intervals detected (for sizing).
    label_info : dict, optional
        Additional label classification details (from classify_label_type).

    Returns
    -------
    SegmentationStrategy with method, params, and reasoning.
    """
    if label_info is None:
        label_info = {}

    if label_type == "L1_event":
        return _strategy_event_locked(paradigm, n_events, label_info)
    elif label_type == "L2_segment":
        return _strategy_interval(paradigm, n_events, label_info)
    elif label_type == "L3_continuous":
        return _strategy_sliding_window_align(paradigm, data_duration, frequency, label_info)
    elif label_type == "L4_session":
        return _strategy_sliding_window_broadcast(paradigm, data_duration, frequency, label_info)
    elif label_type == "L5_hierarchical":
        return _strategy_hierarchical(paradigm, label_info)
    else:
        return _strategy_default_sliding(paradigm, data_duration, frequency)


def _strategy_event_locked(
    paradigm: str, n_events: int, label_info: Dict[str, Any]
) -> SegmentationStrategy:
    """L1 labels → event-locked epoching around onsets."""
    duration_map = {
        "motor_imagery": 4.0,
        "erp": 1.0,
        "p300": 0.8,
        "ssvep": 4.0,
        "emotion": 5.0,
        "sleep": 30.0,
        "default": 2.0,
    }
    offset_map = {
        "erp": -0.2,
        "p300": -0.2,
        "motor_imagery": -0.5,
        "default": 0.0,
    }
    duration = duration_map.get(paradigm, 2.0)
    offset = offset_map.get(paradigm, 0.0)
    baseline = (-0.2, 0.0) if paradigm in ("erp", "p300") else None

    params = {
        "duration": duration,
        "offset": offset,
    }
    if baseline:
        params["baseline"] = list(baseline)

    return SegmentationStrategy(
        method="event_locked",
        label_type="L1_event",
        params=params,
        reason=f"L1 event labels detected ({n_events} events) → epoch {duration}s "
               f"segments locked to event onsets (offset={offset}s, paradigm={paradigm})",
    )


def _strategy_interval(
    paradigm: str, n_events: int, label_info: Dict[str, Any]
) -> SegmentationStrategy:
    """L2 labels → interval-based extraction using explicit start/end."""
    mean_duration = label_info.get("details", {}).get("mean_duration", 0)

    params: Dict[str, Any] = {
        "gap_handling": "mark_unlabeled",
        "pad_to_max": False,
    }
    if mean_duration > 0:
        params["expected_duration"] = round(mean_duration, 2)

    return SegmentationStrategy(
        method="interval",
        label_type="L2_segment",
        params=params,
        reason=f"L2 segment labels detected ({n_events} intervals"
               f"{f', mean {mean_duration:.1f}s' if mean_duration > 0 else ''}"
               f") → extract variable-length segments by explicit [start, end] boundaries",
    )


def _strategy_sliding_window_align(
    paradigm: str,
    data_duration: float,
    frequency: float,
    label_info: Dict[str, Any],
) -> SegmentationStrategy:
    """L3 labels → sliding window + label alignment to match windows."""
    window_map = {
        "motor_imagery": 2.0,
        "emotion": 4.0,
        "sleep": 30.0,
        "default": 2.0,
    }
    stride_ratio_map = {
        "motor_imagery": 0.5,
        "emotion": 0.5,
        "sleep": 1.0,
        "default": 0.5,
    }
    window = window_map.get(paradigm, 2.0)
    stride = window * stride_ratio_map.get(paradigm, 0.5)

    label_freq = label_info.get("details", {}).get("inferred_label_freq", frequency)

    params = {
        "window_duration": window,
        "stride": stride,
        "label_alignment": "interpolate",
        "label_freq": label_freq,
        "aggregate": "majority_vote",
    }

    return SegmentationStrategy(
        method="sliding_window_align",
        label_type="L3_continuous",
        params=params,
        reason=f"L3 continuous labels detected (label_freq={label_freq}Hz) → "
               f"sliding windows ({window}s, stride={stride}s) with per-window "
               f"label alignment via interpolation + majority vote",
    )


def _strategy_sliding_window_broadcast(
    paradigm: str,
    data_duration: float,
    frequency: float,
    label_info: Dict[str, Any],
) -> SegmentationStrategy:
    """L4 labels → sliding window with session label broadcast to all windows."""
    window_map = {
        "motor_imagery": 2.0,
        "emotion": 4.0,
        "sleep": 30.0,
        "default": 2.0,
    }
    window = window_map.get(paradigm, 2.0)
    stride = window * 0.5

    return SegmentationStrategy(
        method="sliding_window_broadcast",
        label_type="L4_session",
        params={
            "window_duration": window,
            "stride": stride,
            "label_broadcast": True,
        },
        reason=f"L4 session-level labels detected → sliding windows ({window}s, "
               f"stride={stride}s) with session label broadcast to all segments",
    )


def _strategy_hierarchical(
    paradigm: str, label_info: Dict[str, Any]
) -> SegmentationStrategy:
    """L5 labels → parse hierarchy, then segment at each level."""
    depth = label_info.get("details", {}).get("nesting_depth", 2)

    return SegmentationStrategy(
        method="hierarchical",
        label_type="L5_hierarchical",
        params={
            "nesting_depth": depth,
            "primary_level": "leaf",
            "collapse_strategy": "innermost_first",
        },
        reason=f"L5 hierarchical labels detected (depth={depth}) → "
               f"parse nested structure, segment at leaf level, "
               f"propagate parent labels as metadata",
    )


def _strategy_default_sliding(
    paradigm: str, data_duration: float, frequency: float
) -> SegmentationStrategy:
    """No labels → default sliding window segmentation."""
    window_map = {
        "motor_imagery": 2.0,
        "erp": 1.0,
        "p300": 0.8,
        "ssvep": 4.0,
        "emotion": 4.0,
        "sleep": 30.0,
        "default": 2.0,
    }
    window = window_map.get(paradigm, 2.0)
    stride = window * 0.5

    return SegmentationStrategy(
        method="sliding_window",
        label_type="none",
        params={
            "window_duration": window,
            "stride": stride,
        },
        reason=f"No label information available → default sliding window "
               f"({window}s, stride={stride}s) for {paradigm} paradigm",
    )
