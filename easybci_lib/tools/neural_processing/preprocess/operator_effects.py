"""Operator effect descriptions for reasoning.md.

Short, human-readable summary of WHAT each preprocessing operator achieves
— used by reasoning_writer.py to render the "Effect" line under each
pipeline step. Kept separate from the operator implementations so the
descriptions can be edited / localized without touching signal-processing
code.

When adding a new operator: add an entry here, otherwise the reasoning.md
will fall back to a generic "Apply <op>" string. That's safe, but a tailored
sentence is much more informative.
"""

from __future__ import annotations

# Map operator name (the registry key — e.g. "notch", "bandpass", "car") to
# the post-step effect description. First sentence describes WHAT the
# operator does on the signal; consumers may append an "after" clause.
#
# Entries cover all 17 core operators surfaced through the codegen
# templates plus a few aliases. Keep each entry to one or two short
# clauses — reasoning.md is read scrolling vertically, not at width.
OPERATOR_EFFECTS: dict[str, str] = {
    "notch": (
        "Notch-filter the target mains frequency and its harmonics to remove "
        "power-line contamination from spectral features, with negligible "
        "distortion to neighboring bands."
    ),
    "bandpass": (
        "Keep the chosen band and suppress out-of-band noise — slow drift, "
        "EMG, and high-frequency noise are all attenuated, letting the target "
        "rhythms stand out."
    ),
    "highpass": (
        "Remove DC and slow drift so downstream analysis is no longer crushed "
        "by baseline wander in the band of interest."
    ),
    "lowpass": (
        "Remove high-frequency noise so the time-domain shape is smoother and "
        "easier to visualize and read rhythms from."
    ),
    "car": (
        "Common Average Reference — subtract the whole-brain mean reference "
        "to improve spatial resolution and accentuate local source activity."
    ),
    "rereference": (
        "Switch the reference scheme, changing the signal's spatial geometry; "
        "removes common-mode noise depending on configuration."
    ),
    "resample": (
        "Downsample to the target rate, shrinking data size and compute cost "
        "while keeping Nyquist above the band of interest."
    ),
    "ica": (
        "Use ICA to decompose the signal into independent components and "
        "automatically remove typical artifact components (EOG / EMG / ECG), "
        "preserving the neural source contributions."
    ),
    "drop_bads": (
        "Drop electrodes marked as bad channels to prevent localized noise "
        "from polluting downstream steps."
    ),
    "drop_nondata_channels": (
        "Drop non-data channels (Trigger / Stim, etc.) and keep only the "
        "EEG / EOG channels usable for modeling."
    ),
    "regression_eog": (
        "Regression-correct using the dedicated EOG channels, subtracting the "
        "EOG-correlated component from each EEG channel — more deterministic "
        "than ICA."
    ),
    "regression_ecg": (
        "Regression-correct using the ECG channel to remove cardiac crosstalk."
    ),
    "scale": (
        "Scale-normalize each channel (z-score / robust) so amplitudes are "
        "comparable across channels and friendly to training."
    ),
    "clip": (
        "Clip samples beyond a threshold to limit the distortion residual "
        "spikes can introduce into statistics."
    ),
    "fill_nan": (
        "Interpolate or fill NaN / Inf samples so downstream operators always "
        "see a complete array."
    ),
    "interp_bads": (
        "Spatially interpolate bad channels to keep the channel layout intact."
    ),
    "detrend": (
        "Remove linear trend to stabilize downstream spectral and statistical "
        "estimates."
    ),
    "downsample": "Downsample — alias of resample. See resample.",
    "common_average_reference": "See car.",
    "epoch": (
        "Slice into epochs by events or a sliding window, producing a "
        "(n_epoch, n_ch, n_sample) training tensor."
    ),
    "segment": "See epoch.",
}


def get_effect(operator: str) -> str:
    """Return the effect description for ``operator``.

    Unknown operators get a generic "Apply <op>" string so reasoning.md never
    breaks on a new step; the registry should still be updated for clarity.
    """
    if not operator:
        return ""
    key = str(operator).strip().lower()
    return OPERATOR_EFFECTS.get(key) or f"Apply {operator}."
