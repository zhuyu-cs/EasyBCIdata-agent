"""Map a RecipeProfile to an operator skeleton (step_string) + adaptation_slots.

Structure anchoring: step kinds/order follow the gold standard
(notch → bandpass → resample → drop_bads). Numeric values become adaptation
slots that component-3 recomputes per new recording. Line-frequency segment
rejection is a label-level rule (recorded, NOT a signal operator).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from easybci_lib.tools.neural_processing.reference.recipe_parser import RecipeProfile


@dataclass
class Skeleton:
    steps: list[str]
    step_string: str
    adaptation_slots: list[dict] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)


def _fmt(v: float) -> str:
    """Render a number without a trailing .0 for integers (50.0 -> '50')."""
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def build_skeleton(rp: RecipeProfile) -> Skeleton:
    steps: list[str] = []
    unmapped: list[str] = []

    # 1) notch per frequency (each maps to a known 'notch:<hz>' operator)
    for f in rp.notch_freqs:
        steps.append(f"notch:{_fmt(f)}")

    # 2) bandpass low,high
    if rp.low_cut and rp.high_cut:
        steps.append(f"bandpass:{_fmt(rp.low_cut)},{_fmt(rp.high_cut)}")
    else:
        unmapped.append("bandpass (missing low_cut/high_cut in recipe)")

    # 3) resample to target
    if rp.target_sfreq:
        steps.append(f"resample:{_fmt(rp.target_sfreq)}")
    else:
        unmapped.append("resample (missing target_sfreq in recipe)")

    # 4) bad-channel drop (CSV-driven in gold; becomes drop_bads:auto skeleton)
    steps.append("drop_bads:auto")

    step_string = "→".join(steps)

    slots = [
        {"param": "bad_channels", "strategy": "from_deep_inspect"},
        {"param": "notch_freqs", "strategy": "from_powerline"},
        {"param": "resample_target_hz", "strategy": "nyquist_bounded"},
        {"param": "reject_time_segments", "strategy": "from_labels"},
    ]
    return Skeleton(steps=steps, step_string=step_string,
                    adaptation_slots=slots, unmapped=unmapped)
