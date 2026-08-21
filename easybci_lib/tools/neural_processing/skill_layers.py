"""Single source of truth for the skill four-layer architecture.

T9 (`01-skill-hierarchy-restructure`) defines four layers under
`easybci_lib/skills/bci/`:

- ``L0`` — IO format layer (`bci/neural-io/<family>/<format>/SKILL.md`)
- ``L1`` — Orchestration layer (`bci/pipeline/SKILL.md`)
- ``L2`` — Domain skill layer (`bci/paradigms/<group>/<leaf>/SKILL.md`),
  grouped by the orthogonal axes modality / paradigm / clinical / analysis /
  online / multimodal.
- ``L3`` — Atomic operator layer (`bci/operators/<group>/<op>/SKILL.md`)

Each non-L1 layer has a fixed list of *groups*. The enums live here and **only**
here; documentation, consistency checker, and index regenerator must import
them so the project never drifts between code and prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

LAYERS: List[str] = ["L0", "L1", "L2", "L3"]

L0_GROUPS: List[str] = [
    "ephys_wide_band",
    "spike_ephys",
    "clinical_ieeg",
    "consumer_eeg",
    "fnirs",
    "streaming",
]

L2_GROUPS: List[str] = [
    "paradigm",
    "modality",
    "clinical",
    "analysis",
    "online",
    "multimodal",
]

L3_GROUPS: List[str] = [
    "filter",
    "channel",
    "reference",
    "spatial",
    "spectral",
    "epoch",
    "event",
    "adaptive_cleaning",
    "meg",
    "dataset",
    "connectivity",
    "qc_operator",
    "spike",
    "psg",
    "fnirs",
    "source",
    "feature_time",
    "misc",
]

GROUPS_BY_LAYER = {
    "L0": L0_GROUPS,
    "L1": [],
    "L2": L2_GROUPS,
    "L3": L3_GROUPS,
}

ANALYSIS_GOALS: List[str] = [
    "classification",
    "source_localization",
    "feature_extraction",
    "clinical_screening",
    "exploratory",
    "generic",
    "connectivity",
    "phase_amplitude_coupling",
    "online_inference",
    "sleep_staging",
]


# ── Path → layer/group inference ────────────────────────────────────────────


def infer_layer_from_path(rel_path: Path | str) -> str | None:
    """Infer the expected layer from a path relative to ``bci/``.

    Returns ``"L0" | "L1" | "L2" | "L3"`` or ``None`` when the path does not
    belong to any of the four known layers (so the consistency checker can
    skip it rather than FAIL on legacy or unrelated files).
    """
    rel = Path(rel_path)
    parts = rel.parts
    if not parts:
        return None
    head = parts[0]
    if head == "neural-io":
        return "L0"
    if head == "pipeline":
        return "L1"
    if head == "paradigms":
        return "L2"
    if head == "operators":
        return "L3"
    return None


def infer_group_from_path(rel_path: Path | str) -> str | None:
    """Infer the group from a *post-migration* path relative to ``bci/``.

    Layer-specific path shapes (after 1-3 / 1-4 / 1-5 migration):

    - L0: ``neural-io/<group>/<format>/SKILL.md``  → ``parts[1]``
    - L2: ``paradigms/<group>/<leaf>/SKILL.md``  → ``parts[1]``
    - L3: ``operators/<group>/<op>/SKILL.md``   → ``parts[1]``

    Returns ``None`` when the path is in an *index* position (no group dir
    yet, e.g. ``operators/SKILL.md``) or is pre-migration flat.
    """
    layer = infer_layer_from_path(rel_path)
    if layer not in ("L0", "L2", "L3"):
        return None
    parts = Path(rel_path).parts
    if len(parts) < 3:
        # neural-io/SKILL.md, operators/SKILL.md, paradigms/SKILL.md → no group dir
        return None
    candidate = parts[1]
    if candidate in GROUPS_BY_LAYER[layer]:
        return candidate
    return None


__all__ = [
    "LAYERS",
    "L0_GROUPS",
    "L2_GROUPS",
    "L3_GROUPS",
    "GROUPS_BY_LAYER",
    "ANALYSIS_GOALS",
    "infer_layer_from_path",
    "infer_group_from_path",
]
