"""Default output format policy for the ``preprocessed/`` layer — single source of truth.

NWB is the only accepted format for the non-AI-ready ``preprocessed/`` output
across every modality. The ``pkl`` escape hatch has been removed: the prior
dual-format support produced a tax across codegen / verifier / reader, and the
only legitimate consumer of a pkl preprocessed file (``AI_ready/`` epoch
construction) already reads NWB via :func:`output.nwb_writer.save_nwb`.

``is_invasive`` is kept for non-format concerns — it still gates the vis-
template choice (4 vs 5 figures) and a couple of channel-cleanup behaviours.

The module is pure: no dependency on pynwb / mne / hdf5 / any heavy library,
so it can be imported safely from any context (codegen, doctor, cli, etc.).
"""
from __future__ import annotations

from typing import Literal, Optional

OutputFormat = Literal["nwb"]
OutputFormatOrAuto = Literal["nwb", "auto"]

INVASIVE_MODALITIES: frozenset[str] = frozenset({
    "seeg", "ecog", "ieeg", "dbs", "spike", "spikes", "unit", "units",
})


def is_invasive(modality: Optional[str]) -> bool:
    """Return True only for *purely* invasive modalities.

    Strict semantics: None / empty / mixed / multimodal / unknown all return
    False. This function no longer governs output-format selection; it remains
    in use for the vis-template choice and channel-cleanup gates.
    """
    if not modality:
        return False
    return modality.strip().lower() in INVASIVE_MODALITIES


def resolve_default_format(
    modality: Optional[str],
    override: str = "auto",
) -> OutputFormat:
    """Resolve the final output format from (modality, override).

    Args:
        modality: data modality string (kept for signature compatibility;
            ignored by the current policy).
        override: one of ``"auto"`` | ``"nwb"``. Both resolve to ``"nwb"``.

    Returns:
        ``"nwb"``.

    Raises:
        ValueError: if override is neither ``"auto"`` nor ``"nwb"``. The
            previously-accepted ``"pkl"`` override has been removed; callers
            asking for pkl will hit this branch.
    """
    if override in ("auto", "nwb"):
        return "nwb"
    raise ValueError(
        f"output_format must be 'auto' or 'nwb' (pkl override has been "
        f"removed); got {override!r}"
    )
