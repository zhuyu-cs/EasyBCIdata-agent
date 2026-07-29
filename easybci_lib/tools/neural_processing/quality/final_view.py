"""Immutable post-pipeline data view — single contract for QC figure inputs.

Purpose
-------
Figure helpers (compare_viz, visualize, paradigm_viz) historically each pulled
``after_data`` and ``out_channels`` out of a shared payload. Any caller that
threaded raw or intermediate values through could silently regress visual
outputs (e.g. a 67-channel dataset showing 67 traces in figures even after the
pipeline dropped 3 channels).

``FinalDataView`` collapses that contract into a single immutable snapshot. By
construction, ``data.shape[0] == len(channels)`` and (when built via
``from_pipeline_result``) only ``data``-category channels survive — the
non-data drop is enforced by the existing channel classifier so the rule is
multi-modal (EEG / MEG / sEEG / ECoG / DBS) and is a no-op on spike data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalDataView:
    """Post-pipeline snapshot consumed by every QC figure helper."""

    data: np.ndarray
    channels: Tuple[str, ...]
    frequency: float
    modality: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"FinalDataView.data must be numpy.ndarray, got {type(self.data).__name__}"
            )
        if self.data.ndim != 2:
            raise ValueError(
                f"FinalDataView expects 2-D data, got ndim={self.data.ndim}"
            )
        if self.data.shape[0] != len(self.channels):
            raise ValueError(
                f"FinalDataView mismatch: data has {self.data.shape[0]} channels "
                f"but {len(self.channels)} names — figures must reflect "
                f"post-pipeline state only"
            )
        # Channels stored as tuple by dataclass freezing; if caller passed list,
        # coerce so equality comparisons in tests are stable.
        object.__setattr__(self, "channels", tuple(self.channels))

    @classmethod
    def from_pipeline_result(
        cls,
        after_data: np.ndarray,
        channels: Sequence[str],
        frequency: float,
        modality: str,
        *,
        enforce_data_only: bool = True,
    ) -> "FinalDataView":
        """Build a view from preprocess() output.

        When ``enforce_data_only`` is True, classify each channel via
        ``channel_classifier.classify_channels`` and keep only the ``data``
        category. Spike-family modalities are no-ops by design (the classifier
        already skips them via ``_NON_APPLICABLE_MODALITIES`` and returns
        ``applicable=False``).

        If the safety net would leave zero channels (an all-EOG file, for
        instance), fall back to the original ``(after_data, channels)`` and log
        a WARNING — the rest of the run should not be killed by a corner case.
        """
        chan_list = list(channels)
        if not enforce_data_only:
            return cls(data=after_data, channels=tuple(chan_list),
                       frequency=frequency, modality=modality)

        from easybci_lib.tools.neural_processing.io.channel_classifier import (
            classify_channels,
        )
        try:
            result = classify_channels(chan_list, modality=modality)
        except Exception as exc:  # classifier mis-behaves on a novel modality
            logger.warning(
                "FinalDataView: classifier failed for modality=%r (%s); "
                "falling back to unfiltered channels", modality, exc,
            )
            return cls(data=after_data, channels=tuple(chan_list),
                       frequency=frequency, modality=modality)

        # Spike-family / unsupported: classifier returns applicable=False —
        # leave channels untouched.
        if not result.get("applicable", False):
            return cls(data=after_data, channels=tuple(chan_list),
                       frequency=frequency, modality=modality)

        categories_map = result.get("categories", {})
        keep_names = [ch for ch in chan_list if categories_map.get(ch) == "data"]
        if not keep_names:
            logger.warning(
                "FinalDataView: classifier safety net would drop all %d "
                "channels for modality=%r — falling back to originals; verify "
                "the pipeline did not strip data channels by mistake",
                len(chan_list), modality,
            )
            return cls(data=after_data, channels=tuple(chan_list),
                       frequency=frequency, modality=modality)

        if keep_names == chan_list:
            # Nothing to drop — all channels are already 'data' category.
            return cls(data=after_data, channels=tuple(chan_list),
                       frequency=frequency, modality=modality)

        name_to_idx = {ch: i for i, ch in enumerate(chan_list)}
        keep_idx = [name_to_idx[ch] for ch in keep_names]
        dropped = [ch for ch in chan_list if ch not in set(keep_names)]
        logger.info(
            "FinalDataView: dropped %d non-data channel(s) via classifier "
            "safety net for modality=%r: %s",
            len(dropped), modality, dropped,
        )
        filtered_data = after_data[keep_idx, :]
        return cls(data=filtered_data, channels=tuple(keep_names),
                   frequency=frequency, modality=modality)
