"""Spike binning — convert spike times to dense arrays.

Separate from the MNE pipeline since spikes have a fundamentally
different data model (event times → binned counts).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.tools.neural_processing._core.timed_array import Frequency

logger = logging.getLogger(__name__)


def bin_spikes(
    spike_trains: List[np.ndarray],
    bin_frequency: float,
    duration: Optional[float] = None,
    unit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Bin spike times into a dense count array.

    Parameters
    ----------
    spike_trains : list of ndarray
        Per-unit spike time arrays (in seconds).
    bin_frequency : float
        Binning rate in Hz (e.g. 1000 → 1ms bins).
    duration : float or None
        Total duration. None → auto from max spike time.
    unit_names : list of str or None
        Unit names. None → auto-numbered.

    Returns
    -------
    dict with:
        data : ndarray shape (n_units, n_bins)
        frequency : float
        channels : list[str]
        duration : float
        meta : dict
    """
    if spike_trains is None:
        logger.warning("No spike trains provided (None) — returning empty binned array")
        return {
            "data": np.zeros((0, 0), dtype=np.float32),
            "frequency": bin_frequency,
            "channels": [],
            "duration": 0.0,
            "meta": {"format": "binned_spikes", "n_units": 0, "total_spikes": 0, "bin_frequency": bin_frequency},
        }
    if isinstance(spike_trains, np.ndarray):
        if spike_trains.size == 0:
            logger.warning("No spike trains provided (empty array) — returning empty binned array")
            return {
                "data": np.zeros((0, 0), dtype=np.float32),
                "frequency": bin_frequency,
                "channels": [],
                "duration": 0.0,
                "meta": {"format": "binned_spikes", "n_units": 0, "total_spikes": 0, "bin_frequency": bin_frequency},
            }
        # 2D array: each row is a unit's spike times (padded with NaN or 0)
        if spike_trains.ndim == 2:
            spike_trains = [spike_trains[i][~np.isnan(spike_trains[i])] if np.issubdtype(spike_trains.dtype, np.floating) else spike_trains[i] for i in range(spike_trains.shape[0])]
        elif spike_trains.ndim == 1:
            spike_trains = [spike_trains]
        else:
            logger.warning("Unexpected spike_trains shape: %s — returning empty binned array", spike_trains.shape)
            return {
                "data": np.zeros((0, 0), dtype=np.float32),
                "frequency": bin_frequency,
                "channels": [],
                "duration": 0.0,
                "meta": {"format": "binned_spikes", "n_units": 0, "total_spikes": 0, "bin_frequency": bin_frequency},
            }
    elif hasattr(spike_trains, '__len__') and len(spike_trains) == 0:
        logger.warning("No spike trains provided (empty list) — returning empty binned array")
        return {
            "data": np.zeros((0, 0), dtype=np.float32),
            "frequency": bin_frequency,
            "channels": [],
            "duration": 0.0,
            "meta": {"format": "binned_spikes", "n_units": 0, "total_spikes": 0, "bin_frequency": bin_frequency},
        }

    freq = Frequency(bin_frequency)

    if duration is None:
        max_time = max(float(np.max(t)) for t in spike_trains if len(t) > 0)
        duration = max_time + 1.0 / bin_frequency
    n_bins = freq.to_ind(duration)

    n_units = len(spike_trains)
    binned = np.zeros((n_units, n_bins), dtype=np.float32)

    for i, times in enumerate(spike_trains):
        if len(times) == 0:
            continue
        indices = np.floor(np.asarray(times, dtype=np.float64) * bin_frequency).astype(int)
        indices = np.clip(indices, 0, n_bins - 1)
        np.add.at(binned[i], indices, 1.0)

    if unit_names is None:
        unit_names = [f"unit_{i}" for i in range(n_units)]

    return {
        "data": binned,
        "frequency": bin_frequency,
        "channels": unit_names,
        "duration": duration,
        "meta": {
            "format": "binned_spikes",
            "n_units": n_units,
            "total_spikes": sum(len(t) for t in spike_trains),
            "bin_frequency": bin_frequency,
        },
    }
