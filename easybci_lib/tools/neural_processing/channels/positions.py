"""Channel position extraction — 2D and 3D coordinates.

Useful for spatial models, topographic plots, and graph neural networks.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def get_positions(
    data_dict: Dict[str, Any],
    dims: int = 3,
    montage_name: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Extract channel positions from loaded data.

    Parameters
    ----------
    data_dict : dict
        Output of load_neural(). Needs "channels" and optionally "meta.positions_3d".
    dims : int
        2 or 3 spatial dimensions.
    montage_name : str or None
        Standard MNE montage name (e.g. "standard_1020"). None → use raw positions.
    normalize : bool
        Min-max normalize to [0, 1].

    Returns
    -------
    ndarray shape (n_channels, dims)
        Channel positions. Unknown positions filled with -0.1.
    """
    channels = data_dict["channels"]
    meta = data_dict.get("meta", {})

    # Try to get from raw data positions
    if "positions_3d" in meta and dims == 3 and montage_name is None:
        positions = np.asarray(meta["positions_3d"], dtype=np.float32)
        if positions.shape == (len(channels), 3):
            if normalize:
                positions = _normalize(positions)
            return positions

    # Fall back to MNE montage
    import mne

    if montage_name is not None:
        montage = mne.channels.make_standard_montage(montage_name)
        ch_pos = montage.get_positions()["ch_pos"]

        if dims == 3:
            native_t = mne.channels.compute_native_head_t(montage)
            pos_map = {
                name: mne.transforms.apply_trans(native_t["trans"], pos).tolist()
                for name, pos in ch_pos.items()
                if not np.all(pos == 0)
            }
        else:
            pos_map = {name: pos[:2].tolist() for name, pos in ch_pos.items()}
    else:
        pos_map = {}

    # Build output array
    fill = -0.1
    positions = np.full((len(channels), dims), fill, dtype=np.float32)
    for i, ch in enumerate(channels):
        # Try exact match, then strip bipolar ref suffix
        name = ch.split("-")[0] if ch not in pos_map else ch
        if name in pos_map:
            positions[i] = pos_map[name][:dims]

    if normalize:
        valid = positions[positions[:, 0] != fill]
        if len(valid) > 0:
            positions[positions[:, 0] != fill] = _normalize(valid)

    return positions


def _normalize(pos: np.ndarray) -> np.ndarray:
    """Min-max normalize, handling constant dimensions."""
    pmin = pos.min(axis=0, keepdims=True)
    pmax = pos.max(axis=0, keepdims=True)
    denom = pmax - pmin
    denom[denom == 0] = 1.0
    return (pos - pmin) / denom
