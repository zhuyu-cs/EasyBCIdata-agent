"""Cross-subject channel alignment.

The valuable pattern from neuralset: when combining data from multiple
subjects with different channel sets, you need a consistent index.

Two modes:
- "unique": global index grows as new channels appear (use for group-level models)
- "original": per-recording order (use for subject-specific models with fixed dim)
"""

from typing import Dict, List, Literal, Optional


class ChannelMapper:
    """Maps channel names → integer indices across recordings.

    Agent use: create once, call update() per recording, then get_indices()
    when building output arrays.

    >>> mapper = ChannelMapper("unique")
    >>> mapper.update(["Fp1", "Fp2", "F3"])   # subject 1
    >>> mapper.update(["Fp1", "C3", "C4"])     # subject 2
    >>> mapper.n_channels  # 5 unique
    5
    >>> mapper.get_indices(["Fp1", "C3"])
    [0, 3]
    """

    def __init__(self, mode: str = "unique"):
        if mode not in ("unique", "original"):
            import logging
            logging.getLogger(__name__).warning(
                "ChannelMapper mode '%s' not recognized, defaulting to 'unique'.", mode
            )
            mode = "unique"
        self.mode = mode
        self._map: Dict[str, int] = {}

    @property
    def n_channels(self) -> int:
        return len(self._map)

    @property
    def names(self) -> List[str]:
        """Channel names in index order."""
        return sorted(self._map, key=self._map.get)

    def update(self, ch_names: List[str]) -> None:
        """Register channels from one recording."""
        if self.mode == "original":
            for i, ch in enumerate(ch_names):
                self._map[ch] = i
        else:
            for ch in ch_names:
                if ch not in self._map:
                    self._map[ch] = len(self._map)

    def get_indices(self, ch_names: List[str]) -> List[int]:
        """Look up indices. Returns sequential indices for unknown channels."""
        if not self._map:
            self.update(ch_names)
        indices = []
        for ch in ch_names:
            if ch in self._map:
                indices.append(self._map[ch])
            else:
                # Auto-register unknown channels
                new_idx = len(self._map)
                self._map[ch] = new_idx
                indices.append(new_idx)
        return indices

    @property
    def output_size(self) -> int:
        """Size of the output channel dimension."""
        return max(self._map.values()) + 1 if self._map else 0
