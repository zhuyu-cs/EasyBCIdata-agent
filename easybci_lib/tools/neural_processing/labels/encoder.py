"""Label encoding — categorical labels to integers or one-hot.

Clean implementation borrowing the key patterns from neuralset's LabelEncoder:
- Predefined mapping support (fixed label→index, reproducible)
- Missing value handling (treat as separate class or ignore)
- One-hot output option

Stripped: pydantic, torch, BaseExtractor inheritance.
"""

from typing import Dict, List, Optional, Set

import numpy as np


class LabelEncoder:
    """Encode categorical labels to integer indices or one-hot vectors.

    >>> enc = LabelEncoder()
    >>> enc.fit(["left", "right", "rest"])
    >>> enc.encode(["left", "right", "left"])
    array([0, 1, 0])
    >>> enc.one_hot(["right"])
    array([[0., 1., 0.]])
    """

    def __init__(
        self,
        predefined_mapping: Optional[Dict[str, int]] = None,
        handle_missing: str = "error",
    ):
        """
        Parameters
        ----------
        predefined_mapping : dict or None
            Fixed label → index. If None, built from data during fit().
        handle_missing : str
            "error" — raise on unknown labels
            "ignore" — encode as -1 (use ignore_index in loss)
            "separate" — add a dedicated class for missing
        """
        self.predefined_mapping = predefined_mapping
        self.handle_missing = handle_missing
        self._map: Dict[str, int] = {}
        self._n_classes: int = 0
        self._fitted = False

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def classes(self) -> List[str]:
        """Class names in index order."""
        inv = {v: k for k, v in self._map.items()}
        return [inv[i] for i in range(self._n_classes)]

    def fit(self, labels) -> "LabelEncoder":
        """Build mapping from observed labels.

        Parameters
        ----------
        labels : iterable of str
            All labels in the dataset.
        """
        unique = sorted(set(labels))

        if self.predefined_mapping:
            unknown = set(unique) - set(self.predefined_mapping)
            if unknown and self.handle_missing == "error":
                import logging
                logging.getLogger(__name__).warning(
                    "Labels %s not in predefined_mapping %s — adding them with new indices.",
                    sorted(unknown), sorted(self.predefined_mapping),
                )
                # Auto-extend mapping for unknown labels
                next_idx = max(self.predefined_mapping.values()) + 1 if self.predefined_mapping else 0
                extended = dict(self.predefined_mapping)
                for lbl in sorted(unknown):
                    extended[lbl] = next_idx
                    next_idx += 1
                self._map = extended
            else:
                self._map = dict(self.predefined_mapping)
        else:
            self._map = {label: i for i, label in enumerate(unique)}

        self._n_classes = len(self._map)
        if self.handle_missing == "separate":
            self._n_classes += 1

        self._fitted = True
        return self

    def encode(self, labels: List[str]) -> np.ndarray:
        """Encode labels → integer indices.

        Returns ndarray of int64. Missing labels get -1 when handle_missing != "error".
        """
        if not self._fitted:
            import logging
            logging.getLogger(__name__).warning("LabelEncoder.encode() called before fit() — auto-fitting from input.")
            self.fit(labels)

        result = np.empty(len(labels), dtype=np.int64)
        for i, label in enumerate(labels):
            if label in self._map:
                result[i] = self._map[label]
            elif self.handle_missing == "error":
                import logging
                logging.getLogger(__name__).warning("Unknown label %r — encoding as -1.", label)
                result[i] = -1
            else:
                result[i] = -1

        return result

    def one_hot(self, labels: List[str]) -> np.ndarray:
        """Encode labels → one-hot float32 array.

        Shape: (n_labels, n_classes). Missing → all zeros or last class.
        """
        indices = self.encode(labels)
        out = np.zeros((len(indices), self._n_classes), dtype=np.float32)

        for i, idx in enumerate(indices):
            if idx >= 0:
                out[i, idx] = 1.0
            elif self.handle_missing == "separate":
                out[i, -1] = 1.0
            # "ignore" → all-zero row

        return out

    def decode(self, indices: np.ndarray) -> List[str]:
        """Inverse transform: indices → label strings."""
        inv = {v: k for k, v in self._map.items()}
        return [inv.get(int(i), "<missing>") for i in indices]
