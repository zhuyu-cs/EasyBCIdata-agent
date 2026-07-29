"""Output adapters for feature extraction results.

Converts FeatureResult objects into formats directly usable by
sklearn, PyTorch, and other ML frameworks.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def to_sklearn(features_dict: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Convert pipeline feature extraction output to sklearn format.

    Parameters
    ----------
    features_dict : dict
        The "features" key from pipeline result (may contain multiple extractors).

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Concatenated features from all extractors.
    y : ndarray or None
        Labels if available.
    meta : dict
        Feature names, extractor info, shapes.
    """
    X_parts = []
    feature_names = []
    y = None

    for extractor_name, feat_data in features_dict.items():
        X_part = feat_data["X"]
        if X_part.ndim > 2:
            X_part = X_part.reshape(X_part.shape[0], -1)
        X_parts.append(X_part)
        feature_names.extend(feat_data.get("feature_names", [f"{extractor_name}_{i}" for i in range(X_part.shape[1])]))

        if y is None and feat_data.get("y") is not None:
            y = feat_data["y"]

    X = np.concatenate(X_parts, axis=1) if X_parts else np.empty((0, 0))

    meta = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "extractors": list(features_dict.keys()),
    }

    return X, y, meta


def save_numpy(
    features_dict: Dict[str, Any],
    output_path: str,
) -> str:
    """Save features as .npz file for later loading.

    Parameters
    ----------
    features_dict : dict
        The "features" key from pipeline result.
    output_path : str
        Path to save the .npz file.

    Returns
    -------
    Path to saved file.
    """
    X, y, meta = to_sklearn(features_dict)
    save_dict = {"X": X, "feature_names": np.array(meta["feature_names"])}
    if y is not None:
        save_dict["y"] = y

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    return output_path
