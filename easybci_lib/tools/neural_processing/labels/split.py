"""Data splitting — train/val/test assignment.

Key patterns from neuralset:
- Group-aware splitting (all data from one subject/session stays together)
- Deterministic hash-based assignment (same content → same split, always)
- Stratified splitting via sklearn
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from easybci_lib.tools.neural_processing._seed import EASYBCI_SEED

logger = logging.getLogger(__name__)


def split_data(
    n_items: int,
    ratios: Optional[Dict[str, float]] = None,
    groups: Optional[List[str]] = None,
    stratify: Optional[List[str]] = None,
    method: str = "random",
    seed: int = EASYBCI_SEED,
    temporal_gap: int = 0,
    n_folds: int = 5,
) -> np.ndarray:
    """Assign items to train/val/test splits.

    Parameters
    ----------
    n_items : int
        Number of items (segments, trials, etc.)
    ratios : dict or None
        Split ratios. Default: {"train": 0.7, "val": 0.15, "test": 0.15}
    groups : list of str or None
        Group labels — items in same group stay in same split.
        Essential for preventing data leakage across subjects.
    stratify : list of str or None
        Stratification labels — maintain class balance in each split.
    method : str
        "random" — sklearn random split
        "hash" — deterministic hash-based (requires groups)
        "sequential" — first N% train, next M% val, rest test
        "temporal" — time-ordered split with optional gap (for L3 continuous)
        "group_kfold" — group k-fold cross-validation (returns fold indices)
        "loso" — leave-one-subject-out (returns fold per subject)
    seed : int
        Random state.
    temporal_gap : int
        Number of items to exclude between train and test in temporal split.
        Prevents temporal leakage from autocorrelated data.
    n_folds : int
        Number of folds for group_kfold method.

    Returns
    -------
    ndarray of str, shape (n_items,)
        Split assignment for each item ("train", "val", or "test").
        For group_kfold/loso: returns fold index as string ("fold_0", "fold_1", ...).
    """
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning("Ratios must sum to 1.0, got %s — normalizing", total)
        ratios = {k: v / total for k, v in ratios.items()}

    if method == "hash":
        return _hash_split(n_items, ratios, groups, seed)
    elif method == "sequential":
        return _sequential_split(n_items, ratios)
    elif method == "random":
        return _random_split(n_items, ratios, groups, stratify, seed)
    elif method == "temporal":
        return _temporal_split(n_items, ratios, temporal_gap)
    elif method == "group_kfold":
        return _group_kfold_split(n_items, groups, n_folds, seed)
    elif method == "loso":
        return _loso_split(n_items, groups)
    else:
        logger.warning("Unknown method: %s — falling back to 'random'", method)
        return _random_split(n_items, ratios, groups, stratify, seed)


def _random_split(
    n_items: int,
    ratios: Dict[str, float],
    groups: Optional[List[str]],
    stratify: Optional[List[str]],
    seed: int,
) -> np.ndarray:
    """Random split, optionally grouped and stratified."""
    from sklearn.model_selection import train_test_split

    splits = np.empty(n_items, dtype=object)
    split_names = list(ratios.keys())

    if groups is not None:
        # Split at group level
        unique_groups = list(set(groups))
        group_stratify = None
        if stratify:
            # Map group → majority class for stratification
            group_to_class = {}
            for g, s in zip(groups, stratify):
                group_to_class.setdefault(g, []).append(s)
            group_stratify = [
                max(set(group_to_class[g]), key=group_to_class[g].count)
                for g in unique_groups
            ]

        group_splits = _split_indices(
            len(unique_groups), ratios, group_stratify, seed
        )
        group_map = {g: group_splits[i] for i, g in enumerate(unique_groups)}
        for i, g in enumerate(groups):
            splits[i] = group_map[g]
    else:
        strat = stratify if stratify else None
        splits[:] = _split_indices(n_items, ratios, strat, seed)

    return splits


def _split_indices(
    n: int,
    ratios: Dict[str, float],
    stratify: Optional[List[str]],
    seed: int,
) -> np.ndarray:
    """Split indices into groups according to ratios."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(n)
    split_names = list(ratios.keys())
    result = np.empty(n, dtype=object)

    if len(split_names) == 1:
        result[:] = split_names[0]
        return result

    # Split off test first
    test_name = split_names[-1]
    test_ratio = ratios[test_name]

    strat_arr = np.array(stratify) if stratify else None
    idx_rest, idx_test = train_test_split(
        indices, test_size=test_ratio, random_state=seed,
        stratify=strat_arr,
    )
    result[idx_test] = test_name

    if len(split_names) == 2:
        result[idx_rest] = split_names[0]
        return result

    # Split remaining into train/val
    val_name = split_names[1]
    adjusted_ratio = ratios[val_name] / (1.0 - test_ratio)
    strat_rest = strat_arr[idx_rest] if strat_arr is not None else None

    idx_train, idx_val = train_test_split(
        idx_rest, test_size=adjusted_ratio, random_state=seed,
        stratify=strat_rest,
    )
    result[idx_train] = split_names[0]
    result[idx_val] = val_name

    return result


def _hash_split(
    n_items: int,
    ratios: Dict[str, float],
    groups: Optional[List[str]],
    seed: int,
) -> np.ndarray:
    """Deterministic hash-based split — same group always lands in same split."""
    if groups is None:
        groups = [str(i) for i in range(n_items)]

    split_names = list(ratios.keys())
    cumulative = np.cumsum([ratios[s] for s in split_names])
    result = np.empty(n_items, dtype=object)

    for i, group in enumerate(groups):
        h = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        for j, threshold in enumerate(cumulative):
            if val <= threshold:
                result[i] = split_names[j]
                break
        else:
            result[i] = split_names[-1]

    return result


def _sequential_split(n_items: int, ratios: Dict[str, float]) -> np.ndarray:
    """Sequential split — first chunk train, then val, then test."""
    result = np.empty(n_items, dtype=object)
    cumulative = np.cumsum(list(ratios.values()))
    names = list(ratios.keys())

    for i in range(n_items):
        pos = i / n_items
        for j, threshold in enumerate(cumulative):
            if pos < threshold:
                result[i] = names[j]
                break
        else:
            result[i] = names[-1]

    return result


def _temporal_split(
    n_items: int,
    ratios: Dict[str, float],
    temporal_gap: int = 0,
) -> np.ndarray:
    """Temporal split — time-ordered with optional gap to prevent leakage.

    Items are ordered by time. A gap of `temporal_gap` items is excluded
    between train and test to avoid autocorrelation leakage.
    """
    result = np.empty(n_items, dtype=object)
    split_names = list(ratios.keys())

    # Calculate split boundaries accounting for gaps
    n_gaps = len(split_names) - 1
    total_gap = temporal_gap * n_gaps
    usable = n_items - total_gap
    if usable <= 0:
        logger.warning(
            "temporal_gap=%d too large for n_items=%d — ignoring gap",
            temporal_gap, n_items,
        )
        total_gap = 0
        usable = n_items
        temporal_gap = 0

    # Assign splits sequentially with gaps
    cumulative_ratios = np.cumsum([ratios[s] for s in split_names])
    boundaries = [int(r * usable) for r in cumulative_ratios]
    boundaries[-1] = usable  # ensure last boundary covers all

    pos = 0
    for split_idx, name in enumerate(split_names):
        n_in_split = boundaries[split_idx] - (boundaries[split_idx - 1] if split_idx > 0 else 0)
        for _ in range(n_in_split):
            if pos < n_items:
                result[pos] = name
                pos += 1
        # Insert gap (marked as "gap" — excluded from training/eval)
        if split_idx < len(split_names) - 1 and temporal_gap > 0:
            for _ in range(temporal_gap):
                if pos < n_items:
                    result[pos] = "gap"
                    pos += 1

    # Fill any remaining items
    while pos < n_items:
        result[pos] = split_names[-1]
        pos += 1

    return result


def _group_kfold_split(
    n_items: int,
    groups: Optional[List[str]],
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Group k-fold split — assign fold indices respecting group boundaries.

    All items from the same group land in the same fold.
    Returns fold assignments as "fold_0", "fold_1", etc.
    """
    result = np.empty(n_items, dtype=object)

    if groups is None:
        # Without groups, just do regular k-fold
        rng = np.random.RandomState(seed)
        fold_indices = rng.randint(0, n_folds, size=n_items)
        for i in range(n_items):
            result[i] = f"fold_{fold_indices[i]}"
        return result

    # Group-level fold assignment
    unique_groups = sorted(set(groups))
    n_groups = len(unique_groups)

    if n_folds > n_groups:
        logger.warning(
            "n_folds=%d > n_groups=%d — reducing to %d folds",
            n_folds, n_groups, n_groups,
        )
        n_folds = n_groups

    # Shuffle groups deterministically then assign round-robin to folds
    rng = np.random.RandomState(seed)
    group_order = list(range(n_groups))
    rng.shuffle(group_order)

    group_to_fold = {}
    for idx, g_idx in enumerate(group_order):
        group_to_fold[unique_groups[g_idx]] = idx % n_folds

    for i, g in enumerate(groups):
        result[i] = f"fold_{group_to_fold[g]}"

    return result


def _loso_split(
    n_items: int,
    groups: Optional[List[str]],
) -> np.ndarray:
    """Leave-one-subject-out split — each unique group gets its own fold.

    Returns fold indices named after the group ("fold_sub01", etc.).
    For use in cross-validation where each subject is held out once.
    """
    result = np.empty(n_items, dtype=object)

    if groups is None:
        # Without groups, every item is its own fold (degenerates to LOO)
        for i in range(n_items):
            result[i] = f"fold_{i}"
        return result

    for i, g in enumerate(groups):
        result[i] = f"fold_{g}"

    return result


def recommend_split_strategy(
    label_type: str = "none",
    n_subjects: int = 1,
    n_trials_per_class: Optional[Dict[str, int]] = None,
    paradigm: str = "default",
    data_duration_s: float = 0.0,
) -> Dict[str, Any]:
    """Recommend a data splitting strategy based on label type and data characteristics.

    Parameters
    ----------
    label_type : str
        Detected label type: "L1_event", "L2_segment", "L3_continuous",
        "L4_session", "L5_hierarchical", or "none".
    n_subjects : int
        Number of subjects in the dataset.
    n_trials_per_class : dict, optional
        Number of trials per class label.
    paradigm : str
        Processing paradigm.
    data_duration_s : float
        Total data duration in seconds.

    Returns
    -------
    Dict with:
        method: str — recommended split method
        ratios: dict — recommended train/val/test ratios
        params: dict — additional parameters for the method
        rationale: str — explanation of why this strategy was chosen
        warnings: list[str] — potential data leakage risks
    """
    if n_trials_per_class is None:
        n_trials_per_class = {}

    warnings: List[str] = []
    total_trials = sum(n_trials_per_class.values()) if n_trials_per_class else 0
    min_class_count = min(n_trials_per_class.values()) if n_trials_per_class else 0

    # L3 continuous → temporal split (prevent autocorrelation leakage)
    if label_type == "L3_continuous":
        # Gap size: ~2 seconds worth of windows to decorrelate
        gap_windows = max(2, int(2.0 * 1.0))  # 2s / stride approximation
        return {
            "method": "temporal",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "params": {"temporal_gap": gap_windows},
            "rationale": (
                "L3 continuous labels indicate time-series regression/classification. "
                "Temporal split preserves time ordering and inserts a gap between splits "
                "to prevent autocorrelation leakage. Adjacent time windows share signal content."
            ),
            "warnings": [
                "Random split would cause severe temporal leakage for continuous data.",
                "Ensure training does not see future data samples.",
            ],
        }

    # L4 session-level → LOSO if multiple subjects, else sequential
    if label_type == "L4_session":
        if n_subjects > 1:
            return {
                "method": "loso",
                "ratios": {"train": 0.8, "test": 0.2},
                "params": {},
                "rationale": (
                    f"L4 session labels + {n_subjects} subjects → leave-one-subject-out. "
                    f"Each subject has a single label, so splits must be at subject level."
                ),
                "warnings": [
                    "Session-level labels cannot be split within a subject.",
                ],
            }
        return {
            "method": "sequential",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "params": {},
            "rationale": (
                "L4 session label with single subject — sequential split preserves "
                "temporal ordering within the session."
            ),
            "warnings": [],
        }

    # L5 hierarchical → group_kfold at outermost level
    if label_type == "L5_hierarchical":
        return {
            "method": "group_kfold",
            "ratios": {"train": 0.8, "test": 0.2},
            "params": {"n_folds": min(5, max(2, n_subjects))},
            "rationale": (
                "L5 hierarchical labels — split at the outermost nesting level "
                "to prevent information leakage across hierarchy boundaries. "
                "Group k-fold ensures all nested items stay together."
            ),
            "warnings": [
                "Splitting within a hierarchical group causes leakage.",
            ],
        }

    # L1/L2 with multiple subjects → group_kfold by subject
    if label_type in ("L1_event", "L2_segment") and n_subjects > 5:
        return {
            "method": "group_kfold",
            "ratios": {"train": 0.8, "test": 0.2},
            "params": {"n_folds": min(5, n_subjects)},
            "rationale": (
                f"{label_type} with {n_subjects} subjects → group k-fold by subject. "
                f"Prevents same subject's data from appearing in both train and test."
            ),
            "warnings": [
                "Random split across subjects causes data leakage "
                "(model learns subject-specific patterns, not task-general ones).",
            ],
        }

    # L1/L2 with single subject → random + stratified
    if label_type in ("L1_event", "L2_segment") and n_subjects <= 1:
        use_stratify = min_class_count >= 5 and len(n_trials_per_class) > 1
        return {
            "method": "random",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "params": {"stratify": use_stratify, "seed": 42},
            "rationale": (
                f"{label_type} with single subject — stratified random split "
                f"maintains class balance. "
                + (f"Classes: {list(n_trials_per_class.keys())[:5]}." if n_trials_per_class else "")
            ),
            "warnings": (
                ["Small class counts — stratification may fail for minor classes."]
                if min_class_count < 10 else []
            ),
        }

    # L1/L2 with 2-5 subjects → LOSO if feasible, else group_kfold
    if label_type in ("L1_event", "L2_segment") and 2 <= n_subjects <= 5:
        if n_subjects <= 3 and total_trials > 100:
            return {
                "method": "loso",
                "ratios": {"train": 0.8, "test": 0.2},
                "params": {},
                "rationale": (
                    f"{n_subjects} subjects with sufficient trials → LOSO. "
                    f"Each subject held out once for robust generalization estimate."
                ),
                "warnings": [],
            }
        return {
            "method": "group_kfold",
            "ratios": {"train": 0.8, "test": 0.2},
            "params": {"n_folds": n_subjects},
            "rationale": (
                f"{n_subjects} subjects → group k-fold (one fold per subject)."
            ),
            "warnings": [],
        }

    # Default: random split
    return {
        "method": "random",
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "params": {"seed": 42},
        "rationale": (
            "No specific label type detected — using random split as default. "
            "Consider providing group labels if multi-subject data is involved."
        ),
        "warnings": [
            "Random split without group awareness may cause data leakage "
            "if items from the same source are correlated.",
        ],
    }
