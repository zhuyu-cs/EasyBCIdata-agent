"""Hierarchical label parsing — multi-level nested experiment structures.

Handles L5 labels where experiments have nested designs:
  Session → Block → Trial → Sample

Input formats:
- Nested JSON with session/block/trial hierarchy
- Multi-level CSV with columns for each hierarchy level
- BIDS-style with run/task/trial levels

Provides flattening at any specified level for downstream epoching.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class HierarchicalLabels:
    """Parsed hierarchical label structure with level-based access."""

    def __init__(
        self,
        levels: List[str],
        tree: List[Dict[str, Any]],
        source: str = "unknown",
    ):
        """
        Parameters
        ----------
        levels : list of str
            Level names from coarsest to finest (e.g., ["session", "block", "trial"]).
        tree : list of dict
            Flattened tree — each dict represents a leaf node with all level values
            plus timing info. Keys: level names + "onset", "duration", "end", "metadata".
        source : str
            Where the labels came from.
        """
        self.levels = levels
        self.tree = tree
        self.source = source

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def n_leaves(self) -> int:
        return len(self.tree)

    def get_level_values(self, level: str) -> List[str]:
        """Get unique values at a specific hierarchy level."""
        if level not in self.levels:
            import logging
            logging.getLogger(__name__).warning(
                "Level '%s' not in %s — returning empty. Available levels: %s",
                level, self.levels, self.levels,
            )
            return []
        return sorted(set(node.get(level, "unknown") for node in self.tree))

    def flatten_at_level(self, level: str) -> List[Dict[str, Any]]:
        """Flatten the hierarchy at the specified level for epoching.

        Returns events suitable for segment_data() or segment_by_intervals(),
        with parent-level info preserved in metadata.

        Parameters
        ----------
        level : str
            The level to flatten at. Events are grouped by this level.

        Returns
        -------
        List of event dicts with keys: onset, duration, label, metadata
        """
        if level not in self.levels:
            import logging
            logging.getLogger(__name__).warning(
                "Level '%s' not in %s — returning empty events.", level, self.levels,
            )
            return []
        parent_levels = self.levels[:level_idx]

        events = []
        for node in self.tree:
            label = str(node.get(level, "unknown"))
            onset = node.get("onset")
            duration = node.get("duration", 0.0)
            end = node.get("end")

            if onset is None:
                continue

            if end is not None and duration == 0.0:
                duration = end - onset

            metadata = {}
            for parent in parent_levels:
                metadata[parent] = node.get(parent, "unknown")
            if node.get("metadata"):
                metadata.update(node["metadata"])

            events.append({
                "onset": float(onset),
                "duration": float(duration),
                "label": label,
                "start": float(onset),
                "end": float(onset + duration) if duration > 0 else float(onset),
                "metadata": metadata,
            })

        events.sort(key=lambda e: e["onset"])
        return events

    def group_by_level(self, level: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group leaves by the specified level value.

        Returns
        -------
        Dict mapping level value → list of leaf nodes
        """
        if level not in self.levels:
            import logging
            logging.getLogger(__name__).warning(
                "Level '%s' not in %s — returning empty groups.", level, self.levels,
            )
            return {}

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for node in self.tree:
            key = str(node.get(level, "unknown"))
            groups.setdefault(key, []).append(node)
        return groups

    def summary(self) -> Dict[str, Any]:
        """Produce a summary of the hierarchical structure."""
        level_info = {}
        for lvl in self.levels:
            values = self.get_level_values(lvl)
            level_info[lvl] = {
                "n_unique": len(values),
                "values": values[:20],
            }

        has_timing = sum(1 for n in self.tree if n.get("onset") is not None)

        return {
            "n_levels": self.n_levels,
            "levels": self.levels,
            "n_leaves": self.n_leaves,
            "level_info": level_info,
            "has_timing": has_timing,
            "timing_coverage": has_timing / max(self.n_leaves, 1),
            "source": self.source,
        }


def parse_hierarchical_labels(
    source: Union[str, Dict, List],
    level_names: Optional[List[str]] = None,
) -> HierarchicalLabels:
    """Parse hierarchical labels from various sources.

    Parameters
    ----------
    source : str, dict, or list
        - str: path to JSON or CSV file
        - dict: nested JSON structure
        - list: list of flat dicts with level columns

    level_names : list of str, optional
        Explicit level names from coarsest to finest.
        If None, inferred from the data structure.

    Returns
    -------
    HierarchicalLabels object with level-based access methods.
    """
    if isinstance(source, str):
        return _parse_from_file(source, level_names)
    elif isinstance(source, dict):
        return _parse_from_nested_dict(source, level_names)
    elif isinstance(source, list):
        return _parse_from_flat_list(source, level_names)
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Unsupported source type %s for hierarchical labels — returning empty.", type(source),
        )
        return HierarchicalLabels(tree=[], levels=level_names or [], source="unknown")


def _parse_from_file(
    filepath: str,
    level_names: Optional[List[str]],
) -> HierarchicalLabels:
    """Parse from a file path."""
    p = Path(filepath)
    if not p.exists():
        import logging
        logging.getLogger(__name__).warning("Label file not found: %s — returning empty.", filepath)
        return HierarchicalLabels(tree=[], levels=level_names or [], source=filepath)

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return _parse_from_nested_dict(data, level_names, source=filepath)
        elif isinstance(data, list):
            return _parse_from_flat_list(data, level_names, source=filepath)
        import logging
        logging.getLogger(__name__).warning("JSON must be dict or list, got %s — treating as empty.", type(data))
        return HierarchicalLabels(tree=[], levels=level_names or [], source=filepath)

    if p.suffix in (".csv", ".tsv"):
        return _parse_from_tabular_file(filepath, level_names)

    import logging
    logging.getLogger(__name__).warning(
        "Unsupported hierarchical label format: %s — returning empty.", p.suffix,
    )
    return HierarchicalLabels(tree=[], levels=level_names or [], source=filepath)


def _parse_from_nested_dict(
    data: Dict[str, Any],
    level_names: Optional[List[str]] = None,
    source: str = "dict",
) -> HierarchicalLabels:
    """Parse from nested JSON structure.

    Expected structure:
    {
      "session": "post_training",  # or "level": "post_training"
      "blocks": [
        {"type": "left_hand", "trials": [
          {"onset": 2.0, "duration": 4.0, "outcome": "success"},
          ...
        ]}
      ]
    }
    """
    tree: List[Dict[str, Any]] = []
    detected_levels: List[str] = []

    _flatten_nested(data, {}, tree, detected_levels, depth=0)

    if level_names:
        levels = level_names
    elif detected_levels:
        levels = _deduplicate_order(detected_levels)
    else:
        levels = ["level_0"]

    return HierarchicalLabels(levels=levels, tree=tree, source=source)


def _flatten_nested(
    node: Any,
    context: Dict[str, str],
    output: List[Dict[str, Any]],
    detected_levels: List[str],
    depth: int,
) -> None:
    """Recursively flatten a nested structure into leaf nodes."""
    if isinstance(node, dict):
        current_context = dict(context)
        child_lists = []
        timing = {}
        metadata = {}

        for key, value in node.items():
            if key in ("onset", "start", "start_time"):
                timing["onset"] = float(value)
            elif key in ("duration",):
                timing["duration"] = float(value)
            elif key in ("end", "stop", "stop_time", "end_time"):
                timing["end"] = float(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                child_lists.append((key, value))
            elif isinstance(value, (str, int, float, bool)):
                # This is a level label
                level_name = _infer_level_name(key, depth)
                current_context[level_name] = str(value)
                if level_name not in detected_levels:
                    detected_levels.append(level_name)
            else:
                metadata[key] = value

        if child_lists:
            for list_key, children in child_lists:
                for child in children:
                    _flatten_nested(child, current_context, output, detected_levels, depth + 1)
        else:
            # Leaf node
            leaf = dict(current_context)
            leaf.update(timing)
            if metadata:
                leaf["metadata"] = metadata
            output.append(leaf)

    elif isinstance(node, list):
        for item in node:
            _flatten_nested(item, context, output, detected_levels, depth)


def _infer_level_name(key: str, depth: int) -> str:
    """Infer a level name from a JSON key."""
    level_keywords = {
        "session": "session",
        "block": "block",
        "trial": "trial",
        "type": f"type_L{depth}",
        "condition": "condition",
        "run": "run",
        "task": "task",
        "phase": "phase",
        "outcome": "outcome",
        "class": "class",
        "category": "category",
    }
    key_lower = key.lower()
    for kw, name in level_keywords.items():
        if kw in key_lower:
            return name
    return key


def _parse_from_flat_list(
    data: List[Dict[str, Any]],
    level_names: Optional[List[str]] = None,
    source: str = "list",
) -> HierarchicalLabels:
    """Parse from a flat list of dicts (each dict has columns for levels).

    Example:
    [
      {"session": "pre", "block": "rest", "trial": 1, "onset": 0.0, "duration": 4.0},
      {"session": "pre", "block": "rest", "trial": 2, "onset": 5.0, "duration": 4.0},
      ...
    ]
    """
    if not data:
        return HierarchicalLabels(levels=level_names or [], tree=[], source=source)

    timing_keys = {"onset", "start", "start_time", "duration", "end", "stop", "end_time", "stop_time"}
    meta_keys = {"metadata", "meta", "notes", "comment"}

    if level_names is None:
        sample_keys = set(data[0].keys())
        non_level = timing_keys | meta_keys
        level_names = [k for k in data[0].keys() if k not in non_level]

    tree = []
    for row in data:
        node: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}

        for key, value in row.items():
            if key in ("onset", "start", "start_time"):
                node["onset"] = float(value)
            elif key == "duration":
                node["duration"] = float(value)
            elif key in ("end", "stop", "stop_time", "end_time"):
                node["end"] = float(value)
            elif key in level_names:
                node[key] = str(value)
            else:
                metadata[key] = value

        if "onset" not in node and "start" in row:
            node["onset"] = float(row["start"])
        if metadata:
            node["metadata"] = metadata
        tree.append(node)

    return HierarchicalLabels(levels=level_names, tree=tree, source=source)


def _parse_from_tabular_file(
    filepath: str,
    level_names: Optional[List[str]],
) -> HierarchicalLabels:
    """Parse from CSV/TSV file with hierarchy columns."""
    p = Path(filepath)
    sep = "\t" if p.suffix == ".tsv" else ","

    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return HierarchicalLabels(levels=level_names or [], tree=[], source=filepath)

    headers = [h.strip() for h in lines[0].split(sep)]
    data = []
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(sep)]
        if len(parts) >= len(headers):
            row = dict(zip(headers, parts))
            data.append(row)

    return _parse_from_flat_list(data, level_names, source=filepath)


def _deduplicate_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
