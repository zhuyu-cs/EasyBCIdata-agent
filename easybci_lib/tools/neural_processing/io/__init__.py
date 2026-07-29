"""Neural data I/O — load any format into numpy arrays.

Agent-callable interface:
    result = load_neural("path/to/file.edf")
    # Returns: {"data": ndarray, "frequency": float, "channels": [...], "meta": {...}}

    events = load_events("path/to/events.csv")
    # Returns: [{"onset": float, "duration": float, "type": str, "metadata": dict}]

    sidecars = detect_sidecar_files("path/to/data.edf")
    # Returns: {"sidecar_files": [...], "data_type": str, "relationships": dict}

    label_info = classify_label_type("path/to/labels.csv", n_samples=76800)
    # Returns: {"label_type": LabelType, "strategy": str, "confidence": float, ...}
"""

from .loader import load_neural
from .event_loader import load_events, detect_time_unit
from .sidecar_detector import detect_sidecar_files, build_event_source_report
from .label_classifier import classify_label_type, LabelType
from .bids_detector import detect_bids_structure
from .hierarchical_labels import parse_hierarchical_labels, HierarchicalLabels
