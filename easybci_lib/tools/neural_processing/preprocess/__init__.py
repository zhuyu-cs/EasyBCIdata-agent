"""Neural signal preprocessing — configurable pipeline.

Agent-callable interface:
    result = preprocess(data_dict, steps=["notch:50", "bandpass:1-40", "resample:256", "scale:robust"])
    # Returns: same dict structure as io.load_neural(), data transformed in place
"""

from .pipeline import preprocess, AVAILABLE_STEPS
from .spikes import bin_spikes
from .alignment import align_to_master, StreamData, AlignedBundle
from .event_tracking import EventTransformLog, rescale_events
from .memory_strategy import (
    compute_execution_strategy,
    estimate_file_memory_mb,
    format_strategy_report,
    ExecutionStrategy,
)
