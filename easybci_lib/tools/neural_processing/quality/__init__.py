"""Data quality validation — signal integrity and completeness checks."""

from .validators import validate_signal, check_channels, check_sampling_rate
from .completeness import check_segments_complete
from .trial_qc import trial_qc, filter_trials
from .label_quality import assess_label_quality
from .final_view import FinalDataView
