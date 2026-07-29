"""Feature extraction module for neural data pipeline.

Provides pipeline-compatible feature extraction steps that transform
preprocessed continuous/epoched data into ML-ready feature matrices.
"""

from .extractors import (
    extract_psd_bands,
    extract_csp,
    extract_tfr,
    extract_connectivity,
    FeatureResult,
)

__all__ = [
    "extract_psd_bands",
    "extract_csp",
    "extract_tfr",
    "extract_connectivity",
    "FeatureResult",
]
