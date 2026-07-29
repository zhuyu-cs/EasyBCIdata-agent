"""AI-ready output — unified serialization with metadata.

Default format is pkl (dict: {data, labels, meta}).
Alternative formats: hdf5, npz, mat — all share the same logical schema.
"""

from .formatter import (
    save_pkl,
    load_pkl,
    save_npz,
    load_npz,
    save_mat,
    load_mat,
    save_output,
    SUPPORTED_FORMATS,
    FORMAT_DESCRIPTIONS,
)
from .batch import build_batch
from .meta import collect_meta
