"""Core abstractions shared across all skills.

Kept deliberately minimal:
- Frequency: Hz ↔ sample conversion (the one abstraction worth keeping)
- TimedArray: numpy array + time metadata (proven useful for overlap/alignment)
- Simple constants
"""

from .timed_array import Frequency, TimedArray
from .constants import SUPPORTED_MODALITIES, NEURAL_FILE_EXTENSIONS
