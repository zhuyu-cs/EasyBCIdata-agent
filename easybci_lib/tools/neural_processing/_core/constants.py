"""Shared constants."""

from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "bci_skills"

SUPPORTED_MODALITIES = ("eeg", "meg", "ieeg", "spike", "emg")

NEURAL_FILE_EXTENSIONS = {
    ".fif": "mne",
    ".edf": "mne",
    ".bdf": "mne",
    ".set": "mne",
    ".ds": "mne",
    ".cnt": "mne",
    ".gdf": "mne",
    ".nwb": "hdf5",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}

DEFAULT_IGNORE_INDEX = -100
