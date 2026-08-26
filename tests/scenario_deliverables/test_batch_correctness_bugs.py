"""Regression tests for silent correctness bugs found in batch-EEG review.

Each test pins a specific bug that produced wrong-but-silent output:

1. epoch/label misalignment when out-of-bounds events are dropped
   (covered in test_bids_events.py).
2. _temporal_split ignored temporal_gap (`max(n-gap, n)` == n), reintroducing
   train/test temporal leakage.
3. _discover_events_csv missed entity-reduced BIDS events sidecars
   (data has run-, events shared above run level).
4. NWB trials operator-precedence bug leaked start_time/stop_time/id into
   per-trial metadata when the trials table had no trial_type column.
5. BIDS channels/electrodes sidecar fallback used an unconstrained glob that
   attached another subject's sidecar in a flat multi-subject directory.
"""

import tempfile
from pathlib import Path

import numpy as np


# --- Bug 2: temporal split must reserve the gap -----------------------------


def _build_temporal_split():
    from easybci_lib.tools.neural_processing.codegen.generator import (
        generate_split_code,
    )

    src = generate_split_code(
        {"method": "temporal", "temporal_gap": 10,
         "ratios": {"train": 0.7, "test": 0.3}}
    )
    ns = {"np": np}
    exec(compile(src, "<split>", "exec"), ns)
    return ns["_temporal_split"]


def test_temporal_split_reserves_gap():
    fn = _build_temporal_split()
    res = fn(100, {"train": 0.7, "test": 0.3}, 10)
    counts = {}
    for x in res:
        counts[x] = counts.get(x, 0) + 1
    # The 10-sample gap must actually be reserved (anti-leakage). Before the
    # fix, `usable = max(n-gap, n)` == n, so no gap rows appeared.
    assert counts.get("gap", 0) == 10
    # train/test are split over the remaining 90 usable samples.
    assert counts.get("train", 0) + counts.get("test", 0) == 90


def test_temporal_split_gap_larger_than_data_is_safe():
    fn = _build_temporal_split()
    # gap bigger than n must not crash or produce negative usable.
    res = fn(5, {"train": 0.7, "test": 0.3}, 100)
    assert len(res) == 5


# --- Bug 3: entity-reduced BIDS events discovery ----------------------------


def test_discover_events_entity_reduced():
    from easybci_lib.tools.neural_processing.io.deep_inspect import (
        _discover_events_csv,
    )

    d = Path(tempfile.mkdtemp())
    (d / "sub-01_ses-1_task-x_run-1_eeg.edf").write_text("x")
    # events shared one entity level above the data file (no run-)
    (d / "sub-01_ses-1_task-x_events.tsv").write_text("onset\tduration\n0.1\t0.2\n")
    got = _discover_events_csv(d / "sub-01_ses-1_task-x_run-1_eeg.edf")
    assert got is not None
    assert got.endswith("sub-01_ses-1_task-x_events.tsv")


def test_discover_events_does_not_cross_subject():
    from easybci_lib.tools.neural_processing.io.deep_inspect import (
        _discover_events_csv,
    )

    d = Path(tempfile.mkdtemp())
    (d / "sub-02_ses-1_task-x_run-1_eeg.edf").write_text("x")
    # a DIFFERENT subject's events file must never be matched
    (d / "sub-01_task-x_events.tsv").write_text("onset\n0.1\n")
    got = _discover_events_csv(d / "sub-02_ses-1_task-x_run-1_eeg.edf")
    assert got is None


def test_discover_events_exact_bids_still_works():
    from easybci_lib.tools.neural_processing.io.deep_inspect import (
        _discover_events_csv,
    )

    d = Path(tempfile.mkdtemp())
    (d / "sub-01_task-x_eeg.edf").write_text("x")
    (d / "sub-01_task-x_events.tsv").write_text("onset\n0.1\n")
    got = _discover_events_csv(d / "sub-01_task-x_eeg.edf")
    assert got is not None
    assert got.endswith("sub-01_task-x_events.tsv")


# --- Bug 5: BIDS channels/electrodes must not cross subjects ----------------


def test_bids_sidecar_channels_no_cross_subject():
    from easybci_lib.tools.neural_processing.io.bids_detector import (
        _find_bids_sidecars,
    )

    d = Path(tempfile.mkdtemp())
    data = d / "sub-02_task-x_eeg.edf"
    data.write_text("x")
    # only ANOTHER subject's channels sidecar exists — must NOT be attached
    (d / "sub-01_task-x_channels.tsv").write_text("name\ttype\nCz\tEEG\n")
    assoc = _find_bids_sidecars(
        data, d, {"subject": "sub-02", "task": "x"}
    )
    assert "channels" not in assoc


def test_bids_sidecar_channels_same_subject_matched():
    from easybci_lib.tools.neural_processing.io.bids_detector import (
        _find_bids_sidecars,
    )

    d = Path(tempfile.mkdtemp())
    data = d / "sub-02_task-x_run-1_eeg.edf"
    data.write_text("x")
    # this subject's channels sidecar, shared above run level
    (d / "sub-02_task-x_channels.tsv").write_text("name\ttype\nCz\tEEG\n")
    assoc = _find_bids_sidecars(
        data, d, {"subject": "sub-02", "task": "x", "run": "1"}
    )
    assert "channels" in assoc
    assert assoc["channels"]["path"].endswith("sub-02_task-x_channels.tsv")
