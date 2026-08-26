"""Batch scan-root widening: comprehensive enumeration at scan time.

Regression guard for the SEEG_ZHU under-coverage bug: the agent passed a
too-deep ``source_root`` (``…/NKT/EEG2100``) so a sibling subtree
(``…/SEEG/NKT/EEG2100``) with more valid signal files was never walked. The fix
widens the scan root to the dataset root (recovered mechanically from the
``*_preprocess_work_dir(s)`` output convention) BEFORE walking — no post-hoc
"detect missed files and add them back" step.
"""
from __future__ import annotations

import os

from easybci_lib.tools.neural_processing.batch.coverage import (
    _is_strict_ancestor,
    dataset_root_from_output_dir,
    enumerate_signal_inputs,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # helpers only stat/walk; contents irrelevant


def _build_two_subtree_dataset(root):
    """Mirror the real bug layout under ``root`` (a tmp_path).

    root/patient/NKT/EEG2100/DT000{1,2}.EEG (+ .21E/.LOG sidecars) + DQ0001.EEG
    root/patient/SEEG/NKT/EEG2100/DT900{1,2}.EEG           # sibling subtree
    root/patient_preprocess_work_dirs/…/DTZZZZ.EEG          # output — must be excluded
    """
    patient = root / "patient"
    a = patient / "NKT" / "EEG2100"          # narrow scan root
    b = patient / "SEEG" / "NKT" / "EEG2100"  # sibling subtree
    for stem in ("DT0001", "DT0002"):
        _touch(a / f"{stem}.EEG")
        _touch(a / f"{stem}.21E")   # NK sidecar -> maps to sibling .EEG
        _touch(a / f"{stem}.LOG")   # NK sidecar
    _touch(a / "DQ0001.EEG")        # scalp signal file (still a .EEG signal)
    for stem in ("DT9001", "DT9002"):
        _touch(b / f"{stem}.EEG")
        _touch(b / f"{stem}.21E")
    # Output container that a widened walk must NOT re-ingest.
    out = root / "patient_preprocess_work_dirs"
    _touch(out / "preprocessed_output" / "preprocessed" / "DTZZZZ.EEG")
    return {"patient": patient, "narrow": a, "sibling": b, "out": out}


def test_dataset_root_from_output_dir_plural_and_singular(tmp_path):
    root = tmp_path / "SEEG_ZHU"
    root.mkdir()
    # plural batch-container form
    out_plural = root / "SEEG_ZHU_preprocess_work_dirs"
    assert dataset_root_from_output_dir(str(out_plural)) == str(root)
    # singular single-file form
    out_singular = root / "patient_preprocess_work_dir"
    assert dataset_root_from_output_dir(str(out_singular)) == str(root)
    # nested path INSIDE a work_dir still recovers the dataset root
    nested = out_plural / "middle_process" / "inspect"
    assert dataset_root_from_output_dir(str(nested)) == str(root)
    # no marker -> None
    assert dataset_root_from_output_dir(str(root / "random_out")) is None
    assert dataset_root_from_output_dir("") is None


def test_strict_ancestor():
    assert _is_strict_ancestor("/a/b", "/a/b/c/d") is True
    assert _is_strict_ancestor("/a/b", "/a/b") is False       # equal, not strict
    assert _is_strict_ancestor("/a/b/c", "/a/b") is False      # child, not ancestor
    assert _is_strict_ancestor("/a/x", "/a/y/z") is False      # sibling branch


def test_widen_recovers_sibling_subtree(tmp_path):
    """The core bug: narrow root misses the sibling subtree; dataset root gets all."""
    layout = _build_two_subtree_dataset(tmp_path)
    dataset_root = dataset_root_from_output_dir(str(layout["out"]))
    assert dataset_root == str(tmp_path)
    assert _is_strict_ancestor(dataset_root, str(layout["narrow"])) is True

    excl = [str(layout["out"])]
    narrow = enumerate_signal_inputs(str(layout["narrow"]), [".eeg"], exclude_under=excl)
    wide = enumerate_signal_inputs(dataset_root, [".eeg"], exclude_under=excl)

    # narrow sees only NKT/EEG2100: DT0001, DT0002, DQ0001 = 3
    assert len(narrow) == 3
    # wide adds the sibling subtree's DT9001, DT9002 = 5 total
    assert len(wide) == 5
    assert len(wide) > len(narrow)  # this is the widen trigger condition

    names = {os.path.basename(p) for p in wide}
    assert names == {"DT0001.EEG", "DT0002.EEG", "DQ0001.EEG", "DT9001.EEG", "DT9002.EEG"}


def test_sidecars_dropped_and_output_subtree_excluded(tmp_path):
    layout = _build_two_subtree_dataset(tmp_path)
    dataset_root = str(tmp_path)
    wide = enumerate_signal_inputs(
        dataset_root, [".eeg"], exclude_under=[str(layout["out"])]
    )
    # No NK sidecars routed as standalone inputs.
    assert not any(p.upper().endswith((".21E", ".LOG")) for p in wide)
    # The output container's DTZZZZ.EEG must never appear.
    assert not any("_preprocess_work_dir" in p for p in wide)
    assert not any(os.path.basename(p) == "DTZZZZ.EEG" for p in wide)


def test_no_widen_when_single_subtree(tmp_path):
    """JIANG-like layout: all files under one subtree → widen yields no more."""
    patient = tmp_path / "patient"
    a = patient / "NKT" / "EEG2100"
    for stem in ("DT0001", "DT0002", "DT0003"):
        _touch(a / f"{stem}.EEG")
    out = tmp_path / "patient_preprocess_work_dirs"
    out.mkdir()
    dataset_root = dataset_root_from_output_dir(str(out))
    excl = [str(out)]
    narrow = enumerate_signal_inputs(str(a), [".eeg"], exclude_under=excl)
    wide = enumerate_signal_inputs(dataset_root, [".eeg"], exclude_under=excl)
    assert len(narrow) == len(wide) == 3  # widen would NOT trigger (not strictly more)


def test_no_widen_when_output_off_tree(tmp_path):
    """Output on a different tree → dataset root not an ancestor → no widen."""
    data = tmp_path / "data" / "patient" / "NKT" / "EEG2100"
    _touch(data / "DT0001.EEG")
    off = tmp_path / "elsewhere" / "out_preprocess_work_dir"
    off.mkdir(parents=True)
    dataset_root = dataset_root_from_output_dir(str(off))
    # dataset_root is tmp_path/elsewhere — NOT an ancestor of the data scan root.
    assert _is_strict_ancestor(dataset_root, str(data)) is False
