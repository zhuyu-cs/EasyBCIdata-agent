"""End-to-end smoke test against the real 1.5 GB Compumedics .SLP bundle.

Run from EasyBCIdata-agent/:
    venv/bin/python tests/psg/test_real_slp.py
"""
import sys
import time
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SLP_PATH = "/home/zhuyu/zhuyu_work_dir/test_data/PSG/EB560E98-28CF-4942-8BFA-ECC8C6E8C894.SLP"


def test_inspect_only():
    """Light inspect — metadata only, should be fast (<1s)."""
    from easybci_lib.tools.neural_processing.io.loader import load_neural

    t0 = time.perf_counter()
    r = load_neural(SLP_PATH, target_hz=100.0, inspect_only=True)
    elapsed = time.perf_counter() - t0

    m = r["meta"]
    print(f"[inspect_only] elapsed: {elapsed:.2f}s")
    print(f"  format:       {m['format']}")
    print(f"  n_channels:   {m['n_channels']}")
    print(f"  duration:     {r['duration']:.1f}s ({r['duration']/3600:.2f}h)")
    print(f"  native_rates: {m['native_rates']}")
    print(f"  device:       {m.get('device', '?')}")
    print(f"  data_unit:    {m.get('data_unit', '?')}")
    assert m["format"] == "compumedics_slp"
    assert m["n_channels"] > 0
    assert r["duration"] > 3600  # at least 1 hour
    print("  ✓ inspect_only PASSED\n")


def test_full_load_bounded():
    """Full load at 100 Hz — verifies resample + finite check."""
    import numpy as np
    from easybci_lib.tools.neural_processing.io.loader import load_neural

    t0 = time.perf_counter()
    r = load_neural(SLP_PATH, target_hz=100.0)
    elapsed = time.perf_counter() - t0

    data = r["data"]
    print(f"[full_load] elapsed: {elapsed:.2f}s")
    print(f"  shape: {data.shape}  dtype: {data.dtype}")
    print(f"  channels: {r['channels']}")
    print(f"  frequency: {r['frequency']}")
    print(f"  all_finite: {bool(np.isfinite(data).all())}")
    assert data.ndim == 2
    assert data.dtype == np.float32
    assert data.shape[0] == len(r["channels"])
    assert np.isfinite(data).all(), "Non-finite values in resampled data!"
    print("  ✓ full_load PASSED\n")


def test_hypnogram():
    """Hypnogram parse — expect all-unscored in the reference case."""
    from easybci_lib.tools.neural_processing.io.psg_annotations import parse_hypnogram

    path = str(Path(SLP_PATH) / "SLPSTAG.DAT")
    stages = parse_hypnogram(path)
    unique = set(stages)
    print(f"[hypnogram]")
    print(f"  length:  {len(stages)} epochs")
    print(f"  unique:  {unique}")
    assert len(stages) > 0
    # Reference case: all epochs are 0x80 = unscored
    assert unique == {"unscored"}, f"Expected all unscored, got {unique}"
    print("  ✓ hypnogram PASSED\n")


def test_events_degradation():
    """Events parse — expect graceful degradation (no pandas-access/mdbtools)."""
    from easybci_lib.tools.neural_processing.io.psg_annotations import parse_events

    path = str(Path(SLP_PATH) / "EVENTS.MDB")
    events, fix_hint = parse_events(path)
    print(f"[events]")
    print(f"  n_events:     {len(events)}")
    print(f"  hint_present: {bool(fix_hint)}")
    if fix_hint:
        print(f"  fix_hint:     {fix_hint[:80]}...")
    assert events == []
    assert fix_hint  # non-empty hint
    print("  ✓ events degradation PASSED\n")


def test_detect_backend():
    """Backend detection works on a dir without .SLP extension check."""
    from easybci_lib.tools.neural_processing.io.loader import _detect_backend

    backend = _detect_backend(Path(SLP_PATH))
    print(f"[detect_backend]")
    print(f"  path:    {SLP_PATH}")
    print(f"  backend: {backend}")
    assert backend == "compumedics"
    print("  ✓ detect_backend PASSED\n")


if __name__ == "__main__":
    print(f"=== PSG Real Data Smoke Test ===")
    print(f"Target: {SLP_PATH}\n")

    test_detect_backend()
    test_inspect_only()
    test_hypnogram()
    test_events_degradation()

    print("--- Full load (may take 30-60s for 1.5 GB) ---")
    test_full_load_bounded()

    print("=== ALL TESTS PASSED ===")
