"""Bug #2: event-locked epoching must survive BIDS-named event files.

Two root causes, two layers:
  A. `_discover_events_csv` (io/deep_inspect.py) only looked for
     ``events_{stem}.csv`` — BIDS ships ``{base}_events.tsv``.
  B. the generated `build_ai_ready.py` `_load_events_csv` template used a
     comma DictReader + only read ``sample_index``/``event_code`` — BIDS TSV
     uses tab separation and ``onset``/``duration``/``trial_type``/``value``
     (onset in seconds, must be converted with fs).
"""

import textwrap

from easybci_lib.tools.neural_processing.io.deep_inspect import _discover_events_csv


# ---------------------------------------------------------------------------
# Layer A — event-file discovery
# ---------------------------------------------------------------------------

def test_discover_legacy_csv_still_found(tmp_path):
    data = tmp_path / "sub01.edf"
    data.write_bytes(b"\x00")
    (tmp_path / "events_sub01.csv").write_text("sample_index,event_code\n0,1\n")
    assert _discover_events_csv(data) == str(tmp_path / "events_sub01.csv")


def test_discover_bids_events_tsv(tmp_path):
    # data file carries the _eeg modality suffix; events drops it → _events.tsv
    data = tmp_path / "sub-01_task-motor_eeg.edf"
    data.write_bytes(b"\x00")
    ev = tmp_path / "sub-01_task-motor_events.tsv"
    ev.write_text("onset\tduration\ttrial_type\tvalue\n0.5\t0.2\tleft\t1\n")
    assert _discover_events_csv(data) == str(ev)


def test_discover_bids_events_tsv_no_modality_suffix(tmp_path):
    # data stem has no _eeg/_meg suffix → naive {stem}_events.tsv
    data = tmp_path / "recording.fif"
    data.write_bytes(b"\x00")
    ev = tmp_path / "recording_events.tsv"
    ev.write_text("onset\ttrial_type\n1.0\tgo\n")
    assert _discover_events_csv(data) == str(ev)


def test_discover_none_when_absent(tmp_path):
    data = tmp_path / "plain.edf"
    data.write_bytes(b"\x00")
    assert _discover_events_csv(data) is None


# ---------------------------------------------------------------------------
# Layer B — generated `_load_events_csv` template
# ---------------------------------------------------------------------------

def _extract_load_events_fn():
    """exec the generated build_ai_ready.py template's _load_events_csv +
    _epoch_from_events in isolation and hand back the callables."""
    from easybci_lib.tools.neural_processing.codegen import generator

    src = generator.generate_build_ai_ready_script(events_present=True)
    ns: dict = {}
    exec(compile(src, "build_ai_ready.py", "exec"), ns)  # noqa: S102
    return ns


def test_template_loads_bids_tsv(tmp_path):
    ns = _extract_load_events_fn()
    load = ns["_load_events_csv"]
    ev = tmp_path / "sub-01_task-motor_events.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\tvalue\n"
        "0.5\t0.2\tleft\t1\n"
        "1.5\t0.2\tright\t2\n"
    )
    fs = 100.0
    events = load(str(ev), fs)
    assert len(events) == 2
    # onset 0.5s * 100Hz = sample 50, not 0
    assert events[0]["start"] == 50
    assert events[1]["start"] == 150
    # trial_type is the human label; distinct labels → distinct classes
    labels = {e["label"] for e in events}
    assert len(labels) == 2


def test_template_loads_legacy_csv(tmp_path):
    ns = _extract_load_events_fn()
    load = ns["_load_events_csv"]
    ev = tmp_path / "events_rec.csv"
    ev.write_text("sample_index,event_code\n40,1\n120,2\n")
    events = load(str(ev), 100.0)
    assert len(events) == 2
    assert events[0]["start"] == 40  # sample_index used verbatim, not scaled
    assert events[1]["start"] == 120


def test_template_bids_value_fallback_for_label(tmp_path):
    ns = _extract_load_events_fn()
    load = ns["_load_events_csv"]
    ev = tmp_path / "x_events.tsv"
    # no trial_type → fall back to value for the label
    ev.write_text("onset\tvalue\n0.0\t7\n2.0\t7\n")
    events = load(str(ev), 50.0)
    assert events[0]["start"] == 0
    assert events[1]["start"] == 100
    assert all(e["label"] == 7 for e in events)


# ---------------------------------------------------------------------------
# epoch/label alignment: out-of-bounds events are dropped from epoching, so
# labels MUST be built from the kept events only. Labeling all `events` would
# silently misalign every epoch after the first dropped one — a corrupt
# training set that raises no error.
# ---------------------------------------------------------------------------


def _extract_fn(name):
    import numpy as np
    from easybci_lib.tools.neural_processing.codegen.generator import (
        generate_build_ai_ready_script,
    )

    src = generate_build_ai_ready_script(events_present=True, segment_duration=1.0)
    ns = {"np": np}
    # The template imports heavy modules at top; exec only the target fn by
    # compiling the whole module namespace with stubbed imports is fragile, so
    # exec the full source in a namespace that already has the stdlib names it
    # needs. numpy is the only hard dep of _epoch_from_events.
    exec(compile(src, "<ai_ready>", "exec"), ns)
    return ns[name], np


def test_epoch_from_events_aligns_labels_when_events_dropped():
    fn, np = _extract_fn("_epoch_from_events")
    fs = 100.0
    # 3 seconds of data -> 300 samples, 2 channels.
    data = np.zeros((2, 300), dtype=np.float32)
    # tmin=-0.2, tmax=1.0 -> window = 120 samples, pre = 20 samples.
    # event at start=250: s=230, e=350 > 300 -> DROPPED.
    events = [
        {"start": 50, "label": 1},   # s=30,  e=150  -> kept
        {"start": 150, "label": 2},  # s=130, e=250  -> kept
        {"start": 250, "label": 3},  # s=230, e=350  -> dropped (out of bounds)
    ]
    epochs, labels = fn(data, fs, events, tmin=-0.2, tmax=1.0)
    assert epochs.shape[0] == 2
    assert list(labels) == [1, 2]  # NOT [1,2,3]; aligned to kept epochs
    assert epochs.shape[0] == len(labels)


def test_epoch_from_events_all_in_bounds():
    fn, np = _extract_fn("_epoch_from_events")
    fs = 100.0
    data = np.zeros((2, 500), dtype=np.float32)
    events = [{"start": 50, "label": 7}, {"start": 200, "label": 9}]
    epochs, labels = fn(data, fs, events, tmin=-0.2, tmax=1.0)
    assert epochs.shape[0] == 2
    assert list(labels) == [7, 9]
