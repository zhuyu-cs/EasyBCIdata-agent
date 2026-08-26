"""BUG 3: reject_by_labels excises samples, so raw event onsets no longer map
linearly to post-reject sample indices. Events must be remapped in the SECONDS
domain (time-base invariant, survives a later resample) before epoching:

- an event inside an excised window is dropped,
- an event after excised windows shifts left by the total excised time before it.

These tests exercise the pure remapping helper rendered into the AI_ready
template, exec'd in isolation.
"""

import re
from pathlib import Path

import numpy as np

_SRC = Path(
    "easybci_lib/tools/neural_processing/codegen/generator.py"
).read_text(encoding="utf-8")


def _extract_ai_ready_fn(name):
    """Pull one def out of _AI_READY_TEMPLATE, undo the .format() brace-doubling,
    and exec it in a numpy namespace."""
    start = _SRC.index("_AI_READY_TEMPLATE = '''")
    end = _SRC.index("\n'''", start)
    tmpl = _SRC[start:end]
    # undo doubling so the body is valid python (no .format placeholders in
    # the helpers we test).
    body = tmpl.replace("{{", "{").replace("}}", "}")
    m = re.search(r"\ndef %s\(.*?(?=\ndef |\Z)" % re.escape(name), body, re.S)
    assert m, "function %s not found in _AI_READY_TEMPLATE" % name
    ns = {"np": np}
    exec(compile(m.group(0), "<ai_ready:%s>" % name, "exec"), ns)
    return ns[name]


def test_remap_drops_event_inside_excised_window():
    fn = _extract_ai_ready_fn("_remap_events_after_reject")
    events = [
        {"start": 100, "onset_s": 1.0, "label": 0},   # before excision → unchanged
        {"start": 500, "onset_s": 5.0, "label": 1},   # inside [4,6) → dropped
        {"start": 800, "onset_s": 8.0, "label": 2},   # after → shift left by 2 s
    ]
    fs = 100.0
    rejected = [[4.0, 6.0]]  # 2 seconds excised
    out = fn(events, rejected, fs)
    labels = [e["label"] for e in out]
    assert labels == [0, 2], "event inside excised window must be dropped"
    starts = {e["label"]: e["start"] for e in out}
    assert starts[0] == 100          # 1.0 s unchanged
    assert starts[2] == int(round((8.0 - 2.0) * fs))  # 8-2 = 6 s → 600 samples


def test_remap_noop_when_no_reject():
    fn = _extract_ai_ready_fn("_remap_events_after_reject")
    events = [{"start": 300, "onset_s": 3.0, "label": 7}]
    out = fn(events, [], 100.0)
    assert len(out) == 1
    assert out[0]["start"] == 300


def test_remap_multiple_windows_accumulate():
    fn = _extract_ai_ready_fn("_remap_events_after_reject")
    events = [{"start": 1000, "onset_s": 10.0, "label": 5}]
    fs = 100.0
    rejected = [[1.0, 2.0], [4.0, 6.0]]  # 1 + 2 = 3 s excised before t=10
    out = fn(events, rejected, fs)
    assert len(out) == 1
    assert out[0]["start"] == int(round((10.0 - 3.0) * fs))  # 700 samples


def test_remap_event_uses_onset_s_not_stale_start():
    """When onset_s is present it is authoritative (start was computed against
    the ORIGINAL sample axis and is stale after excision)."""
    fn = _extract_ai_ready_fn("_remap_events_after_reject")
    events = [{"start": 999999, "onset_s": 8.0, "label": 1}]
    fs = 100.0
    out = fn(events, [[4.0, 6.0]], fs)
    assert out[0]["start"] == int(round(6.0 * fs))


def _exec_pipeline_reject():
    """Exec op_reject_by_labels + its deps out of the rendered pipeline.py."""
    from easybci_lib.tools.neural_processing.codegen import generator as g
    src = g.generate_pipeline_script(
        steps=["reject_by_labels:seizure"],
        data_info={"n_channels": 2, "sampling_rate": 100,
                   "channel_names": ["a", "b"]},
        modality="eeg", analysis_goal="classification",
    )
    ns = {}
    exec(compile(src, "<pipeline>", "exec"), ns)
    return ns


def test_op_reject_records_intervals_in_seconds():
    ns = _exec_pipeline_reject()
    op = ns["op_reject_by_labels"]
    fs = 100.0
    n = 1000
    d = {
        "data": np.random.randn(2, n).astype("float32"),
        "frequency": fs,
        "meta": {
            "annotations": {
                "onset": [5.0],        # seizure at 5 s, dur 1 s, ±1 s pad
                "duration": [1.0],
                "description": ["seizure onset"],
            }
        },
    }
    out = op(d, "seizure")
    iv = out["meta"].get("rejected_intervals")
    assert iv, "rejected_intervals must be recorded"
    (a, b), = iv
    # window = [onset-pad, onset+dur+pad] = [4, 7] s
    assert abs(a - 4.0) < 0.05
    assert abs(b - 7.0) < 0.05
    # sample axis shrank by ~3 s * 100 = 300 samples
    assert out["data"].shape[1] == n - out["meta"]["rejected_samples"]
    assert abs(out["meta"]["rejected_samples"] - 300) <= 2


def test_op_reject_noop_has_no_intervals():
    ns = _exec_pipeline_reject()
    op = ns["op_reject_by_labels"]
    d = {
        "data": np.random.randn(2, 500).astype("float32"),
        "frequency": 100.0,
        "meta": {"annotations": {"onset": [1.0], "duration": [0.1],
                                 "description": ["baseline rest block"]}},  # no keyword hit
    }
    out = op(d, "seizure")
    assert out["data"].shape[1] == 500
    assert not out["meta"].get("rejected_intervals")
