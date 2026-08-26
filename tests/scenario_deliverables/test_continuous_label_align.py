"""R1: L3_continuous per-sample labels must be reduced to one label per sliding
window (window-center sample), matching _sliding's geometry. The old code
loaded labels.npy verbatim (length == n_samples) as if it were per-epoch, so
epochs.shape[0] != len(labels) and every window was silently mislabelled.

Exercises the pure alignment helper rendered into the AI_ready template.
"""

import re
from pathlib import Path

import numpy as np

_SRC = Path(
    "easybci_lib/tools/neural_processing/codegen/generator.py"
).read_text(encoding="utf-8")


def _extract_ai_ready_fn(name):
    start = _SRC.index("_AI_READY_TEMPLATE = '''")
    end = _SRC.index("\n'''", start)
    body = _SRC[start:end].replace("{{", "{").replace("}}", "}")
    m = re.search(r"\ndef %s\(.*?(?=\ndef |\Z)" % re.escape(name), body, re.S)
    assert m, "function %s not found in _AI_READY_TEMPLATE" % name
    ns = {"np": np}
    # _align_continuous_labels prints to sys.stderr on mismatch
    import sys as _sys
    ns["sys"] = _sys
    exec(compile(m.group(0), "<ai_ready:%s>" % name, "exec"), ns)
    return ns[name]


def _n_windows(n_samples, fs, dur, stride):
    win = int(dur * fs)
    step = max(int(stride * fs), 1)
    starts = list(range(0, n_samples - win + 1, step))
    return len(starts) if starts else 1


def test_persample_labels_reduced_to_window_count():
    fn = _extract_ai_ready_fn("_align_continuous_labels")
    fs, dur, stride = 100.0, 2.0, 1.0        # win=200, step=100
    n_samples = 1000
    labels = np.arange(n_samples)            # per-sample ramp
    out = fn(labels, n_samples, fs, dur, stride)
    nw = _n_windows(n_samples, fs, dur, stride)
    assert out.shape[0] == nw, "labels must match window count, got %d vs %d" % (out.shape[0], nw)
    # window 0 covers [0,200), center=100 → label 100
    assert out[0] == 100


def test_already_per_window_passthrough():
    fn = _extract_ai_ready_fn("_align_continuous_labels")
    fs, dur, stride = 100.0, 2.0, 1.0
    n_samples = 1000
    nw = _n_windows(n_samples, fs, dur, stride)
    labels = np.arange(nw) * 10              # already one-per-window
    out = fn(labels, n_samples, fs, dur, stride)
    assert out.shape[0] == nw
    assert np.array_equal(out, labels)


def test_mismatch_truncates_not_silent(capsys):
    fn = _extract_ai_ready_fn("_align_continuous_labels")
    fs, dur, stride = 100.0, 2.0, 1.0
    n_samples = 1000
    nw = _n_windows(n_samples, fs, dur, stride)
    labels = np.arange(nw + 5)               # neither n_samples nor n_windows-consistent length
    out = fn(labels, n_samples, fs, dur, stride)
    assert out.shape[0] == min(nw, nw + 5) == nw
    err = capsys.readouterr().err
    assert "align" in err.lower() or "label" in err.lower(), "mismatch must warn loudly"


def test_classification_labels_pick_center_class():
    fn = _extract_ai_ready_fn("_align_continuous_labels")
    fs, dur, stride = 100.0, 1.0, 1.0        # win=100, step=100 → non-overlapping
    n_samples = 300
    labels = np.zeros(n_samples, dtype=int)
    labels[100:200] = 1                      # middle window all class 1
    labels[200:300] = 2
    out = fn(labels, n_samples, fs, dur, stride)
    # windows: [0,100) center50→0, [100,200) center150→1, [200,300) center250→2
    assert list(out) == [0, 1, 2]
