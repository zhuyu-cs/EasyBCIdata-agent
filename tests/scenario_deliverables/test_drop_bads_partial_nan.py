"""R3: op_drop_bads keeps channels with <=50% NaN (intentional threshold), but
those survivors carried raw NaN forward into later MNE-based ops. Surviving
channels must be NaN-cleaned in place (interpolated) so no NaN propagates, and
the cleanup must be recorded in meta — without lowering the 50% drop threshold.
"""

import re
from pathlib import Path

import numpy as np

_SRC = Path(
    "easybci_lib/tools/neural_processing/codegen/generator.py"
).read_text(encoding="utf-8")


def _extract_op(name):
    start = _SRC.index("_PIPELINE_SCRIPT_TEMPLATE = '''")
    end = _SRC.index("\n'''", start)
    body = _SRC[start:end].replace("{{", "{").replace("}}", "}")
    m = re.search(r"\ndef %s\(.*?(?=\ndef |\Z)" % re.escape(name), body, re.S)
    assert m, "op %s not found" % name
    ns = {"np": np}
    import re as _re
    ns["_re"] = _re
    exec(compile(m.group(0), "<op:%s>" % name, "exec"), ns)
    return ns[name]


def test_partial_nan_channel_survives_but_cleaned():
    op = _extract_op("op_drop_bads")
    rng = np.random.RandomState(0)
    data = rng.randn(4, 1000).astype(float) * 10.0  # comparable variance
    # channel 1: 40% NaN → below the 50% drop threshold, must SURVIVE
    idx = rng.choice(1000, 400, replace=False)
    data[1, idx] = np.nan
    d = {"data": data, "channels": ["c0", "c1", "c2", "c3"], "meta": {}}
    out = op(d, "auto")
    assert "c1" in out["channels"], "40% NaN channel must not be dropped"
    out_arr = np.asarray(out["data"])
    assert not np.isnan(out_arr).any(), "surviving channels must be NaN-free"


def test_full_nan_channel_still_dropped():
    op = _extract_op("op_drop_bads")
    rng = np.random.RandomState(1)
    data = rng.randn(3, 500).astype(float) * 10.0
    data[2, :] = np.nan  # 100% NaN → dropped
    d = {"data": data, "channels": ["a", "b", "c"], "meta": {}}
    out = op(d, "auto")
    assert "c" not in out["channels"]
    assert not np.isnan(np.asarray(out["data"])).any()


def test_clean_channels_untouched():
    op = _extract_op("op_drop_bads")
    rng = np.random.RandomState(2)
    data = rng.randn(3, 500).astype(float) * 10.0
    d = {"data": data.copy(), "channels": ["a", "b", "c"], "meta": {}}
    out = op(d, "auto")
    # no NaN anywhere → all kept, values preserved (float32 cast tolerance)
    assert out["channels"] == ["a", "b", "c"]
    assert np.allclose(np.asarray(out["data"]), data, atol=1e-4)
