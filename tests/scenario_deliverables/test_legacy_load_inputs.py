"""R5: legacy _load_inputs auto-discovery only ever returned ONE file. When
inputs_routing.json is absent and no explicit nwb path is given, it walked
preprocessed/sub-*/ses-*/ and took cands[0] then broke out of all loops — so a
multi-subject work_dir silently processed a single arbitrary recording. The
auto-discovery branch must return every discovered nwb.
"""

import re
from pathlib import Path
import tempfile

_SRC = Path(
    "easybci_lib/tools/neural_processing/codegen/generator.py"
).read_text(encoding="utf-8")


def _extract_ai_ready_fn(name):
    start = _SRC.index("_AI_READY_TEMPLATE = '''")
    end = _SRC.index("\n'''", start)
    body = _SRC[start:end].replace("{{", "{").replace("}}", "}")
    m = re.search(r"\ndef %s\(.*?(?=\ndef |\Z)" % re.escape(name), body, re.S)
    assert m, "fn %s not found" % name
    ns = {}
    import sys as _sys, json as _json
    ns["sys"] = _sys
    ns["json"] = _json
    ns["Path"] = Path
    exec(compile(m.group(0), "<ai_ready:%s>" % name, "exec"), ns)
    return ns[name]


def _make_workdir_multi():
    wd = Path(tempfile.mkdtemp())
    base = wd / "preprocessed_output" / "preprocessed"
    for sub in ("sub-01", "sub-02", "sub-03"):
        ses = base / sub / "ses-1"
        ses.mkdir(parents=True)
        (ses / ("%s_ses-1_task-x_preprocessed.nwb" % sub)).write_text("x")
    return wd


def test_auto_discovery_returns_all_subjects():
    fn = _extract_ai_ready_fn("_load_inputs")
    wd = _make_workdir_multi()
    # argv[1] is NOT a real file → auto-discovery path
    inputs = fn(wd, ["build_ai_ready.py", str(wd / "does_not_exist.nwb")])
    subs = sorted(i["subject_id"] for i in inputs)
    assert subs == ["01", "02", "03"], "must discover every subject, got %r" % subs
    # each entry must carry its own stem/sub/ses, not share one
    stems = {i["stem_safe"] for i in inputs}
    assert len(stems) == 3


def test_routing_table_still_preferred():
    fn = _extract_ai_ready_fn("_load_inputs")
    wd = _make_workdir_multi()
    mp = wd / "middle_process"
    mp.mkdir(parents=True)
    import json
    (mp / "inputs_routing.json").write_text(json.dumps({
        "inputs": [{"subject_id": "99", "session_id": "1", "stem_safe": "r",
                    "file_id": "f1", "data_path": "x", "events_path": None}]
    }), encoding="utf-8")
    inputs = fn(wd, ["build_ai_ready.py", str(wd)])
    assert len(inputs) == 1 and inputs[0]["subject_id"] == "99"
