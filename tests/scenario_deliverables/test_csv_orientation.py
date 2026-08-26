"""R4: _load_csv decided (n_channels, n_samples) orientation purely by
`shape[0] > shape[1]`, silently transposing any recording where n_channels >
n_samples (short / high-density). When a header row names the channels, the
header column-count is authoritative and must anchor the orientation.
"""

import numpy as np
import tempfile
from pathlib import Path

from easybci_lib.tools.neural_processing.io.loader import _load_csv


def _write_csv(rows_of_cols, header=None):
    p = Path(tempfile.mkdtemp()) / "d.csv"
    lines = []
    if header:
        lines.append(",".join(header))
    for row in rows_of_cols:
        lines.append(",".join(str(x) for x in row))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_header_anchors_orientation_high_density():
    # 6 channels (columns) named in header, only 3 samples (rows). This is
    # n_channels > n_samples — the shape heuristic would wrongly transpose it.
    header = ["C1", "C2", "C3", "C4", "C5", "C6"]
    rows = [[i + 0.1 * c for c in range(6)] for i in range(3)]  # 3 rows x 6 cols
    path = _write_csv(rows, header=header)
    res = _load_csv(path)
    assert res["channels"] == header, "header must anchor channel axis"
    assert res["data"].shape == (6, 3), "data must be (n_channels, n_samples)"


def test_header_normal_orientation_still_works():
    header = ["C1", "C2"]
    rows = [[i, i + 100] for i in range(50)]  # 50 samples x 2 channels
    path = _write_csv(rows, header=header)
    res = _load_csv(path)
    assert res["channels"] == header
    assert res["data"].shape == (2, 50)


def test_no_header_falls_back_to_heuristic():
    # No header: (100 samples x 2 channels) — tall matrix stays as (2, 100).
    rows = [[i, i + 100] for i in range(100)]
    path = _write_csv(rows, header=None)
    res = _load_csv(path)
    assert res["data"].shape == (2, 100)
