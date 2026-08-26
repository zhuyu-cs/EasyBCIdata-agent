"""R2: split ratios must be normalized before use. _sequential_split /
_temporal_split use np.cumsum(ratios.values()) directly as thresholds; if the
ratios don't sum to 1.0 the tail split is under/over-filled silently. A single
normalization at the split_data entry point must fix all methods.
"""

import numpy as np

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_split_code,
)


def _build(method, ratios):
    src = generate_split_code({"method": method, "ratios": ratios})
    ns = {"np": np}
    exec(compile(src, "<split>", "exec"), ns)
    return ns


def test_sequential_unnormalized_ratios_still_partition():
    ns = _build("sequential", {"train": 7, "val": 1.5, "test": 1.5})  # sums to 10
    res = ns["split_data"](1000, method="sequential",
                           ratios={"train": 7, "val": 1.5, "test": 1.5})
    counts = {k: int(np.sum(res == k)) for k in ("train", "val", "test")}
    assert sum(counts.values()) == 1000, "every item must be assigned"
    # proportions must follow the normalized ratios (0.7/0.15/0.15), not raw.
    assert abs(counts["train"] / 1000 - 0.7) < 0.03
    assert abs(counts["test"] / 1000 - 0.15) < 0.03
    assert counts["test"] > 0, "tail split must not be starved"


def test_temporal_unnormalized_ratios_still_partition():
    ns = _build("temporal", {"train": 70, "test": 30})  # sums to 100
    res = ns["split_data"](1000, method="temporal",
                           ratios={"train": 70, "test": 30}, temporal_gap=0)
    counts = {k: int(np.sum(res == k)) for k in ("train", "test")}
    assert counts["train"] + counts["test"] == 1000
    assert abs(counts["train"] / 1000 - 0.7) < 0.03


def test_normalized_ratios_unchanged():
    ns = _build("sequential", {"train": 0.7, "val": 0.15, "test": 0.15})
    res = ns["split_data"](1000, method="sequential",
                           ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    counts = {k: int(np.sum(res == k)) for k in ("train", "val", "test")}
    assert abs(counts["train"] / 1000 - 0.7) < 0.03
    assert sum(counts.values()) == 1000


def test_zero_sum_ratios_safe():
    ns = _build("sequential", {"train": 0, "test": 0})
    # must not divide-by-zero / crash; falls back to equal or all-last.
    res = ns["split_data"](10, method="sequential", ratios={"train": 0, "test": 0})
    assert len(res) == 10
