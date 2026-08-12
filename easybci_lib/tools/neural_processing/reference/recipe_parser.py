"""Parse a gold-standard preprocessing project into a machine-readable recipe.

Reads (all read-only):
  Code/config.json                              — recipe + reject_keywords
  Data/Processed/<stem>/intermediate/<stem>_preprocessing_summary.json
  Data/Processed/<stem>/bad_channels/<stem>_bad_channels.csv
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RecipeProfile:
    root: str
    stem: str
    modality: str
    source_sfreq: float
    target_sfreq: float
    low_cut: float
    high_cut: float
    notch_freqs: list[float]
    notch_q: float
    filter_order: int
    duration_sec: float
    n_signal_channels: int
    n_bad_channels: int
    reject_keywords: list[str] = field(default_factory=list)
    final_edf: str = ""

    @property
    def bad_channel_ratio(self) -> float:
        if self.n_signal_channels <= 0:
            return 0.0
        return self.n_bad_channels / self.n_signal_channels


def _load_json_bom(path: Path) -> dict[str, Any]:
    # summary.json is UTF-8 with BOM; utf-8-sig strips it.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _count_excluded(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = str(row.get("exclude", "")).strip().lower()
            if val in {"1", "true", "yes", "y", "bad", "exclude"}:
                n += 1
    return n


def parse_recipe(reference_dir: str) -> RecipeProfile:
    root = Path(reference_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"reference_dir not found: {reference_dir}")

    config_path = root / "Code" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {root}/Code/")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    pre = config.get("preprocessing", {}) or {}
    labels = config.get("labels", {}) or {}
    cases = config.get("cases", {}) or {}
    stem = str(cases.get("process_stem") or "").strip()
    if not stem:
        raise ValueError("config.cases.process_stem missing — cannot locate case")

    proc = root / "Data" / "Processed" / stem
    summary_path = proc / "intermediate" / f"{stem}_preprocessing_summary.json"
    summary = _load_json_bom(summary_path) if summary_path.is_file() else {}

    csv_path = proc / "bad_channels" / f"{stem}_bad_channels.csv"
    n_bad = _count_excluded(csv_path)

    n_signal = int(summary.get("n_signal_channels")
                   or pre.get("n_signal_channels") or 0)

    # Locate a final EDF (best-effort; name varies e.g. *_clean_excluded114.edf).
    final_dir = proc / "final"
    final_edf = ""
    if final_dir.is_dir():
        edfs = sorted(final_dir.glob("*.edf"))
        preferred = [p for p in edfs if "excl" in p.name.lower() or "clean" in p.name.lower()]
        pick = (preferred or edfs)
        if pick:
            final_edf = str(pick[0])

    def _num(*keys, default=0.0, src=None):
        src = src if src is not None else {}
        for k in keys:
            if k in src and src[k] is not None:
                return src[k]
        return default

    notch = pre.get("notch_freqs") or []
    return RecipeProfile(
        root=str(root),
        stem=stem,
        modality="seeg",  # gold standard is sEEG; ingest schema may override
        source_sfreq=float(_num("source_sfreq", src=summary) or 0.0),
        target_sfreq=float(_num("target_sfreq", src=summary) or _num("target_sfreq", src=pre) or 0.0),
        low_cut=float(_num("low_cut", src=summary) or _num("low_cut", src=pre) or 0.0),
        high_cut=float(_num("high_cut", src=summary) or _num("high_cut", src=pre) or 0.0),
        notch_freqs=[float(x) for x in notch],
        notch_q=float(pre.get("notch_q") or 0.0),
        filter_order=int(pre.get("filter_order") or 0),
        duration_sec=float(_num("duration_sec", src=summary) or _num("duration_sec", src=pre) or 0.0),
        n_signal_channels=n_signal,
        n_bad_channels=n_bad,
        reject_keywords=[str(k) for k in (labels.get("reject_keywords") or [])],
        final_edf=final_edf,
    )
