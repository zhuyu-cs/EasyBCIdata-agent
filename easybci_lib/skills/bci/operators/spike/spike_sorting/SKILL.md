---
name: spike_sorting
description: "Spike sorting workflow — detection, feature extraction, clustering, and quality assessment of single-unit activity"
layer: L3
group: spike
metadata:
  tags: [operator, spike, sorting, kilosort, clustering, neuropixels, single_unit]
  modalities: [spike]
  step_string: "spike_sort"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic]
  analysis_goal_forbidden: [online_inference, source_localization, phase_amplitude_coupling]
---
# Spike Sorting

## Function

Spike sorting is the process of assigning each detected extracellular action potential (spike) to its source neuron. Unlike other preprocessing operators that transform continuous signals, spike sorting is a complete multi-stage pipeline that takes raw high-pass filtered extracellular recordings and produces a set of classified single units with spike times and quality metrics.

## Pipeline Stages

```
Raw AP-band data (30kHz, int16)
    │
    ▼
[1] DETECTION — Threshold crossings to find candidate spike events
    │
    ▼
[2] EXTRACTION — Cut waveform snippets around each detection
    │
    ▼
[3] FEATURE EXTRACTION — PCA or template-based dimensionality reduction
    │
    ▼
[4] CLUSTERING — Assign spikes to putative units (Kilosort, MountainSort, etc.)
    │
    ▼
[5] QUALITY METRICS — ISI violations, SNR, contamination, presence ratio
    │
    ▼
[6] CURATION — Label units as 'good' (single-unit) or 'mua' (multi-unit)
```

## Input Requirements

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Sampling rate | 30,000 Hz | Must be >= 20kHz for spike waveform resolution |
| Data type | int16 or float32 | Raw extracellular voltage |
| High-pass | >= 300 Hz | AP-band filtered (remove LFP) |
| Channels | 32-384 | Multi-channel probe (Neuropixels, Utah array, etc.) |
| Duration | >= 10 min | Longer is better for stable clustering |

## Available Methods

| Method | Strengths | When to Use |
|--------|-----------|-------------|
| Kilosort (2/3/4) | Fast, GPU-accelerated, template-based | Neuropixels, high channel count, default choice |
| MountainSort5 | Accurate, no GPU needed, well-validated | Neuropixels / Utah arrays when GPU or Kilosort unavailable; handles 384-ch probes |
| SpykingCircus2 | Optimized for dense probes, parallelized | High-density Neuropixels, good fallback when Kilosort unavailable |
| SpyKING CIRCUS | Good for tetrodes, parallelized | Tetrode recordings (legacy; prefer SpykingCircus2) |
| IronClust | MATLAB-based, Kilosort alternative | Legacy workflows |
| Tridesclous | Python-native, real-time capable | Online sorting, smaller datasets |

## Quality Metrics (Post-Sorting)

These metrics evaluate each sorted unit:

| Metric | Good Unit | MUA | Noise | Description |
|--------|-----------|-----|-------|-------------|
| ISI violations (%) | < 0.5% | 0.5-2% | > 2% | Spikes within 1ms refractory period |
| SNR | > 5 | 2-5 | < 2 | Peak amplitude / noise RMS |
| Firing rate | 0.1-100 Hz | variable | > 300 Hz suspicious | Mean spikes per second |
| Presence ratio | > 0.9 | 0.5-0.9 | < 0.5 | Fraction of recording with activity |
| Amplitude cutoff | < 0.1 | 0.1-0.3 | > 0.3 | Fraction of spikes below detection |
| Contamination % | < 10% | 10-50% | > 50% | Estimated fraction of non-unit spikes |

## Classification Criteria

**Single unit (good):**
- ISI violations < 0.5%
- SNR > 5
- Presence ratio > 0.9
- Stable waveform shape throughout recording
- Clear refractory period gap in ISI histogram

**Multi-unit activity (mua):**
- ISI violations 0.5-2%
- May include spikes from multiple nearby neurons
- Still spatially localized, useful for population analyses

**Noise (reject):**
- ISI violations > 2%
- Very low SNR
- No consistent waveform
- Likely electrical noise or movement artifacts

## Waveform Features

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| Peak-to-valley | Time from negative peak to positive peak | 0.2-0.8 ms |
| Half-width | Duration at half-maximum amplitude | 0.1-0.5 ms |
| Repolarization slope | Slope of return to baseline | varies |
| Spread | Spatial extent across channels | 20-100 um |
| Peak channel | Channel with largest amplitude | — |

**Cell type classification from waveform:**
- Narrow waveform (half-width < 0.25 ms): putative fast-spiking interneuron
- Broad waveform (half-width > 0.35 ms): putative pyramidal/excitatory neuron

## Pre-flight Checklist

Before proceeding with spike sorting:

1. **Determine data state**: Is this raw extracellular recording (no `units/` group in NWB) or already sorted (has `units/spike_times`)? Use `inspect_neural` or direct h5py inspection.
2. **Check available sorters**: Run `python3 -c "import spikeinterface.sorters as ss; print(ss.installed_sorters())"` — Kilosort may NOT be installed even if it's the default recommendation.
3. **Check for LFP band**: NWB files from Neuropixels typically have BOTH AP band (`ElectricalSeriesAP*`, 30kHz) AND LFP band (`ElectricalSeriesLF*`, 2.5kHz). Only the AP band is used for spike sorting.
4. **Estimate memory**: 384ch × 30kHz × int16 ≈ 2.16 GB/min. Use memory-mapped I/O via SpikeInterface.

## EasyBCI Operator vs. Full Sorting

**CRITICAL**: EasyBCI's built-in `spike_sort` operator (used via `preprocess_neural` or `save_processed`) is a **POST-SORTING** tool only — it expects `spike_times` to already exist in the NWB file. Running it on raw data will fail with `ValueError: No spike_times found`.

| Tool | Scope | When to Use |
|------|-------|-------------|
| `preprocess_neural(data, steps=['spike_sort'])` | Post-sorting: bin, curate, export | Data already has sorted units |
| `bin_spikes(data_path)` | Spike train binning | Post-sorting, after spike_times exist |
| `save_processed(data, preprocess_steps=['spike_sort'])` | Full post-sorting pipeline | Data already sorted |
| **SpikeInterface via terminal/Python** | Full sorting from raw | Raw NWB with AP-band data, no existing units |

**For raw Neuropixels data → sorted units, use SpikeInterface directly in a terminal Python script.** See the SpikeInterface Integration section below for the full workflow.

## SpikeInterface Integration

SpikeInterface provides a unified Python API for all spike sorters. Below is the **recommended end-to-end script** for raw NWB → sorted + curated units.

### Complete End-to-End Script (MountainSort5)

```python
"""Full spike sorting pipeline: raw NWB → curated units + quality metrics."""
import spikeinterface.full as si
from pathlib import Path

# === CONFIG ===
NWB_PATH = "/path/to/raw_without_gt.nwb"
OUTPUT_DIR = "/path/to/spike_sorting_output"
SORTER = "mountainsort5"  # or "spykingcircus2", "kilosort3" (if GPU available)

# === 1. LOAD RAW DATA ===
# Neuropixels NWB: ElectricalSeriesAP* = AP band (30kHz), ElectricalSeriesLF* = LFP (2.5kHz)
# SpikeInterface auto-selects the AP band by default via read_nwb_recording
recording = si.read_nwb_recording(NWB_PATH, electrical_series_name="ElectricalSeriesAPImec0")

print(f"Loaded: {recording.get_num_channels()} ch, "
      f"{recording.get_sampling_frequency()} Hz, "
      f"{recording.get_total_duration():.0f}s")

# === 2. PREPROCESSING ===
# Bandpass 300-6000 Hz — isolate spike band, remove LFP and high-freq noise
recording_bp = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
# Common median reference — remove shared noise across channels
recording_cmr = si.common_reference(recording_bp, reference="global", operator="median")

# === 3. RUN SORTER ===
sorting = si.run_sorter(
    SORTER,
    recording_cmr,
    output_folder=f"{OUTPUT_DIR}/sorter_output",
    verbose=True,
    remove_existing_folder=True,
)
print(f"Sorted: {len(sorting.get_unit_ids())} units found")

# === 4. POST-PROCESSING ===
# Extract waveforms
we = si.extract_waveforms(
    recording_cmr, sorting,
    folder=f"{OUTPUT_DIR}/waveforms",
    ms_before=1.0, ms_after=2.0,
    max_spikes_per_unit=500,
    overwrite=True,
)

# Compute quality metrics
metrics = si.compute_quality_metrics(
    we,
    metric_names=["snr", "isi_violation", "presence_ratio",
                  "amplitude_cutoff", "firing_rate", "num_spikes"],
)
print(f"Quality metrics for {len(metrics)} units:")
print(metrics[["snr", "isi_violations_ratio", "firing_rate", "presence_ratio"]])

# === 5. AUTO-CURATION ===
sorting_curated = si.auto_curation(
    sorting, metrics,
    isi_violations_ratio_threshold=0.5,
    snr_threshold=5.0,
    presence_ratio_threshold=0.9,
    firing_rate_range=(0.1, 300),
)

labels = sorting_curated.get_property("curation_label")
from collections import Counter
print(f"Curation: {Counter(labels)}")

# === 6. SAVE ===
si.NwbSortingExtractor.write_sorting(sorting_curated, f"{OUTPUT_DIR}/sorted_curated.nwb")
metrics.to_csv(f"{OUTPUT_DIR}/quality_metrics.csv")

# === 7. SUMMARY ===
n_good = sum(1 for l in labels if l == "good")
n_mua = sum(1 for l in labels if l == "mua")
n_noise = sum(1 for l in labels if l == "noise")
print(f"\n=== FINAL RESULTS ===")
print(f"Good units (single-unit): {n_good}")
print(f"MUA:                     {n_mua}")
print(f"Noise (rejected):        {n_noise}")
print(f"Total detected:          {len(labels)}")
print(f"Output: {OUTPUT_DIR}/")
```

### Kilosort3 Variant (GPU required)

```python
SORTER = "kilosort3"
recording_cmr = si.common_reference(recording, reference="global", operator="median")
sorting = si.run_sorter(SORTER, recording_cmr, output_folder=f"{OUTPUT_DIR}/ks3_output")
# ... post-processing identical
```

### Quick Test (first 60s)

```python
import spikeinterface.full as si
rec = si.read_nwb_recording("data.nwb")
rec = rec.frame_slice(0, int(60 * rec.get_sampling_frequency()))
rec = si.bandpass_filter(rec, 300, 6000)
rec = si.common_reference(rec, reference="global", operator="median")
sorting = si.run_sorter("mountainsort5", rec, output_folder="test_sort")
print(f"Units: {len(sorting.get_unit_ids())}")
```

## Common Issues

- **No spike_times in NWB**: When `inspect_neural` or `preprocess_neural` fails with "No spike_times found", the data is raw extracellular recording — use SpikeInterface directly, not EasyBCI's `spike_sort` operator. See Pre-flight Checklist above.
- **Kilosort not installed**: If `ss.installed_sorters()` doesn't include kilosort*, fall back to MountainSort5 or SpykingCircus2. Both handle 384-ch Neuropixels well.
- **Drift**: Long recordings on Neuropixels show electrode drift. Use Kilosort's drift correction or DREDge.
- **Overmerging**: Two neurons with similar waveforms merged into one unit. Check ISI histogram for bimodality.
- **Oversplitting**: One neuron split into multiple units. Check cross-correlogram for same-neuron signature.
- **Low yield**: Few good units. May indicate poor tissue contact, too aggressive thresholding, or wrong probe geometry.
- **Memory**: Full Neuropixels recordings (384ch x 30kHz x hours) are hundreds of GB. Process in blocks or use memory-mapped I/O.

## Evaluation Metrics (for benchmarking)

When comparing Agent's sorting to expert GT:

| Metric | Description |
|--------|-------------|
| Unit yield | N_detected / N_gt units |
| Classification accuracy | Agreement on good/mua/noise labels |
| Spike assignment accuracy | % of GT spikes correctly assigned to matching unit |
| False positive rate | Spikes assigned to non-existent units |
| Waveform similarity | Cosine similarity of mean waveforms |
| Timing precision | Jitter between matched spike times (should be < 0.5 ms) |

## Reference Code

### End-to-End Sorting Script

A complete, adaptable script for sorting raw Neuropixels NWB data is available at:

    scripts/sort_raw_neuropixels.py

Usage: `python scripts/sort_raw_neuropixels.py <input.nwb> <output_dir>`

This is the canonical starting point for agent-driven spike sorting. It covers:
sorter auto-detection, NWB AP-band loading, preprocessing, sorting, waveform extraction,
quality metrics, auto-curation, and result export.

### Load Spike Data from NWB (h5py)

```python
import h5py
import numpy as np

with h5py.File("recording.nwb", "r") as f:
    spike_times = f["units/spike_times"][:]
    spike_index = f["units/spike_times_index"][:]

    # Extract per-unit spike trains
    units = {}
    prev = 0
    for i, end in enumerate(spike_index):
        units[i] = spike_times[prev:int(end)]
        prev = int(end)

# units[0] = array of spike times for unit 0 (in seconds)
```

### Spike Train Binning

```python
import numpy as np

bin_size = 0.025  # 25 ms bins
duration = spike_times[-1]
n_bins = int(np.ceil(duration / bin_size))
n_units = len(units)
binned = np.zeros((n_units, n_bins), dtype=np.float32)

for i, times in units.items():
    bin_indices = np.floor(times / bin_size).astype(int)
    bin_indices = bin_indices[bin_indices < n_bins]
    np.add.at(binned[i], bin_indices, 1.0)

# Convert to firing rate (spikes/sec)
firing_rates = binned / bin_size
```

### Quality Metrics Computation

```python
import numpy as np

def compute_unit_metrics(spike_times_per_unit, duration):
    """Compute quality metrics for sorted units."""
    metrics = []
    for unit_id, times in spike_times_per_unit.items():
        isi = np.diff(times)

        # ISI violations: spikes within 1ms refractory period
        n_violations = np.sum(isi < 0.001)
        isi_violation_ratio = n_violations / max(len(isi), 1)

        # Firing rate
        firing_rate = len(times) / duration

        # Presence ratio: fraction of recording with activity
        n_bins = 100
        bin_edges = np.linspace(0, duration, n_bins + 1)
        presence = np.sum(np.histogram(times, bins=bin_edges)[0] > 0) / n_bins

        metrics.append({
            'unit_id': unit_id,
            'n_spikes': len(times),
            'firing_rate_hz': firing_rate,
            'isi_violation_ratio': isi_violation_ratio,
            'presence_ratio': presence,
        })
    return metrics
```

### Key API

- **h5py**: `f["units/spike_times"][:]`, `f["units/spike_times_index"][:]`
- **Binning**: `np.add.at(binned[i], bin_indices, 1.0)` — accumulate spike counts
- **SpikeInterface**: `si.run_sorter()`, `si.compute_quality_metrics()`, `si.extract_waveforms()`

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| **Drift uncorrected (long Neuropixels session)** | Sorted units' mean waveform amplitude monotonically decreases / shifts across recording. | Plot waveform amplitude over time per unit; slope > 20% / 30 min → enable Kilosort drift correction (KS 2.5+) or DREDge. |
| **Cluster collapse** | All channels' spikes merge into a single 'unit'. | `n_units == 1` after sorting on 384-ch recording → upstream CAR was too aggressive or `clip_outliers` removed every distinct waveform; lower CAR aggressiveness. |
| **Over-splitting (one neuron → multiple clusters)** | High cross-correlogram peak between two clusters at lag 0 + bimodal ISI in joined train. | `si.curation.find_redundant_units` (CCG correlation > 0.7 within ±2 ms); merge candidates. |
| **Over-merging (multiple neurons → one cluster)** | Bimodal ISI histogram on a single unit; mean waveform shows two peaks. | ISI < refractory ratio > 2% + bimodal waveform PCA → split with `si.curation.split_unit_by_peak`. |
| **NaN warps / kilosort GPU instability** | KiloSort exits early with cryptic CUDA error; `Templates.npy` contains NaN. | Detect post-sort by `np.isnan(templates).any()` → recovery is to drop the failed batch or fall back to `mountainsort5` / `spykingcircus2`. |
| **Sparse-firing unit rejected as noise** | Real units with < 0.1 Hz firing rate are auto-curated to "noise". | `presence_ratio < 0.5` defaults are too strict for sparse cortical neurons — relax to 0.3 when investigating sparse populations. |
| **Bandpass too narrow** | Waveform troughs look square / clipped. | Upstream `bandpass:600,3000` is too narrow; use `bandpass:300,6000` for Neuropixels. |
| **Operating on already-sorted NWB** | `units/spike_times` already present → re-sorting unnecessarily. | Pre-check `inspect_neural` for `units` table; if present, skip to QC & binning. |

Detection helper:

```python
import numpy as np


def diagnose_sorting(metrics_df, templates):
    """Return (status, details). `metrics_df` is the SpikeInterface QC dataframe."""
    if np.isnan(templates).any():
        return "nan_templates", {"hint": "sorter run failed mid-batch; switch sorters"}
    n_good = (metrics_df["snr"] > 5).sum() if "snr" in metrics_df else 0
    if n_good == 0:
        return "no_good_units", {"hint": "loosen presence_ratio threshold or check CAR"}
    return "ok", {"n_good": int(n_good)}
```

## Reference Implementation (EasyBCI-Adapted)

The framework operator `operator_spike_sort` is a **post-sorting** thin
wrapper — it expects `units/spike_times` to already exist in the NWB,
loaded into `data_dict["meta"]["units"]`. Below is the in-framework
adapter that binds the in-process post-sorting curation; full from-raw
sorting is documented in the SpikeInterface Integration section above.

```python
from typing import Any, Dict
import time

import numpy as np

from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_spike_sort(
    data_dict: Dict[str, Any],
    *,
    isi_violation_threshold: float = 0.5,
    snr_threshold: float = 5.0,
    presence_threshold: float = 0.9,
    firing_rate_range: tuple = (0.1, 300.0),
) -> Dict[str, Any]:
    """EasyBCI-adapted post-sorting curation step.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; requires ``meta["units"]`` dict {unit_id: spike_times_s}
        loaded from an already-sorted NWB.
    isi_violation_threshold : float
        Maximum ISI < 1 ms violation ratio for 'good' label (default 0.5).
    snr_threshold : float
        Minimum SNR for 'good' label (default 5.0).
    presence_threshold : float
        Minimum presence ratio for 'good' label (default 0.9).
    firing_rate_range : (float, float)
        Acceptable firing-rate window (Hz) for 'good' label.

    Returns
    -------
    dict
        OperatorIO with ``meta["unit_labels"]`` ∈ {"good", "mua", "noise"} per unit.

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` if ``meta["units"]`` missing — raw sorting must
        be done outside EasyBCI (see SpikeInterface Integration above).

    Modality coverage
    -----------------
    spike (NWB-sorted): yes. Raw extracellular (no units table): forbidden
    — raises with recoverable=False instructing to run SpikeInterface first.

    References
    ----------
    Pachitariu 2024 (Kilosort4); Chung 2017 (MountainSort); Yger 2018 (SpyKING CIRCUS).
    """
    units = (data_dict.get("meta") or {}).get("units")
    if not units:
        raise EasyBCIOperatorError(
            operator="spike_sort",
            reason="meta['units'] missing — sorting must run outside EasyBCI via SpikeInterface",
            recoverable=False,
        )
    t0 = time.monotonic()
    duration = float(data_dict.get("duration") or 0.0)
    labels: Dict[Any, str] = {}
    for uid, times in units.items():
        times = np.asarray(times)
        n = len(times)
        if n < 2 or duration <= 0:
            labels[uid] = "noise"
            continue
        isi = np.diff(times)
        isi_violation = float(np.mean(isi < 0.001))
        firing_rate = n / duration
        n_bins = 100
        edges = np.linspace(0, duration, n_bins + 1)
        presence = float(np.sum(np.histogram(times, bins=edges)[0] > 0) / n_bins)
        if (
            isi_violation < isi_violation_threshold / 100.0
            and firing_rate_range[0] <= firing_rate <= firing_rate_range[1]
            and presence >= presence_threshold
        ):
            labels[uid] = "good"
        elif isi_violation < 0.02 and firing_rate < 300:
            labels[uid] = "mua"
        else:
            labels[uid] = "noise"

    elapsed = time.monotonic() - t0
    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    new_meta = dict(out.get("meta") or {})
    new_meta["unit_labels"] = labels
    new_meta["spike_sort"] = {
        "isi_violation_threshold": isi_violation_threshold,
        "snr_threshold": snr_threshold,
        "presence_threshold": presence_threshold,
        "firing_rate_range": list(firing_rate_range),
    }
    out["meta"] = new_meta
    record_step_elapsed(
        "spike_sort", elapsed,
        (data_dict.get("meta") or {}).get("step_cache_key"),
    )
    return out
```

## References

1. Pachitariu, M., Sridhar, S., & Stringer, C. (2024). *Spike sorting with
   Kilosort4*. Nature Methods 21(5): 914–921.
   doi:10.1038/s41592-024-02232-7 — the modern Kilosort baseline; default
   sorter when GPU is available.
2. Chung, J. E. et al. (2017). *A Fully Automated Approach to Spike Sorting*.
   Neuron 95(6): 1381–1394.e6. doi:10.1016/j.neuron.2017.08.030 — MountainSort
   (and MountainSort5) family; recommended when GPU is absent.
3. Yger, P. et al. (2018). *A spike sorting toolbox for up to thousands of
   electrodes validated with ground truth recordings in vitro and in vivo*.
   eLife 7: e34518. doi:10.7554/eLife.34518 — SpyKING CIRCUS / SpykingCircus2;
   dense-probe alternative.
4. Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). *Unsupervised spike
   detection and sorting with wavelets and superparamagnetic clustering*.
   Neural Computation 16(8): 1661–1687. doi:10.1162/089976604774201631 —
   Wave_clus; superparamagnetic clustering reference.
5. Buccino, A. P. et al. (2020). *SpikeInterface, a unified framework for
   spike sorting*. eLife 9: e61834. doi:10.7554/eLife.61834 — the SpikeInterface
   abstraction this skill builds against.
6. Steinmetz, N. A. et al. (2018). *Challenges and opportunities for
   large-scale electrophysiology with Neuropixels probes*. Current Opinion
   in Neurobiology 50: 92–100. doi:10.1016/j.conb.2018.01.009 — drift,
   yield, and best-practice recommendations.

## Related Skills

- **`threshold_spike`** — fast threshold-only extraction; use instead when sorting is too slow (online BCI, real-time feedback). See that skill's `When to Use / NOT to Use` section for the 30-second decision rule.
- **`waveform_snippet`** — extract waveforms around threshold detections (QC / sorting bootstrap).
- **`mua_binning`** — bin MUA spike train; complementary to `bin_spikes` (sorted unit-level).
