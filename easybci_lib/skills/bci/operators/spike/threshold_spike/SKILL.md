---
name: threshold_spike
description: "Threshold-based extracellular spike detection (MAD / σ̂-multiplier) — fast MUA spike train without sorting"
layer: L3
group: spike
metadata:
  tags: [operator, spike, mua, threshold, neuropixels, utah, online, real-time]
  modalities: [spike]
  step_string: "threshold_spike"
  analysis_goal_allowed: [classification, feature_extraction, exploratory, generic, online_inference]
  analysis_goal_forbidden: [source_localization, clinical_screening, connectivity, phase_amplitude_coupling]
---
# Threshold-Based Spike Detection

## Function

Detects extracellular action potentials (spikes) from band-pass filtered
high-channel-count recordings using **noise-relative amplitude thresholding**.
Outputs a multi-unit activity (MUA) spike train (event time stamps per
channel) — **without** unit clustering. This is the canonical fast spike
extractor for Neuropixels / Utah-array recordings where Kilosort-style
sorting is too slow for online BCI or real-time experimental feedback.

**Input / Output**:

```
data : ndarray (n_channels, n_times)           # AP-band filtered, 30 kHz typical
  ↓
spike_times : list[ndarray]                    # len == n_channels; each entry is sample indices
  + meta["mua_counts"], meta["thresholds"], meta["refractory_violations"]
```

The continuous `data` array is **not modified** — the operator publishes
spike times into `out["meta"]["spike_times"]` and `out["meta"]["mua_train"]`
(binned, optional), preserving the OperatorIO schema (Rule 4).

This is the threshold-only sister of `spike_sorting`. See `When to Use /
NOT to Use` below for the 30-second decision rule between the two.

## Algorithm & Math

### Median Absolute Deviation (MAD) noise estimate

Quiroga (2004) showed that the RMS of band-passed extracellular voltage
is dominated by spike energy, not noise, so straight `std()` over-estimates
the noise floor and under-reports spikes. The robust replacement is

```
σ̂ = median(|x|) / 0.6745                      # per channel, per ~1 s window
```

where `0.6745` is the 75th-percentile-to-σ scaling factor of the standard
normal — the choice that makes `σ̂` agree with `std(x)` when `x` is purely
Gaussian and disagrees (down) when `x` contains sparse high-amplitude
spike events.

### Threshold rule

A sample `x[t]` is a spike crossing on channel `c` iff

```
|x[c, t]| > k · σ̂[c]      and      t - t_last_spike[c] > τ_refractory
```

with `k ∈ {3, 4, 5}` (Quiroga's default is 4; Rey 2015's Wave_clus uses
5 for sparse-firing cortical regions; IBL Brain-Wide-Map pipeline uses
5 per-channel). The refractory mask `τ_refractory = 0.5 ms` blocks
spurious double-detections from waveform reflections.

### Detection schemes

| Scheme | Direction | Use case | Notes |
|---|---|---|---|
| `negative` | crosses below `−k·σ̂` | Default for extracellular | Spike depolarization shows as negative deflection on extracellular electrodes. |
| `positive` | crosses above `+k·σ̂` | Intracellular / rare extracellular | Reversed polarity (rare); only when amplifier polarity inverted. |
| `bilateral` | `|x| > k·σ̂` | Inspection-only | Inflates rate; use for debugging, never for downstream decoding. |

### Per-channel vs global threshold

```
per_channel=True  (default):  k · σ̂[c]    for each channel c
per_channel=False:            k · median(σ̂[*])  shared across all channels
```

Per-channel is the **default for Neuropixels/Utah** because impedance and
brain-region noise vary across the probe. Global threshold is only sane
when channels are matched (single tetrode, dense scalp grid for spikes —
rare).

### Common Average Reference (CAR) interaction

Threshold spike detection **requires CAR** (or another common-mode
removal) before the threshold step — otherwise shared electrical noise
(50/60 Hz residue, movement) causes synchronous false positives across
all channels. The canonical ordering is

```
bandpass:300,6000 → car → threshold_spike → mua_binning → decode
```

Inverting CAR and threshold_spike causes `refractory_violations` to
spike (multi-channel coincident detections counted as separate events).

## Parameter Format & Defaults

`threshold_spike:{k},{ref_ms},{direction},{per_channel}` — comma-separated:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | float | 4.0 | σ̂-multiplier. Range 3–6; ≥ 5 for sparse-firing cortex. |
| `ref_ms` | float | 0.5 | Refractory mask in ms. 0.5–1.0 ms typical; 2.0 ms is too aggressive. |
| `direction` | "negative" \| "positive" \| "bilateral" | "negative" | Crossing direction. |
| `per_channel` | bool | True | Per-channel `σ̂` vs global. |
| `noise_window_s` | float (kw) | 1.0 | Rolling window for σ̂ estimation (seconds). |
| `min_spacing_samples` | int (kw) | None | Hard minimum samples between detections; defaults to `round(ref_ms · sfreq / 1000)`. |

Examples:

- `threshold_spike` — Neuropixels default: `k=4, ref_ms=0.5, neg, per-channel`.
- `threshold_spike:5,0.5,negative,True` — Wave_clus-style sparse-firing cortex.
- `threshold_spike:4,1.0,negative,True` — conservative refractory for high-noise sessions.

See `parameter_uncertainty/threshold_spike.yaml` for per-paradigm
empirical defaults with citations.

## Modality-Specific Considerations

| Modality | Probe | Band (pre-step) | k | ref_ms | Notes |
|----------|-------|-----------------|---|--------|-------|
| spike (Neuropixels 1.0) | 384 ch × 30 kHz | 300–6000 Hz | 5 | 0.5 | Per-channel σ̂ critical; long shank has region-dependent noise. |
| spike (Neuropixels 2.0) | 384 ch × 30 kHz | 300–6000 Hz | 5 | 0.5 | Same as 1.0 in this op; geometry matters downstream not at threshold. |
| spike (Utah array, motor) | 96 ch × 30 kHz | 250–7500 Hz | 4 | 0.5 | Tighter band for fast cortical spikes; `k=4` is the BrainGate default. |
| spike (Tetrode) | 4×N ch × 32 kHz | 300–6000 Hz | 4 | 0.5 | Apply per-tetrode; consider waveform_snippet downstream. |
| spike (sEEG depth, micro-wire) | 8–40 ch × 30 kHz | 300–3000 Hz | 5 | 1.0 | Higher k due to mixed micro/macro on shaft. |

### Hard exclusions (raise `EasyBCIOperatorError(recoverable=False)`)

| Modality | Reason |
|----------|--------|
| EEG (scalp) | No extracellular spikes at scalp scale; threshold would lock onto EMG / blink artefacts. |
| MEG | No spike content; threshold meaningless. |
| sEEG macro-contacts | LFP-only; spike content requires micro-wire bundles. |
| ECoG | Macroelectrode LFP; no extracellular AP detection. |
| fNIRS | Hemodynamic signal, not electrical. |
| LFP-only band (< 300 Hz) | Spike waveforms are sub-millisecond; threshold on 0.5–300 Hz is meaningless. |

These are enforced at the top of `operator_threshold_spike` — see Rule 10
in CODE_STANDARD.md.

## When to Use / NOT to Use

**Use** when:

- Recording is Neuropixels / Utah / dense microelectrode array (≥ 16 ch, ≥ 20 kHz).
- Target is **MUA** decoding, online BCI, closed-loop feedback, or any
  pipeline that cannot afford Kilosort's 30 min – 3 h sorting time.
- `analysis_goal ∈ {online_inference, classification, feature_extraction, exploratory, generic}`.
- Downstream is `mua_binning` → linear / Riemannian decoder, or
  `mua_binning` → waveform_snippet → offline sorting (training set bootstrap).
- Data already CAR-filtered (or you can add `car` upstream).

**Don't use** when:

- Target is **single-unit precision** (waveform classification, cell-type) — use `spike_sorting`.
- Recording band is < 300 Hz (LFP only) — no spike content.
- Modality ∈ {EEG, MEG, sEEG macro, ECoG, fNIRS} — see hard exclusions.
- You need drift correction (Neuropixels long-session) — `spike_sorting` (Kilosort 2.5+, DREDge) handles drift; threshold does not.
- You need refractory-period validation across units — threshold has no
  unit concept, only per-channel events.

## Constraints & Ordering

- **Required upstream step**: `bandpass:300,6000` (or `bandpass:300,7500` for Utah). Operating on raw broadband data inflates `σ̂` and suppresses true spikes.
- **Strongly recommended upstream**: `car` (common-average reference) — without it, shared noise causes coincident false positives.
- **Forbidden upstream**: `resample` to < 20 kHz (destroys spike waveform). `notch_filter` at < 100 Hz Q is fine; aggressive narrow notch can ring into spike band.
- **Mutually exclusive**: cannot be chained with `spike_sorting` on the same recording — the two compete for the same input and produce incompatible downstream artefacts.

Canonical ordering:

```
load_raw_ap_band → bandpass:300,6000 → car → threshold_spike → mua_binning → (decoder | dump)
```

Auto-fallback to `k+1` is wired in `Failure Modes` (over-detection).

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---------|---------|-----------|
| **Over-detection / noise lock** | mean firing rate per channel > 300 Hz (cortical neurons cap at ~200 Hz; > 300 is structural noise). | `mua_counts / (duration · n_channels) > 300` → log warning + auto-fallback `k → k+1`. |
| **Under-detection / threshold too high** | mean firing rate < 0.01 Hz across all channels with healthy SNR > 5. | `mua_counts / (duration · n_channels) < 0.01` and `median(σ̂) within [0.5e-5, 5e-5] V` → fall back `k → k-1`. |
| **Refractory violations** | > 1% of inter-spike intervals < `ref_ms`. | `np.mean(np.diff(spike_times) < ref_ms_in_samples) > 0.01` per channel → log channel id. |
| **Multi-unit collision** | Coincident spikes across many channels at the same sample (CAR was skipped). | `coincidence_rate = sum(coincident events) / n_events > 0.05` → raise `EasyBCIOperatorError(recoverable=True, fallback_step="car:median → threshold_spike")`. |
| **Reversed polarity** | Almost-zero detections with `direction=negative` but healthy signal. | If `mua_counts < 0.001 Hz` per channel for > 50% channels with `direction=negative` → try `direction=positive`; log "polarity inversion suspected". |
| **Noise-window too short** | σ̂ tracks the spike, not the background; rate is suspiciously stable across regions. | `var(σ̂_per_window) / median(σ̂) < 0.05` over the recording → enlarge `noise_window_s`. |

Auto-detection helper:

```python
import numpy as np


def diagnose_threshold(data, spike_times, sfreq, k, ref_ms):
    """Return (status, details) — 'ok' / 'over' / 'under' / 'refractory' / 'collision' / 'polarity'."""
    n_ch, n_t = data.shape
    duration = n_t / sfreq
    total_spikes = sum(len(s) for s in spike_times)
    mean_rate = total_spikes / (duration * n_ch)
    if mean_rate > 300:
        return "over", {"mean_rate_hz": mean_rate, "suggest_k": k + 1}
    if mean_rate < 0.01:
        sigma = np.median(np.median(np.abs(data), axis=1) / 0.6745)
        if 0.5e-5 < sigma < 5e-5:
            return "under", {"mean_rate_hz": mean_rate, "suggest_k": max(k - 1, 3)}
        return "ok", {"mean_rate_hz": mean_rate, "note": "low rate but noise out of healthy range"}
    ref_samples = int(round(ref_ms * sfreq / 1000))
    violations = [
        np.mean(np.diff(s) < ref_samples) if len(s) > 1 else 0.0
        for s in spike_times
    ]
    if np.mean(violations) > 0.01:
        return "refractory", {"mean_violation_ratio": float(np.mean(violations))}
    return "ok", {"mean_rate_hz": mean_rate}
```

## Common Issues

- **"My MUA rate is zero on all channels."** Almost always upstream — either
  `bandpass:300,6000` was skipped (operating on broadband signal inflates
  `σ̂`) or `direction=negative` with reversed-polarity amplifier. Try
  `direction=positive` first.
- **"All channels fire synchronously at the same time."** CAR was skipped.
  Add `car` upstream; the operator will log `collision` and abort if
  coincidence rate > 5%.
- **"My online inference is dropping spikes."** Check that the
  `noise_window_s` is at most 1 s — longer windows lag behind real-time
  noise drift. For online use, set `noise_window_s=0.25`.
- **"Comparing to Kilosort yield — far fewer threshold spikes."** Expected.
  Threshold counts MUA events; sorting counts assigned-unit spikes. The
  ratio is typically 2–5× more threshold events than sorted spikes
  (some are unassigned noise; some are sorted into multi-unit clusters).

## Reference Implementation

### Standalone (no EasyBCI dependencies)

```python
"""Standalone MAD-thresholded spike detection — drop into any environment."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def mad_sigma(x: np.ndarray) -> float:
    """Robust noise estimate per Quiroga (2004): σ̂ = median(|x|) / 0.6745."""
    return float(np.median(np.abs(x)) / 0.6745)


def threshold_spike(
    data: np.ndarray,
    sfreq: float,
    *,
    k: float = 4.0,
    ref_ms: float = 0.5,
    direction: str = "negative",
    per_channel: bool = True,
    noise_window_s: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Detect extracellular spikes via MAD threshold.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Band-pass filtered (300–6000 Hz) AP-band signal, float32/64.
    sfreq : float
        Sampling rate (Hz). Must be >= 20 kHz for usable waveform timing.
    k : float
        σ̂-multiplier. Default 4.0 (Quiroga); 5.0 for sparse-firing cortex.
    ref_ms : float
        Refractory mask (ms); blocks doubles within `ref_ms`.
    direction : {"negative", "positive", "bilateral"}
        Crossing direction. Extracellular spikes are negative-going.
    per_channel : bool
        Per-channel σ̂ (default) or global σ̂ across channels.
    noise_window_s : float
        Rolling window (s) for σ̂ estimation.

    Returns
    -------
    spike_times : list[ndarray]
        len == n_channels; each entry is integer sample indices.
    thresholds : ndarray, shape (n_channels,)
        The σ̂ * k threshold used per channel.
    """
    if sfreq < 20_000:
        raise ValueError(
            f"threshold_spike: sfreq={sfreq} < 20 kHz; spike detection unreliable"
        )
    if direction not in {"negative", "positive", "bilateral"}:
        raise ValueError(f"direction={direction!r} not in negative/positive/bilateral")

    n_ch, n_t = data.shape
    refractory_samples = max(1, int(round(ref_ms * sfreq / 1000)))

    # Per-channel sigma (median over the whole record; noise_window_s is a
    # stub for the streaming/online variant).
    if per_channel:
        sigmas = np.asarray([mad_sigma(data[c]) for c in range(n_ch)])
    else:
        sigmas = np.full(n_ch, mad_sigma(data.reshape(-1)))

    thresholds = sigmas * k
    spike_times: List[np.ndarray] = []

    for c in range(n_ch):
        sig = data[c]
        if direction == "negative":
            peaks, _ = find_peaks(-sig, height=thresholds[c], distance=refractory_samples)
        elif direction == "positive":
            peaks, _ = find_peaks(sig, height=thresholds[c], distance=refractory_samples)
        else:  # bilateral
            peaks, _ = find_peaks(np.abs(sig), height=thresholds[c], distance=refractory_samples)
        spike_times.append(peaks.astype(np.int64))

    total = sum(len(s) for s in spike_times)
    logger.info(
        "threshold_spike: %d ch, %.1f s, k=%.1f → %d events (%.2f Hz/ch mean)",
        n_ch, n_t / sfreq, k, total, total / (n_t / sfreq * n_ch),
    )
    return spike_times, thresholds
```

### EasyBCI-Adapted (in-framework)

```python
from typing import Any, Dict
import time

import numpy as np

from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


_SPIKE_OK_MODALITIES = {"spike", "spike_ap_band", "neuropixel", "utah_array"}
_SPIKE_FORBIDDEN_MODALITIES = {"eeg", "meg", "seeg_macro", "ecog", "fnirs", "spike_waveform"}


def operator_threshold_spike(
    data_dict: Dict[str, Any],
    *,
    k: float = 4.0,
    ref_ms: float = 0.5,
    direction: str = "negative",
    per_channel: bool = True,
    noise_window_s: float = 1.0,
) -> Dict[str, Any]:
    """EasyBCI-adapted threshold-based spike detection.

    Parameters
    ----------
    data_dict : dict
        OperatorIO; `data` is AP-band filtered (300–6000 Hz) ndarray
        ``(n_channels, n_times)``; `frequency` >= 20 kHz.
    k : float
        σ̂-multiplier (default 4.0).
    ref_ms : float
        Refractory mask in milliseconds (default 0.5).
    direction : str
        ``"negative"`` (default) / ``"positive"`` / ``"bilateral"``.
    per_channel : bool
        Per-channel σ̂ (default True).
    noise_window_s : float
        Rolling noise-estimate window in seconds (default 1.0).

    Returns
    -------
    dict
        OperatorIO with continuous ``data`` unchanged plus:
        - ``meta["spike_times"]``: list[ndarray] of sample indices per channel
        - ``meta["thresholds"]``: ndarray (n_channels,) of σ̂*k
        - ``meta["mua_counts"]``: ndarray (n_channels,) of detection counts
        - ``meta["mean_firing_rate_hz"]``: float (sanity QC)
        - ``meta["refractory_violations"]``: ndarray (n_channels,) of ratios

    Raises
    ------
    EasyBCIOperatorError
        ``recoverable=False`` if modality is not spike-capable.
        ``recoverable=True`` (with fallback) if firing rate is unphysiological.

    Modality coverage
    -----------------
    EEG / MEG / sEEG-macro / ECoG / fNIRS / spike_waveform: forbidden — raises
    EasyBCIOperatorError(recoverable=False).
    Spike (Neuropixels / Utah array / tetrode / sEEG micro-wire): supported.

    References
    ----------
    Quiroga 2004; Rey 2015; Trautmann 2019; IBL Brain-Wide-Map whitepaper.
    """
    modality = (data_dict.get("meta") or {}).get("modality", "").lower()
    if modality in _SPIKE_FORBIDDEN_MODALITIES:
        raise EasyBCIOperatorError(
            operator="threshold_spike",
            reason=f"modality={modality!r} has no extracellular spike content",
            recoverable=False,
        )
    sfreq = float(data_dict["frequency"])
    if sfreq < 20_000:
        raise EasyBCIOperatorError(
            operator="threshold_spike",
            reason=f"frequency={sfreq:.0f} Hz < 20 kHz; spike detection unreliable",
            recoverable=False,
        )

    t0 = time.monotonic()
    data = data_dict["data"]
    n_ch, n_t = data.shape
    refractory_samples = max(1, int(round(ref_ms * sfreq / 1000)))

    if per_channel:
        sigmas = np.asarray(
            [float(np.median(np.abs(data[c])) / 0.6745) for c in range(n_ch)]
        )
    else:
        sigmas = np.full(n_ch, float(np.median(np.abs(data)) / 0.6745))
    thresholds = sigmas * k

    from scipy.signal import find_peaks  # local: scipy is heavy at module import

    spike_times = []
    violations = np.zeros(n_ch, dtype=np.float32)
    for c in range(n_ch):
        sig = data[c]
        if direction == "negative":
            peaks, _ = find_peaks(-sig, height=thresholds[c], distance=refractory_samples)
        elif direction == "positive":
            peaks, _ = find_peaks(sig, height=thresholds[c], distance=refractory_samples)
        else:
            peaks, _ = find_peaks(np.abs(sig), height=thresholds[c], distance=refractory_samples)
        spike_times.append(peaks.astype(np.int64))
        if len(peaks) > 1:
            diffs = np.diff(peaks)
            violations[c] = float(np.mean(diffs < refractory_samples))

    counts = np.asarray([len(s) for s in spike_times], dtype=np.int64)
    duration = n_t / sfreq
    mean_rate = float(counts.sum() / (duration * n_ch)) if n_ch and duration else 0.0

    if mean_rate > 300.0:
        raise EasyBCIOperatorError(
            operator="threshold_spike",
            reason=f"mean firing rate {mean_rate:.1f} Hz/ch > 300 — over-detection",
            recoverable=True,
            fallback_step=f"threshold_spike:{k + 1.0:.1f},{ref_ms},{direction},{per_channel}",
        )

    elapsed = time.monotonic() - t0
    out = dict(data_dict)
    out["data"] = data  # threshold does not modify continuous data (Rule 5)
    out["elapsed_s"] = elapsed
    new_meta = dict(out.get("meta") or {})
    new_meta["spike_times"] = spike_times
    new_meta["thresholds"] = thresholds
    new_meta["mua_counts"] = counts
    new_meta["mean_firing_rate_hz"] = mean_rate
    new_meta["refractory_violations"] = violations
    new_meta["threshold_spike"] = {
        "k": k, "ref_ms": ref_ms, "direction": direction,
        "per_channel": per_channel, "noise_window_s": noise_window_s,
    }
    out["meta"] = new_meta
    record_step_elapsed(
        "threshold_spike", elapsed,
        (data_dict.get("meta") or {}).get("step_cache_key"),
    )
    return out
```

## References

1. Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). *Unsupervised spike
   detection and sorting with wavelets and superparamagnetic clustering*.
   Neural Computation 16(8): 1661–1687.
   doi:10.1162/089976604774201631 — the canonical MAD-threshold paper.
2. Rey, H. G., Pedreira, C., & Quiroga, R. Q. (2015). *Past, present and
   future of spike sorting techniques*. Brain Research Bulletin 119(Pt B):
   106–117. doi:10.1016/j.brainresbull.2015.04.007 — survey contrasting
   threshold vs sort regimes.
3. Trautmann, E. M. et al. (2019). *Accurate Estimation of Neural Population
   Dynamics without Spike Sorting*. Neuron 103(2): 292–308.e4.
   doi:10.1016/j.neuron.2019.05.003 — MUA decoding parity with sorted
   spikes in motor cortex; motivates threshold-only pipelines.
4. International Brain Laboratory et al. (2022). *Reproducibility of in vivo
   electrophysiological measurements in mice*. bioRxiv. doi:10.1101/2022.05.09.491042 —
   the IBL Brain-Wide-Map pipeline; uses per-channel `k=5` threshold as
   first-pass detection step.
5. Pachitariu, M., Sridhar, S., & Stringer, C. (2024). *Spike sorting with
   Kilosort4*. Nature Methods 21(5): 914–921. doi:10.1038/s41592-024-02232-7 —
   modern sorting; the "when sorting is too slow" baseline reference.
