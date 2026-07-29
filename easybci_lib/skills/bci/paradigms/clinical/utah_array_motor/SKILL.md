---
name: utah_array_motor
description: "Utah array (96 ch, motor cortex) — threshold MUA decoding for human motor BCI (BrainGate-class)"
layer: L2
group: clinical
metadata:
  tags: [utah_array, motor_cortex, m1, braingate, tetraplegia, cursor, kalman, mua]
  modalities: [spike]
  paradigms: [motor_decoding, cursor_control, neural_prosthesis]
  analysis_goal_allowed:
    - classification
    - feature_extraction
    - clinical_screening
    - online_inference
  analysis_goal_forbidden:
    - source_localization
    - phase_amplitude_coupling
---
# Utah Array Motor Decoding

## Neuroscience Background

The **Utah Multi-Electrode Array** (UEA, Blackrock Microsystems) is the
clinically deployed 96-channel intracortical probe used in human motor
BCI trials since 2004 — the BrainGate consortium, the BrainGate2
clinical trial, and the Shenoy / Chestek labs. Each UEA is a
4 × 4 mm silicon shank with 10 × 10 platinum-iridium electrodes (one
corner inactive: 96 active), 1.5 mm length, implanted in the **hand
knob of primary motor cortex (M1)**.

The dominant analysis paradigm is **threshold-based MUA decoding of
movement intent** — Hochberg et al. (2012) showed that a tetraplegic
patient could control a robotic arm via a 96 ch UEA + Kalman decoder,
and the BrainGate cohort has reached cursor control at multiple-words-
per-minute typing throughput (Willett et al. 2021) using analogous
threshold MUA features. The UEA samples at 30 kHz; signals are
band-passed (250 Hz – 7.5 kHz) and threshold-crossed at the amplifier
to produce per-channel spike events in real time.

Why **threshold instead of sorting** in human motor BCI:

- **Latency**: BrainGate uses 10–20 ms windows for online cursor control;
  Kilosort sorts in 30 min – 3 h, ruling out sorting for closed-loop use.
- **Decoder parity**: Christie et al. (2015), Trautmann et al. (2019)
  demonstrated that MUA decoding accuracy is within 5–10% of sorted
  decoding on motor tasks — the marginal gain from sorting does not
  justify the throughput cost in clinical context.
- **Stability**: Threshold MUA is robust to chronic-implant signal drift
  (months to years post-implant); single-unit isolation degrades faster.

## Probe & Channel Selection

| Parameter | Default | Notes |
|---|---|---|
| Active channels | 96 (10×10 − 4 corner exclusions) | The 4 corners and any explicitly broken sites are dropped at acquisition. |
| Sample rate | 30 kHz (Blackrock NSP) | The Blackrock `.ns5` raw-band stream. |
| Filter cascade | 250 – 7500 Hz | Tighter than Neuropixels (300–6000 Hz); Blackrock-default. |
| Threshold | −4.5 × RMS per channel | Set in Cerebus / NSP firmware; **negative-going** crossings only. |
| Refractory | 0.5 ms | Hardware-enforced on Blackrock NSP. |

**Channel selection conventions**:

- **Keep all 96 channels** by default; per-channel SNR varies along the
  array but sorting / clustering is not the analytical product.
- **Drop "dead" channels** flagged by impedance > 1 MΩ or peak-to-peak
  voltage in dead range (`np.ptp < 1e-6 V`).
- **For dual-array implants** (Hochberg / BrainGate2 protocols use 1 ×
  motor + 1 × premotor array): keep both as parallel populations,
  concat to 192 ch.

## Frequency Bands

| Band | Range | Use |
|---|---|---|
| Spike (AP) | 250 – 7500 Hz | Primary decoding feature (threshold MUA). |
| LFP | 0.1 – 250 Hz (low-cut acquisition) | Secondary; small-marginal information for cursor decoding. |
| Spike-band power (SBP) | 300 – 1000 Hz envelope | Alternative to threshold MUA; Chestek-lab work, Brandman 2018. |

The two main feature streams in deployed BCIs are:

1. **Threshold MUA spike rates** (BrainGate canonical), binned at 10–20 ms.
2. **Spike-band power (SBP)**, computed by Hilbert envelope of the 300–1000 Hz
   filtered signal then binning at 10–20 ms. SBP is more stable across
   sessions when probe drift causes unit turnover.

Both are computed with the same upstream `bandpass → CAR` chain.

## Recommended Pipeline

### Threshold MUA path (canonical BrainGate, **online**)

```
load_ns5 → bandpass:250,7500 → car → threshold_spike:k=4,ref_ms=0.5 →
mua_binning:20,True,rate → riemannian_covariance / linear_kalman → cursor
```

When to take this path:
- `analysis_goal ∈ {online_inference, classification, clinical_screening}`
- Subject is **chronically implanted human** (BrainGate / Synchron /
  Neuralink class).
- Latency budget per decoder step < 30 ms.

### Spike-band power (SBP) path (drift-robust)

```
load_ns5 → bandpass:300,1000 → car → hilbert → log_band_power → bin_20ms → decoder
```

When to take this path:
- Probe has been implanted **> 6 months** (unit population drifting).
- Decoding goal is dimensions-of-movement (X/Y velocity) rather than
  discrete categories.

### Offline analysis path (sorting, post-experiment)

```
load_ns5 → bandpass:250,7500 → spike_sort (MountainSort5) → quality_metrics →
bin_spikes:25 → trial_averaged_psth → tuning_curves
```

When:
- Offline cell-type / tuning characterization
- Publication-grade single-unit analysis
- `analysis_goal ∈ {feature_extraction, exploratory}`

## Common Artifacts

| Artifact | Cause | Mitigation |
|---|---|---|
| **Movement-coupled common-mode** | Subject head movement modulates reference. | CAR (mandatory for threshold path). |
| **60 Hz line noise + ground loops** | Clinical environment with many devices. | `notch_filter:60` + harmonics; `zapline` if non-stationary. |
| **Stim artefacts (if iEEG stim concurrent)** | DBS / functional mapping stimulation. | Stim-locked blanking (mask ±2 ms around stim TTL). |
| **Chronic drift** | Months-post-implant: spikes drift across channels. | Use SBP feature instead of threshold MUA; re-fit decoder weekly (BrainGate protocol). |
| **Dead / saturated channel** | Broken electrode contact. | Drop via `drop_bads:peak_to_peak<1e-6 OR std>10*median`. |

## Quality Metrics

For the threshold MUA path:

| Metric | Healthy chronic implant | Degraded |
|---|---|---|
| Mean threshold-crossing rate (per channel) | 10–100 Hz | < 1 Hz or > 200 Hz |
| Refractory violation ratio | < 1% | > 5% |
| SNR per channel (RMS / σ̂) | > 3 | < 1.5 |
| Channels with usable spikes (rate > 1 Hz) | 60–90 / 96 | < 30 |
| Coincidence rate across channels | < 2% | > 10% (CAR missing or stim artefact) |

For the offline sorted path: standard `spike_sorting` quality metrics
(ISI / SNR / presence ratio).

## Classification / Decoding Baselines

Standard tasks in the literature:

| Task | Decoder | Typical accuracy (chronic BrainGate) |
|---|---|---|
| 2D cursor velocity | Linear (Wiener) | ~70–85% bit rate vs ground-truth target |
| 2D cursor velocity | Kalman | +5–10% over Wiener; published default since 2008 |
| 4-class movement (hand grasp / open / left / right) | Linear / Riemannian MDM | 85–95% |
| 26-class typing (BrainGate2 character set) | RNN over SBP | ~90 cpm @ word error rate < 25% (Willett 2021) |
| Handwriting decoding | RNN over neural population | ~ 90 cpm (Willett 2021) |

## Public Datasets

| Dataset | Size | Format | Access |
|---|---|---|---|
| BrainGate participant-T11 sessions | Multiple-month chronic data | `.ns5` + behavioral | Restricted (IRB; through BrainGate consortium) |
| Brand-T7 motor cortex (Pandarinath 2018) | LFADS benchmark | NWB | Public on DANDI |
| Hochberg 2012 (Nature) data | Robotic arm reach sessions | `.ns5` | Restricted (clinical) |
| FALCON benchmark (Karpowicz 2024) | M1 + S1 multi-day | NWB | Public neurobench.io |

## Pitfalls & Failure Modes

- **Forgetting CAR.** Without CAR, threshold path locks onto 60 Hz +
  movement common-mode; coincidence rate balloons to 50%+. The CAR is
  hardware-implementable on Cerebus NSP and is the default in BrainGate
  protocols.
- **Per-channel vs global threshold.** UEA channels span ~3–10× SNR
  range across the array; **always** use per-channel `k·σ̂`.
- **Refractory too aggressive.** BrainGate default is 0.5 ms; some
  papers use 1.0 ms. > 1.0 ms over-suppresses high-firing units (motor
  pyramidal cells can reach 80–120 Hz transient firing during overt
  movement intent).
- **Comparing to monkey results.** Human M1 firing rates are typically
  lower than rhesus (lower training duration, paralyzed limb), so do
  not transfer Hochberg's monkey thresholds verbatim — recompute σ̂.
- **Stimulation contamination.** If a concurrent DBS / functional
  mapping stim is active, blank ±2 ms around stim TTL; do not rely on
  bandpass to suppress.

## Boundary with Related Paradigms

| Related paradigm | Boundary |
|---|---|
| **`neuropixel_population.md`** | Use Neuropixels paradigm when the recording is a research-grade Si-CMOS probe (rodent / NHP / human research). UEA is the **clinical chronic implant** path with fixed geometry and clinical-grade behavioral / decoder protocols. |
| **`closed_loop_bci.md`** | UEA → cursor / typing is a closed-loop BCI; that paradigm specifies the loop semantics (safety guards, feedback timing). This paradigm specifies the upstream feature pipeline. |
| **`online_inference.md`** | The "online MUA" path here defers to `online_inference` for decoder latency budget contract. This paradigm specifies the front-end (acquire → bandpass → CAR → threshold → bin). |
| **`motor_imagery.md`** | MI is **EEG-based** (scalp), no implant; entirely separate physiology and decoding feature set. Don't confuse the two. |

## Standalone End-to-End Script

```python
"""Utah array .ns5 → threshold MUA → 20 ms population rate matrix."""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
import neo


NS5_PATH = "/data/braingate_session.ns5"
OUTPUT = Path("/tmp/uea_mua")
OUTPUT.mkdir(exist_ok=True)


# 1. Load 30 kHz raw band via neo
reader = neo.rawio.BlackrockRawIO(filename=str(NS5_PATH))
reader.parse_header()
sig = reader.get_analogsignal_chunk(channel_indexes=None)        # int16, (n_t, n_ch)
sig = sig.T.astype(np.float32) * reader.header["signal_channels"]["gain"][0]
sfreq = float(reader.get_signal_sampling_rate())
print(f"Loaded {sig.shape} @ {sfreq} Hz")


# 2. Drop dead channels
ptp = np.ptp(sig, axis=1)
keep = (ptp > 1e-6) & (ptp < 1e-2)        # 1 µV to 10 mV is realistic
sig = sig[keep]
print(f"After dead-channel drop: {sig.shape[0]} ch")


# 3. Bandpass 250-7500 Hz via mne (or scipy)
from scipy.signal import butter, sosfiltfilt
sos = butter(4, [250, 7500], btype="bandpass", fs=sfreq, output="sos")
sig = sosfiltfilt(sos, sig, axis=1).astype(np.float32)


# 4. CAR (median)
sig = sig - np.median(sig, axis=0, keepdims=True)


# 5. MAD per-channel threshold
sigma = np.median(np.abs(sig), axis=1, keepdims=True) / 0.6745
threshold = 4.0 * sigma                        # k=4 per BrainGate default
ref_samp = int(round(0.5 * sfreq / 1000))


# 6. Detect (negative crossings)
spike_times = []
for c in range(sig.shape[0]):
    peaks, _ = find_peaks(-sig[c], height=threshold[c, 0], distance=ref_samp)
    spike_times.append(peaks)


# 7. Bin at 20 ms
duration = sig.shape[1] / sfreq
bin_s = 0.020
n_bins = int(duration / bin_s)
edges = np.arange(n_bins + 1) * bin_s
rates = np.zeros((sig.shape[0], n_bins), dtype=np.float32)
for i, idx in enumerate(spike_times):
    counts, _ = np.histogram(idx / sfreq, bins=edges)
    rates[i] = counts / bin_s

np.save(OUTPUT / "mua_rate_20ms.npy", rates)
print(f"MUA rate matrix: {rates.shape} → {OUTPUT}/mua_rate_20ms.npy")
```

## EasyBCI Pipeline Spec

### Threshold MUA path (canonical, online)

```yaml
modality: spike
paradigm: utah_array_motor
analysis_goal: online_inference
steps:
  - load:blackrock,stream=ns5
  - drop_bads:peak_to_peak
  - bandpass:250,7500
  - car:median
  - threshold_spike:4,0.5,negative,True
  - mua_binning:20,True,rate
```

### SBP path (drift-robust)

```yaml
modality: spike
paradigm: utah_array_motor
analysis_goal: online_inference
steps:
  - load:blackrock,stream=ns5
  - drop_bads:peak_to_peak
  - bandpass:300,1000
  - car:median
  - hilbert
  - log_band_power
```

### Offline sorted path (research / publication)

```yaml
modality: spike
paradigm: utah_array_motor
analysis_goal: feature_extraction
steps:
  - load:blackrock,stream=ns5
  - drop_bads:peak_to_peak
  - bandpass:250,7500
  - spike_sort:isi_violation_threshold=0.5,snr_threshold=5.0
  - bin_spikes:25
```

## References

1. Hochberg, L. R. et al. (2012). *Reach and grasp by people with
   tetraplegia using a neurally controlled robotic arm*. Nature 485:
   372–375. doi:10.1038/nature11076 — BrainGate canonical paper; UEA + Kalman.
2. Pandarinath, C. et al. (2017). *High performance communication by
   people with paralysis using an intracortical brain-computer
   interface*. eLife 6: e18554. doi:10.7554/eLife.18554 — typing throughput
   benchmark; threshold MUA features.
3. Willett, F. R. et al. (2021). *High-performance brain-to-text
   communication via handwriting*. Nature 593: 249–254.
   doi:10.1038/s41586-021-03506-2 — handwriting decoder; RNN over SBP.
4. Christie, B. P. et al. (2015). *Comparison of spike sorting and
   thresholding of voltage waveforms for intracortical brain-machine
   interface performance*. J. Neural Eng. 12(1): 016009.
   doi:10.1088/1741-2560/12/1/016009 — quantifies the threshold-vs-sort
   parity gap on UEA motor decoding.
5. Trautmann, E. M. et al. (2019). *Accurate Estimation of Neural
   Population Dynamics without Spike Sorting*. Neuron 103(2): 292–308.
   doi:10.1016/j.neuron.2019.05.003 — threshold-MUA decoding parity.
6. Stavisky, S. D. et al. (2019). *Speech-related dorsal motor cortex
   activity does not interfere with iBCI cursor control*. J. Neural Eng.
   16: 056022. doi:10.1088/1741-2552/ab4c30 — UEA in clinical motor
   protocol; CAR + threshold default.
7. Brandman, D. M. et al. (2018). *Robust closed-loop control of a
   cursor in a person with tetraplegia using Gaussian process regression*.
   Neural Computation 30(11): 2986–3008. doi:10.1162/neco_a_01129 — SBP
   feature for chronic stability.
