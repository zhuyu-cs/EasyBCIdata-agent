---
name: sleep_staging
layer: L2
group: clinical
metadata:
  analysis_goal_allowed:
  - sleep_staging
  - clinical_screening
  - feature_extraction
  - exploratory
  analysis_goal_forbidden:
  - online_inference
tags:
- eeg
- sleep
- staging
- polysomnography
- psg
- spindle
- k_complex
- slow_wave
- rem
- nrem
- hypnogram
- respiratory
- spo2
- plm
modality: eeg
---
# Polysomnography (PSG) Sleep Preprocessing

## Overview

Full polysomnography preprocessing for sleep studies — multi-channel data
from heterogeneous signal types (EEG, EOG, EMG, respiratory, SpO2, limb,
position) with per-route filtering, clinical event detection, and
epoch-level quality assessment. Designed for Compumedics `.SLP` bundles and
EDF-based PSG, handling mixed native sampling rates via resample-to-one.

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Format | Compumedics .SLP (directory bundle) or EDF |
| EEG channels | 2–6 (C3/C4/O1/O2/F3/F4 vs M1/M2) |
| EOG channels | 2 (E1-M2, E2-M2) |
| EMG channels | 2–4 (chin, limb-L/R) |
| Respiratory | Airflow (NPress/CPress/Therm), Effort (Thor/Abdo) |
| Oximetry | SpO2 (1 Hz), Pulse (1 Hz), Pleth (200 Hz) |
| Position | Enum channel (supine/lateral/prone) |
| Recording duration | 6–10 hours (full night) |
| Native rates | 200 Hz (EEG/EOG/EMG/Pleth), 25 Hz (respiratory), 1 Hz (SpO2) |
| Common resample target | 100 Hz (AASM standard for staging) |
| Epoch length | 30 seconds (AASM) |

## Channel Routing Strategy

PSG channels are grouped into routes with independent filtering parameters.
The generated pipeline processes each route with its optimal settings before
merging into the final output array.

| Route | Channels | Bandpass | Notch | Rationale |
|-------|----------|----------|-------|-----------|
| EEG | F3/F4/C3/C4/O1/O2-M1/M2 | 0.3–35 Hz | Yes (50/60) | Preserve delta (0.5–2 Hz) + spindles (11–16 Hz); nothing above 35 Hz relevant. |
| EOG | E1-M2, E2-M2 | 0.3–10 Hz | No | Capture slow eye movements; fast activity is crosstalk. Do NOT artifact-reject (REM detection depends on this). |
| EMG (chin) | EMG-L, EMG-R | 10–100 Hz | Yes | Isolate muscle tone; rectify + RMS for stage discrimination. Do NOT ICA (signal IS the target). |
| Respiratory | NPress, CPress, Thor, Abdo, Therm | 0.05–3 Hz | No | Normal breathing 0.15–0.5 Hz; preserve waveform morphology for apnea detection. |
| SpO2/Pulse | SpO2, Pulse | None | No | Already 1 Hz sampled; any filtering destroys clinical values. |
| Pleth | Pleth | 0.5–8 Hz | No | Pulse waveform envelope. |
| Position | Position | None | No | Enum/step — meta only, not in signal array. |

## Recommended Pipeline

```yaml
pipeline:
  # Phase 1: Global cleanup
  - notch:auto                  # 50/60 Hz on EEG + EMG routes only
  - drop_bads:auto              # Remove flat/saturated channels

  # Phase 2: Per-route filtering (codegen generates per-route bandpass calls)
  - bandpass:0.3,35             # EEG route
  - bandpass:0.3,10             # EOG route
  - bandpass:10,100             # EMG route
  - bandpass:0.05,3             # Respiratory route

  # Phase 3: PSG-specific clinical detection (at native rate)
  - respiratory_events          # Apnea/hypopnea/desat (AASM criteria)
  - plm_detect                  # Periodic limb movements (WASM criteria)

  # Phase 4: Downsample
  - resample:100                # AASM standard; EEG/EOG/EMG only

  # Phase 5: Epoch quality
  - epoch_qc_sleep              # 30s epoch-level QC, hypnogram-aware
```

### Step Rationale

1. **notch:auto** — Remove line noise. Applied only to EEG and EMG routes
   (respiratory/SpO2 are too low-rate; EOG does not benefit).
2. **drop_bads:auto** — Detect and remove flat/railed channels across all
   routes before filtering (prevents filter ringing on dead channels).
3. **Per-route bandpass** — Each signal type has fundamentally different
   frequency content. A single bandpass cannot serve all.
4. **respiratory_events** — Must run BEFORE resample; needs native 25 Hz
   for accurate amplitude envelope on airflow/effort.
5. **plm_detect** — Must run BEFORE resample; needs native 200 Hz for
   0.5–10 s event duration accuracy on limb EMG.
6. **resample:100** — AASM standard for staging. Applied to EEG/EOG/EMG.
   Respiratory (25 Hz) and SpO2 (1 Hz) stay at native rate.
7. **epoch_qc_sleep** — After all processing; scores each 30 s epoch
   across channels, adapts thresholds to sleep stage when hypnogram available.

## Critical Constraints

- **DO NOT high-pass above 0.5 Hz** — slow waves (0.5–2 Hz) define N3.
  Even 0.3 Hz is aggressive; 0.1 Hz is safer if DC drift is not extreme.
- **DO NOT ICA on EOG/EMG** — these signals ARE the features for staging,
  not artifacts to remove.
- **DO NOT drop PSG auxiliary channels** — respiratory/SpO2/position are
  clinically essential. The `sleep_staging` goal sets `inject_drop_nondata=False`.
- **Hypnogram is OPTIONAL** — study may be unscored (all epochs `0x80`).
  All operators degrade gracefully without labels.
- **30-second epochs are non-negotiable** — AASM scoring unit. Do not use
  other epoch lengths for staging-related analysis.
- **Resample target ≤ 256 Hz** — sufficient for all sleep features
  (spindle peak ~13 Hz, nothing above 35 Hz). 100 Hz is standard.

## Sleep Stages (AASM Scoring Manual)

| Stage | EEG Features | Typical % TST |
|-------|-------------|---------------|
| Wake (W) | Alpha (8–13 Hz) posterior, eye blinks | Variable |
| N1 | Theta (4–7 Hz), vertex sharp waves | 5% |
| N2 | Sleep spindles (12–14 Hz), K-complexes | 45–55% |
| N3 (SWS) | Delta (0.5–2 Hz) > 75 µV, ≥ 20% epoch | 15–25% |
| REM | Low-voltage mixed, sawtooth waves, rapid eye movements | 20–25% |

## Key Graphoelements

| Feature | Frequency | Duration | Amplitude | Location |
|---------|-----------|----------|-----------|----------|
| Sleep spindle | 11–16 Hz | 0.5–2 s | 25–50 µV | Central (C3/C4) |
| K-complex | 0.5–1.5 Hz | > 0.5 s | > 75 µV | Frontal |
| Slow oscillation | 0.5–1 Hz | 0.8–2 s | > 75 µV | Frontal |
| Sawtooth wave | 2–6 Hz | Trains | 20–50 µV | Central/frontal |

## Multi-Channel Quality Indicators

| Indicator | Channel | Method | Clinical Use |
|-----------|---------|--------|-------------|
| EMG tone | Chin EMG | RMS envelope, percentile per stage | Wake vs REM discrimination |
| REM density | EOG | Rapid movements per minute | REM characterization |
| SpO2 nadir | SpO2 | Minimum in event window | Severity grading |
| PLM index | Limb EMG | Events/hour (WASM series criteria) | RLS screening |
| AHI | Airflow + SpO2 | Apnea+hypopnea events/hour | OSA severity |

## Complete Pipeline Example

```python
import numpy as np
from easybci_lib.tools.neural_processing.io.loader import load_neural
from easybci_lib.tools.neural_processing.io.psg_annotations import (
    parse_hypnogram, parse_events,
)

# Load PSG data (handles Compumedics .SLP bundles: mixed native rates →
# single common rate, per-channel sensitivity scaling).
loaded = load_neural("STUDY.SLP", target_hz=100.0)
data = loaded["data"]               # (n_channels, n_samples), physical units
sfreq = loaded["frequency"]         # 100.0
channels = loaded["channels"]
meta = loaded["meta"]

# Parse annotations (both degrade gracefully if missing/unscored)
stages = parse_hypnogram(meta["hypnogram_path"])
events, events_hint = parse_events(meta["events_path"])

# Route channels by type for per-route filtering
eeg_idx = [i for i, c in enumerate(channels) if any(
    k in c for k in ("F3", "F4", "C3", "C4", "O1", "O2"))]
eog_idx = [i for i, c in enumerate(channels) if "E1" in c or "E2" in c]
emg_idx = [i for i, c in enumerate(channels) if "EMG" in c]
resp_idx = [i for i, c in enumerate(channels) if any(
    k in c for k in ("Thor", "Abdo", "NPress", "CPress", "Therm"))]

from scipy.signal import butter, filtfilt, sosfilt

def _bandpass(sig, lo, hi, fs):
    sos = butter(4, [lo / (fs/2), hi / (fs/2)], btype="bandpass", output="sos")
    return sosfilt(sos, sig, axis=-1).astype(np.float32)

# Apply per-route bandpass
for i in eeg_idx:
    data[i] = _bandpass(data[i], 0.3, 35, sfreq)
for i in eog_idx:
    data[i] = _bandpass(data[i], 0.3, 10, sfreq)
for i in emg_idx:
    data[i] = _bandpass(data[i], 10, min(100, sfreq/2 - 1), sfreq)
for i in resp_idx:
    data[i] = _bandpass(data[i], 0.05, 3, sfreq)

# Epoch into 30-second windows (AASM standard)
epoch_samples = int(30.0 * sfreq)
n_epochs = data.shape[1] // epoch_samples
epochs = data[:, :n_epochs * epoch_samples].reshape(
    len(channels), n_epochs, epoch_samples)
epochs = epochs.transpose(1, 0, 2)   # (n_epochs, n_channels, n_samples)

# Stage labels aligned to epochs (truncate/pad if length mismatch)
epoch_stages = stages[:n_epochs] if stages else ["unscored"] * n_epochs
```

## References

1. Berry, R. B. et al. (2020). *The AASM manual for the scoring of
   sleep and associated events*, v2.6. American Academy of Sleep Medicine.
2. Iber, C. et al. (2007). *The AASM manual for the scoring of sleep*,
   1st edition. AASM.
3. Warby, S. C. et al. (2014). *Sleep-spindle detection: crowdsourcing
   and evaluating performance of experts, non-experts and automated
   methods*. Nature Methods 11(4): 385–392. doi:10.1038/nmeth.2855.
4. Zucconi, M. et al. (2006). *WASM standards for recording and scoring
   PLM*. Sleep Med. 7(2): 175–183. doi:10.1016/j.sleep.2005.12.008.
