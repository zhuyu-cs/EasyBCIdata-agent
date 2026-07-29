---
name: connectivity
description: Functional connectivity analysis (phase-locking, coherence, Granger)
  — preserves channel structure
version: 2.0.0
layer: L2
group: analysis
metadata:
  modalities:
  - eeg
  - meg
  - seeg
  - ecog
  - lfp
  paradigms:
  - connectivity
  - dynamic_connectivity
  - resting_state
  analysis_goal_allowed:
  - connectivity
  - exploratory
  - feature_extraction
  - source_localization
  analysis_goal_forbidden:
  - online_inference
analysis_goal: connectivity
tags:
- connectivity
- phase_locking
- coherence
- granger
- dtf
- pdc
- wpli
- plv
- pli
---
# Functional Connectivity

## Neuroscience Background

Functional connectivity (FC) measures statistical dependence between
recordings at *different* sites — the unit of analysis is the **channel
pair**, not the channel.  Phase-based measures (PLV, PLI, wPLI) ask
whether two channels share a consistent phase relationship; magnitude-
based measures (coherence) add amplitude correlation; directed measures
(Granger, DTF, PDC) ask whether one channel predicts the other.  All
non-trivial FC interpretations require explicit handling of **volume
conduction** in EEG/MEG and **shared reference** artefacts in iEEG.

Distinguish from related concepts:

- **Effective connectivity**: causal influence from a model (DCM, Granger).
- **Structural connectivity**: physical wiring (DTI / fiber tracts).
- **Functional connectivity** (this skill): statistical dependence in
  recorded activity, agnostic to mechanism.

## Channel Selection (electrode positions)

FC is meaningful only when **all clean channels** are kept — every
dropped channel removes one row/column from the connectivity matrix.

| Modality | Note |
|----------|------|
| EEG (10–20) | Use ≥ 32 channels for stable cortex-wide FC; 16 channels usable for ROI-pair tests. |
| EEG (high-density 128/256) | Source-projected FC (sLORETA / beamformer) is more interpretable than sensor-space. |
| MEG | Sensor-space FC is biased by spatial leakage; project to source space and use leakage-corrected FC. |
| sEEG / ECoG | Use bipolar reference within each lead; reject contact pairs adjacent to white matter. |
| LFP | Each contact independently; cross-correlogram first to set lag bounds. |

Critical: `drop_nondata_channels` is NOT injected for the `connectivity`
analysis goal (REGISTRY enforces `inject_drop_nondata=False`).  EOG / ECG
channels stay in the matrix as control rows — they should show **no**
genuine connectivity to brain channels and serve as a sanity check.

## Frequency Bands of Interest

| Band | Range (Hz) | Common FC use |
|------|------------|---------------|
| Delta | 0.5–4 | Sleep, anaesthesia; long-range cortical FC. |
| Theta | 4–8 | Working memory FC (frontoparietal); hippocampal-cortical. |
| Alpha | 8–13 | Resting-state network FC; visual gating. |
| Beta | 13–30 | Sensorimotor binding; long-range top-down attention. |
| Low gamma | 30–80 | Local feature binding; bistable perception. |
| High gamma | 80–150 | Local processing; spatial precision. |
| HFO | 80–500 | iEEG only; epileptogenic biomarker. |

Connectivity metrics are **frequency-specific by construction**: PLV at
12 Hz says nothing about PLV at 30 Hz on the same channel pair.  Always
report metric *and* band.

## Recommended Pipeline

```
notch:50 (NARROW Q≥30) → bandpass:1,80 → drop_bads:auto → ICA:gentle → bipolar_ref or laplacian_ref → epoch → connectivity_metric
```

Step rationale:

1. **Narrow notch.**  Aggressive notch (Q < 10) ripples in the passband
   and distorts cross-channel phase.  REGISTRY enforces
   `allow_aggressive_notch=False` for connectivity.
2. **Wide bandpass.**  Cover all bands of interest in one pass; per-band
   bandpassing happens at metric time so the input stays consistent.
3. **drop_bads:auto** — flat / spiky channels poison every pair they
   appear in.  Tag them, drop them; never interpolate (interpolation
   spreads neighbouring signal into the channel).
4. **ICA: gentle.**  Remove obvious EOG / ECG / EMG only.  Aggressive
   ICA strips genuine cortical components and creates near-zero PLV
   across the whole matrix (false negative).
5. **Re-reference to local source.**  Bipolar (sEEG / ECoG) or surface
   Laplacian (high-density EEG) makes "channel" a local-field measure.
   Common-average reference (CAR) injects the global mean into every
   channel and creates **spurious** connectivity — its variance
   contribution increases with channel count, so 64-ch CAR is much
   worse than 16-ch CAR.
6. **Epoch.**  Stationary connectivity per epoch (pre-cue, post-cue,
   etc.).  Avoid >30 s epochs without verifying weak-stationarity.

Forbidden steps:
- `scale:per_channel` — destroys cross-channel amplitude relationships
  that coherence relies on.
- `interpolate_bads` — creates artificially high FC between
  interpolated channels and their neighbours.

## Connectivity Metrics — Choice Matrix

| Metric | Volume conduction robust? | Direction? | Output | Use when |
|--------|--------------------------|-----------|--------|---------|
| **Coherence** | No (high false positives in EEG) | No | Magnitude-squared per band | Single-modality, well-referenced sEEG / ECoG. |
| **PLV (phase-locking value)** | No | No | Magnitude in [0, 1] | Cleanly referenced data; preferred for sEEG / ECoG. |
| **PLI (phase-lag index)** | **Yes** | No | Asymmetry index | Scalp EEG / MEG sensor space — the default safe choice. |
| **wPLI (weighted PLI)** | **Yes** (best) | No | Weighted phase asymmetry | When the noise floor matters; superior to plain PLI. |
| **Granger causality** | Partial | **Yes** | Transfer function | Stationary epochs, ≥ 200 samples. |
| **DTF (Directed Transfer Function)** | Partial | **Yes** | Per-band directed | Same as Granger but multivariate-conditioned. |
| **PDC (Partial Directed Coherence)** | Partial | **Yes** | Per-band directed | Like DTF but normalised across senders. |

**Decision tree.**

```
EEG / MEG sensor space?  → wPLI (skip plain coherence)
sEEG / ECoG bipolar?     → PLV or coherence
Need direction?          → Add Granger / DTF / PDC after PLV/wPLI
Volume conduction strong? → wPLI mandatory; consider source projection
```

## Common Artifacts

| Artifact | Symptom on FC | Treatment |
|----------|---------------|-----------|
| Common reference (CAR over few channels) | Spurious near-uniform FC | Switch to bipolar / Laplacian. |
| Volume conduction | High coherence on neighbouring channels at all freqs | Use PLI / wPLI / source-space. |
| Single bad channel | "Star pattern" — high FC from that channel to all others | Drop it; do not interpolate. |
| Eye-blink leakage | Frontopolar high FC to whole scalp at < 5 Hz | ICA (gentle) on EOG component. |
| ECG | Periodic 1–2 Hz coupling everywhere | ICA on ECG component, or notch around heart-rate harmonics. |

## Quality Metrics

- **Surrogate FC**: shuffle phase / time across trials; compute the same
  metric.  Real FC must exceed surrogate p < 0.05 after multiple
  comparisons correction (FDR over channel pairs).
- **N-effective**: ≥ 100 epochs for stable PLV / PLI; ≥ 200 for Granger.
- **Reference channel sanity**: EOG ↔ Cz wPLI should be ~0; if > 0.1,
  ICA was insufficient.

Recommended grade thresholds:

| Grade | Surrogate-corrected p | N_epochs | EOG↔EEG wPLI |
|-------|------------------------|----------|--------------|
| PASS  | ≥ 5% pairs significant | ≥ 100 | < 0.05 |
| WARN  | 1–5% | 50–99 | 0.05–0.1 |
| FAIL  | < 1% | < 50 | > 0.1 |

## Public Datasets

| Dataset | Modality | FC focus |
|---------|----------|---------|
| HCP MEG | MEG | Resting-state networks; canonical for source-space FC. |
| Cam-CAN MEG | MEG | Aging cohort; lifespan FC. |
| MNI Open iEEG Atlas | sEEG | Resting iEEG; baseline FC distribution by region. |
| OpenNeuro RS-EEG | EEG | Multiple resting-state datasets with curated artifact rejection. |
| IBL Mouse Brain | LFP | Multi-region LFP; mouse decision-making. |

## Pitfalls & Failure Modes

- **Choosing coherence on sensor-space EEG.**  Volume conduction
  produces near-1.0 coherence on neighbouring channels at all
  frequencies.  Switch to wPLI.
- **Reporting raw FC without surrogates.**  PLV on 30 trials of random
  noise is ~0.2 by chance.  Always surrogate-correct.
- **Time-varying FC without stationarity check.**  Sliding-window FC on
  non-stationary task data conflates stationarity violations with
  genuine FC change.  See `dynamic_connectivity.md`.
- **Symmetry assumed for directional metrics.**  Granger / DTF / PDC are
  *directional* — `FC[i, j] ≠ FC[j, i]`.  Treating them as symmetric
  drops half the information.
- **Multiple-comparison sloppiness.**  An N×N matrix has N(N-1)/2 pairs;
  at α=0.05 unc., 5% of an empty matrix is "significant" by chance.
  FDR is the bare minimum; cluster-level correction is preferred.

## Boundary with Related Paradigms

- **vs `phase_amplitude_coupling`**: PAC is *within-channel cross-band*;
  connectivity is *between-channel within-band*.  Different recipes.
- **vs `dynamic_connectivity`**: this skill is stationary FC per epoch
  or session; `dynamic_connectivity.md` (when added) covers
  sliding-window / HMM-microstate analyses.
- **vs `resting_state`**: resting state is the *condition* (eyes-open /
  eyes-closed, no task); connectivity is the *measurement*.  Combinable.

## Standalone End-to-End Pipeline

```python
"""Standalone wPLI demo on synthetic two-source data."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    b, a = butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def wpli(x: np.ndarray, y: np.ndarray) -> float:
    """Weighted Phase-Lag Index for two band-passed signals."""
    Hx = hilbert(x)
    Hy = hilbert(y)
    cross = Hx * np.conj(Hy)
    imag = cross.imag
    return float(np.abs(np.mean(np.abs(imag) * np.sign(imag))) / (np.mean(np.abs(imag)) + 1e-20))


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    sfreq = 256.0
    duration_s = 30.0
    t = np.arange(int(sfreq * duration_s)) / sfreq
    common = np.sin(2 * np.pi * 10 * t)
    a = common + rng.standard_normal(t.size) * 0.5
    b = np.roll(common, int(0.05 * sfreq)) + rng.standard_normal(t.size) * 0.5
    a_band = bandpass(a, sfreq, 8, 12)
    b_band = bandpass(b, sfreq, 8, 12)
    score = wpli(a_band, b_band)
    print(f"wPLI(8–12 Hz, lag 50 ms) = {score:.3f}")
    return score


if __name__ == "__main__":
    main()
```

Expected output: `wPLI ≈ 0.5–0.8`.

## EasyBCI Pipeline Spec

```yaml
pipeline_name: eeg-connectivity-baseline
modality: eeg
paradigm: connectivity
analysis_goal: connectivity
steps:
  - "notch:50,Q=30"
  - "bandpass:1,80"
  - "drop_bads:auto"
  - "ica:gentle,artifact=eog"
  - "laplacian_ref"
features:
  type: "wpli"
  bands_hz: [[4, 8], [8, 13], [13, 30], [30, 80]]
  n_surrogates: 200
contract:
  inject_drop_nondata: false
  allow_aggressive_notch: false
  output_shape: "(n_bands, n_channels, n_channels)"
```

## Output Channel Contract

Connectivity outputs are **N×N matrices per band per epoch**, not
per-channel feature vectors.  The mini-repo writes them under
`preprocessed_output/AI_ready/{subject}/{session}/connectivity_*.pkl`.
QC figures: per-band heatmaps + cross-frequency phase plots; saved via
`FinalDataView.connectivity_heatmap(...)` so multi-modal contracts
remain consistent.

## References

1. Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J.
   (1999). *Measuring phase synchrony in brain signals*. Human Brain
   Mapping 8: 194–208. doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C
   — PLV.
2. Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). *Phase lag index:
   assessment of functional connectivity from multi channel EEG and MEG
   with diminished bias from common sources*. Human Brain Mapping 28:
   1178–1193. doi:10.1002/hbm.20346 — PLI.
3. Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., &
   Pennartz, C. M. A. (2011). *An improved index of phase-synchronization
   for electrophysiological data in the presence of volume-conduction,
   noise and sample-size bias*. NeuroImage 55: 1548–1565.
   doi:10.1016/j.neuroimage.2011.01.055 — wPLI.
4. Bastos, A. M. & Schoffelen, J.-M. (2016). *A Tutorial Review of
   Functional Connectivity Analysis Methods and Their Interpretational
   Pitfalls*. Frontiers in Systems Neuroscience 9: 175.
   doi:10.3389/fnsys.2015.00175 — required reading.
5. Granger, C. W. J. (1969). *Investigating Causal Relations by
   Econometric Models and Cross-spectral Methods*. Econometrica 37:
   424–438. doi:10.2307/1912791 — Granger causality.
6. Baccalá, L. A. & Sameshima, K. (2001). *Partial directed coherence:
   a new concept in neural structure determination*. Biological
   Cybernetics 84: 463–474. doi:10.1007/PL00007990 — PDC.
7. Kaminski, M. & Blinowska, K. J. (1991). *A new method of the
   description of the information flow in the brain structures*.
   Biological Cybernetics 65: 203–210. doi:10.1007/BF00198091 — DTF.
