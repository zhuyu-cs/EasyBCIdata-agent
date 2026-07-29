---
name: phase_amplitude_coupling
description: 'Phase-amplitude coupling (PAC): low-frequency phase modulating high-frequency
  amplitude'
version: 1.0.0
layer: L2
group: analysis
metadata:
  tags:
  - pac
  - cfc
  - cross_frequency_coupling
  - mi
  - hilbert
  - theta_gamma
  modalities:
  - eeg
  - seeg
  - ecog
  - lfp
  paradigms:
  - phase_amplitude_coupling
  - cfc
  - theta_gamma
  analysis_goal_allowed:
  - phase_amplitude_coupling
  - exploratory
  - feature_extraction
  analysis_goal_forbidden:
  - classification
  - online_inference
---
# Phase-Amplitude Coupling (PAC)

## Neuroscience Background

Phase-amplitude coupling is the canonical case of **cross-frequency
coupling (CFC)**: the *amplitude* of a faster oscillation is non-uniformly
distributed across the *phase* of a slower oscillation.  The most
established example is **theta–gamma coupling** in the rodent hippocampus
(Bragin et al. 1995) — gamma bursts ride the trough of the theta cycle,
gating information transfer to neocortex.  Cortical alpha–gamma coupling
in human EEG (Canolty et al. 2006) extended the same idea to scalp
recordings, and PAC has since been reported in working memory (theta–
gamma over PFC), seizure dynamics (delta–HFO in iEEG), motor planning
(beta–gamma in M1), and consciousness research (delta–alpha in
anaesthesia).

The mechanistic interpretation: the slow oscillation reflects local
inhibition / excitation cycles (PING / ING circuits) that gate windows
of high-frequency activity.  The behavioural readout is **modulation
strength**, not absolute high-frequency power.

## Channel Selection (electrode positions)

PAC is typically a **single-channel within-region** measurement; the
phase and amplitude come from the same site.  This contrasts with
inter-channel **phase-locking** (`connectivity` skill).

| Modality | Recommended placement |
|----------|----------------------|
| Scalp EEG | Avoid frontopolar (Fp1, Fp2 — eye-blink contamination at the slow frequency).  Cz, Pz, O1/O2 are clean. |
| iEEG (sEEG / ECoG) | Within-electrode bipolar pair; references picked to maximise local field signal-to-noise. |
| LFP | Single contact; CSD-derived if depth electrode array. |

For task-driven PAC, pick the channel from the prior literature for the
behaviour: e.g., Cz / FCz for working-memory theta–gamma; M1 contact for
motor beta–gamma.

## Frequency Bands of Interest

Three canonical pairs (the "phase frequency" and "amplitude frequency"):

| Phase band | Amplitude band | Phenomenon |
|------------|----------------|-----------|
| Theta (4–8 Hz) | Gamma (30–80 Hz) | Hippocampal / PFC working memory; cortical attention. |
| Alpha (8–13 Hz) | Gamma (30–80 Hz) | Sensorimotor gating; visual cortex modulation. |
| Delta (1–4 Hz) | High-gamma (80–150 Hz) | iEEG seizure dynamics; deep-sleep spindles. |
| Beta (13–30 Hz) | High-gamma (70–200 Hz) | Motor planning; basal ganglia oscillations. |

The **bandwidth** of each band matters: phase bands should be narrow
(2–4 Hz) so the Hilbert phase is meaningful; amplitude bands can be
wider (10–30 Hz) since we average power.

## Recommended Pipeline

```
notch:50 (NARROW) → bandpass:1,200 (PRESERVE WIDE) → drop_bads → epoch → hilbert(phase) + hilbert(amplitude) → modulation_index
```

**No ICA.**  ICA decomposes the signal into linear independent components,
destroying the cross-frequency phase relationships PAC measures.  This
is the most common failure mode in the literature.

**No aggressive notch.**  A wide notch (e.g., > 4 Hz around 50 Hz)
distorts the phase across the entire spectrum on either side.  Use the
narrowest possible notch (`Q ≥ 30`).

**No re-referencing change between filtering and Hilbert.**  Re-reference
once at the start; then leave it alone.

## Common Artifacts

- **Edge effects in Hilbert.**  The first / last (`1 / f_low`) seconds of
  the recording show meaningless instantaneous phase.  Pad with at
  least one full slow-oscillation period and trim post-Hilbert.
- **Spurious PAC from sharp transients.**  A spike (epileptic discharge,
  blink) injects energy across all frequencies — its phase aligns
  trivially with its own amplitude, producing inflated MI.  Reject
  trials with z-score amplitude > 5 in the slow band before computing
  PAC.
- **Volume conduction in EEG.**  Two scalp electrodes pick up the same
  cortical generator; their PAC is identical.  Use Laplacian or CSD
  reference to localise.

## Quality Metrics

- **Modulation Index (MI; Tort et al. 2010)**: Kullback-Leibler divergence
  between the observed amplitude-by-phase distribution and a uniform
  distribution.  Values 0 (no coupling) to log(N_bins).  Healthy
  task-PAC: MI ≥ 0.005 with N_bins = 18.
- **Mean Vector Length (MVL; Canolty 2006)**: magnitude of complex
  ``mean(amplitude · exp(i·phase))``.  Normalise by ``mean(amplitude)``
  for the **normalised MVL** in [0, 1].
- **Surrogate p-value**: shuffle the phase time-series 200 times,
  compute MI on each, count proportion ≥ observed.  Required —
  unshuffled MI is heavily biased upward by short data lengths.

Recommended grade thresholds:

| Grade | MI vs surrogate | Min trials | Max edge artefact ratio |
|-------|-----------------|-----------|-------------------------|
| PASS  | p < 0.01 | ≥ 100 | < 5% |
| WARN  | 0.01 ≤ p < 0.05 | 50–100 | 5–10% |
| FAIL  | p ≥ 0.05 | < 50 | > 10% |

## Classification / Decoding Baselines

PAC is **typically a feature-extraction step**, not a classifier.  Common
combinations:

- **MI + LDA**: per-trial MI in M phase × N amplitude bins fed to LDA.
- **PAC matrices (M × N comodulogram) + CNN**: treat the comodulogram as
  an image; works well for seizure-detection.
- **PAC + Riemannian**: build covariance on PAC features; manifold
  classifiers (MDM) are robust.

Typical MI-decoding accuracy on theta-γ working memory paradigms: 65–
75% binary (encode vs maintain).

## Public Datasets

| Dataset | Modality | Notes | Link / DOI |
|---------|----------|-------|-----------|
| HCP Working Memory MEG | MEG | n-back task; theta-gamma PAC over PFC. | https://www.humanconnectome.org/ |
| Stanford XNAT iEEG | sEEG | Seizure-zone vs healthy contacts; delta-HFO PAC. | doi:10.1126/science.1149639 (Canolty et al. 2006) |
| MNI Open iEEG Atlas | sEEG | Resting state intracranial; baseline PAC distribution. | doi:10.1038/s41597-019-0036-3 |
| OpenNeuro PAC Working Memory | EEG | Public theta-γ task. | https://openneuro.org/datasets/ds000117 |

## Pitfalls & Failure Modes

- **PAC on broadband data (no bandpass) → noise.**  The whole point of
  the pipeline is the narrow phase / amplitude bands.
- **PAC on filtered → ICA → filtered data.**  ICA destroys the
  cross-frequency phase.  Hard rule: never apply ICA to a recording
  intended for PAC analysis.
- **Skipping surrogate testing.**  Unshuffled MI on 30 s of data is
  ~0.01 by chance.  Always run ≥ 200 shuffle surrogates.
- **Phase-amplitude band overlap.**  If `phase_band ∩ amplitude_band
  ≠ ∅` (e.g., 6–10 Hz vs 8–14 Hz), the bandpass filters share spectral
  support and the Hilbert envelope of one bleeds into the phase of the
  other.
- **Sample-size dependence of MI.**  MI is biased upward at short epoch
  lengths.  Bin counts matter (Tort recommends 18); short data should
  use the surrogate-normalised z-MI rather than raw MI.

## Boundary with Related Paradigms

- **vs `connectivity`**: connectivity is *between-channel, same-band*;
  PAC is *within-channel, cross-band*.  When the question is "do C3 and
  C4 share alpha rhythm?" → connectivity (PLV / coherence).  When the
  question is "does C3's theta phase modulate C3's gamma amplitude?" →
  PAC.
- **vs `motor_imagery`**: MI's ERD is single-band power modulation;
  PAC asks about cross-band gating.  Combinable: ERD-conditional PAC,
  comparing theta-γ MI in imagery vs rest windows.
- **vs `seeg_epilepsy`**: epileptic tissue often shows pathological
  delta-HFO PAC.  When the analysis goal is biomarker mapping rather
  than oscillation characterisation, the seeg_epilepsy skill provides
  the clinical contract; PAC is the underlying measurement.

## Standalone End-to-End Pipeline

```python
"""Standalone PAC demo on synthetic theta-gamma data."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    b, a = butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="band")
    return filtfilt(b, a, x)


def modulation_index(phase_signal: np.ndarray, amp_signal: np.ndarray, n_bins: int = 18) -> float:
    """Tort 2010 MI: KL divergence of phase-binned amplitude distribution
    from uniform.  Returns MI ∈ [0, log(n_bins)]."""
    phase = np.angle(hilbert(phase_signal))
    amp = np.abs(hilbert(amp_signal))
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    indices = np.digitize(phase, bin_edges) - 1
    p = np.array([amp[indices == i].mean() if (indices == i).any() else 0 for i in range(n_bins)])
    p = p / (p.sum() + 1e-20)
    p = np.clip(p, 1e-12, None)
    uniform = 1.0 / n_bins
    kl = np.sum(p * np.log(p / uniform))
    return float(kl)


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    sfreq = 1000.0
    duration_s = 30.0
    t = np.arange(int(sfreq * duration_s)) / sfreq
    # Synthetic: theta phase modulates gamma amplitude
    theta = np.sin(2 * np.pi * 6 * t)
    gamma_envelope = (1 + 0.6 * np.cos(2 * np.pi * 6 * t - np.pi)) / 2
    gamma = gamma_envelope * np.sin(2 * np.pi * 60 * t)
    noise = rng.standard_normal(t.size) * 0.4
    x = theta + 0.5 * gamma + noise

    # Pipeline: narrow bandpass → Hilbert → MI
    phase_band = bandpass(x, sfreq, 4, 8)
    amp_band = bandpass(x, sfreq, 30, 80)
    mi = modulation_index(phase_band, amp_band)

    # Surrogate p-value
    surrogates = []
    for _ in range(200):
        shift = rng.integers(int(sfreq), x.size - int(sfreq))
        phase_shuffled = np.roll(phase_band, shift)
        surrogates.append(modulation_index(phase_shuffled, amp_band))
    p_value = (np.array(surrogates) >= mi).mean()
    print(f"MI = {mi:.4f}, p (200 surrogates) = {p_value:.4f}")
    return mi, p_value


if __name__ == "__main__":
    main()
```

Expected output: `MI ≈ 0.05`, `p < 0.01`.

## EasyBCI Pipeline Spec

```yaml
pipeline_name: eeg-pac-baseline
modality: eeg
paradigm: phase_amplitude_coupling
analysis_goal: phase_amplitude_coupling
steps:
  - "notch:50,Q=30"
  - "bandpass:1,200"
  - "drop_bads:auto"
features:
  type: "modulation_index"
  phase_band_hz: [4, 8]
  amp_band_hz: [30, 80]
  n_bins: 18
  surrogate_n: 200
# IMPORTANT: REGISTRY[phase_amplitude_coupling] sets allow_ica=False and
# allow_aggressive_notch=False — the pipeline_check rejects steps that
# violate these.
```

## References

1. Canolty, R. T. et al. (2006). *High Gamma Power Is Phase-Locked to
   Theta Oscillations in Human Neocortex*. Science 313: 1626–1628.
   doi:10.1126/science.1128115 — first cortical PAC.
2. Tort, A. B. L., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010).
   *Measuring phase-amplitude coupling between neuronal oscillations of
   different frequencies*. Journal of Neurophysiology 104: 1195–1210.
   doi:10.1152/jn.00106.2010 — Modulation Index.
3. Aru, J. et al. (2015). *Untangling cross-frequency coupling in
   neuroscience*. Current Opinion in Neurobiology 31: 51–61.
   doi:10.1016/j.conb.2014.08.002 — caveats and pitfalls; required
   reading before publishing PAC results.
4. Bragin, A. et al. (1995). *Gamma (40–100 Hz) oscillation in the
   hippocampus of the behaving rat*. Journal of Neuroscience 15:
   47–60. doi:10.1523/JNEUROSCI.15-01-00047.1995.
5. Hülsemann, M. J., Naumann, E., & Rasch, B. (2019). *Quantification of
   phase-amplitude coupling in neuronal oscillations: comparison of
   phase-locking value, mean vector length, modulation index, and
   generalized-linear-modeling-cross-frequency-coupling*. Frontiers in
   Neuroscience 13: 573. doi:10.3389/fnins.2019.00573 — comparison of
   metrics.
