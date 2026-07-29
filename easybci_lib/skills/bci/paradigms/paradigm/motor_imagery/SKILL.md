---
name: motor_imagery
description: 'Motor Imagery BCI processing: CSP features, mu/beta bands, spatial filtering'
version: 2.0.0
layer: L2
group: paradigm
metadata:
  tags:
  - eeg
  - mi
  - motor_imagery
  - csp
  - erds
  - bci
  modalities:
  - eeg
  paradigms:
  - motor_imagery
  - mi
  analysis_goal_allowed:
  - classification
  - feature_extraction
  - exploratory
  - generic
  - source_localization
  - online_inference
  analysis_goal_forbidden:
  - clinical_screening
---
# Motor Imagery (MI)

> **Gold-standard reference template (T2 Sub-phase A.2).** All other paradigm
> skills should mirror this section structure: Neuroscience Background →
> Channel Selection → Frequency Bands of Interest → Recommended Pipeline →
> Common Artifacts → Quality Metrics → Classification / Decoding Baselines →
> Public Datasets → Pitfalls & Failure Modes → Boundary with Related
> Paradigms → Standalone End-to-End Pipeline → EasyBCI Pipeline Spec →
> References.

## Neuroscience Background

Motor imagery (MI) is the mental simulation of a movement without overt
muscle activation. The neurophysiological hallmark is **event-related
desynchronization** (ERD) of the **mu rhythm** (8–13 Hz) and **beta** (13–30
Hz) over the contralateral sensorimotor cortex during imagery, followed by
**event-related synchronization** (ERS — the "beta rebound") within ~1 s
after imagery termination.  These oscillations originate in the
thalamocortical loops of the motor system; ERD is interpreted as the
desynchronisation of competing dipolar generators when the cortex
transitions from idle to active.

Historical context: Pfurtscheller & Aranibar (1977) first quantified ERD
in EEG; the BCI-as-rehabilitation paradigm dates to Birbaumer's group
(Wolpaw et al. 2002 review) and the Berlin group's competition pipelines
(Blankertz et al. 2008).  CSP + LDA was the de-facto baseline through
~2017; deep models (EEGNet, ConvNet) have since matched but not decisively
exceeded CSP on the canonical BCI Competition IV-2a benchmark when sample
size is realistic.

## Channel Selection (electrode positions)

Working in the international 10–20 / 10–10 / 10–5 systems:

| Tier | Electrodes | Notes |
|------|-----------|-------|
| **Required** | C3, C4, Cz | The hand / foot homunculus.  Without these three, no ERD baseline can be measured. |
| **Recommended** | FC3, FC4, CP3, CP4 | Sharpen the spatial filter; CSP convergence is meaningfully better with 7+ central electrodes. |
| **Helpful** | F3, F4, P3, P4, T7, T8 | For rejecting frontal EMG / temporal blinks via ICA / regression. |
| **Optional** | full 10–20 (32-ch) or 10–10 (64-ch) | Diminishing return above 64; CSP starts to overfit on small trial counts. |

For high-density caps (128/256 ch), Laplacian re-referencing of C3/C4 to
their 4 nearest neighbors (small Laplacian) is often a stronger feature
than CSP alone.

## Frequency Bands of Interest

| Band | Range (Hz) | Physiology | Use in MI |
|------|------------|------------|-----------|
| Delta | 0.5–4 | Slow drifts, sleep | Removed (drift artefact). |
| Theta | 4–8 | Cognitive control | Removed unless studying executive load. |
| **Mu** (α-band over sensorimotor cortex) | **8–13** | Rolandic alpha; modulated by motor planning | **Primary feature.**  ERD ratio 0.4–0.7× baseline during imagery. |
| **Beta** | **13–30** | Sensorimotor binding, post-movement rebound | **Primary feature.**  ERS spike at 0.5–1 s post-stop. |
| Low gamma | 30–40 | Fine motor control | Optional — informative on iEEG but lost in scalp EEG noise. |
| EMG band | > 40 | Muscle activity | Removed (artefact). |

The "gold-standard" decoding band for binary MI is the **subject-specific
mu+beta envelope**: empirically estimate the peak-power frequency around
8–14 Hz on the resting baseline, then decode in a ±2 Hz window around it
(Blankertz et al. 2008's CSP+LDA does this implicitly via the projected
band-power; explicit subject-specific selection beats generic 8–30 Hz on
small-sample subjects).

## Recommended Pipeline

```
notch:50 → bandpass:0.5,40 → drop_bads:auto → resample:256 → ICA:eog → scale:robust
```

Step rationale:

1. **notch:50** — power-line interference.  60 Hz in North America; check
   the recording metadata.  Mu-band features are insensitive to a
   correctly-applied notch; beta features lose < 1 dB at 50 Hz.
2. **bandpass:0.5,40** — preserve mu + beta + low gamma; drop slow drift
   and broadband EMG.  Low cutoff at 0.5 (not 1.0) preserves
   movement-related cortical potentials (MRCP) for two-stage decoders
   that combine ERD with the readiness potential.
3. **drop_bads:auto** — remove flat or excessively noisy channels before
   spatial filtering; bad channels contaminate CSP projections worse than
   any other downstream step.
4. **resample:256** — standardise sampling rate.  256 Hz suffices for
   ≤ 80 Hz analysis (Nyquist = 128).  Skip if recording is already at a
   target rate.
5. **ICA:eog** — remove blinks; *only* fit ICA when the recording has
   ≥ 32 channels and ≥ 60 s of data — below that, ICA components are
   under-determined and the extracted "blink" component leaks
   sensorimotor signal.  Consider regression-based EOG removal as an
   alternative for small-channel-count datasets.
6. **scale:robust** — robust scaling (median + IQR) is more stable than
   z-score under heavy-tailed trial-level outliers (movement bursts).

For online inference, replace the FIR bandpass with the IIR variant in
``bandpass_filter`` and skip ICA (substitute ``asr`` for adaptive
artefact removal).

## Common Artifacts

| Artifact | Spectral signature | Spatial signature | Treatment |
|----------|--------------------|-------------------|-----------|
| EMG (jaw clench, facial muscles) | Broadband > 20 Hz; spectral edge shifts > 30 Hz | Highest at temporal / lateral channels | Tighten bandpass to ≤ 30 Hz; reject trials with edge frequency > 30 Hz. |
| Eye blinks (EOG) | Below 8 Hz; large-amplitude transients | Frontopolar (Fp1, Fp2) | ICA removal targeting eyeblink IC; regression on FP1/FP2 if no ICA. |
| Heartbeat (ECG) | 1–2 Hz periodic | Same on all channels | Often acceptable in MI; ICA removes if needed. |
| Saccades / lateral eye movement | Below 30 Hz, abrupt | Frontotemporal | Same as blink, plus check imagined hand might co-occur. |
| Movement of the cap | Low-frequency drift; correlated across nearby channels | Local | Trial rejection; bandpass usually catches it. |

## Quality Metrics

Per-trial:

- **ERD percent**: ``(P_baseline - P_imagery) / P_baseline × 100`` in the
  subject-specific mu+beta band over the contralateral electrode (C3 for
  right-hand, C4 for left-hand).  Healthy: 30–70%.
- **Cross-trial coherence**: same-condition trials should agree to within
  ± 10% on per-channel ERD; high variance suggests inattention.
- **Spectral edge frequency** (95th-percentile spectral edge in the
  imagery window): > 30 Hz → likely EMG contamination.

Per-session:

- **Resting alpha peak height** (8–13 Hz peak on Cz at rest): < 1 SD
  above baseline → poor subject; expect chance-level decoding.
- **Trial count per class**: ≥ 20 for CSP+LDA; ≥ 60 for deep models.

Recommended grade thresholds (encoded in `quality/grader.py`):

| Grade | Mu peak (dB above baseline noise) | ERD% (mean across trials) | Trial count / class |
|-------|----------------------------------|--------------------------|---------------------|
| PASS  | ≥ 6 | ≥ 25% | ≥ 20 |
| WARN  | 3–6 | 15–25% | 10–19 |
| FAIL  | < 3 | < 15% | < 10 |

## Classification / Decoding Baselines

| Method | Typical accuracy on BCI Comp IV-2a (4-class) | Notes |
|--------|---------------------------------------------|-------|
| **CSP + LDA** | 65–75% | Two-class CSP per pair (one-vs-rest), filter-bank optional. The classical baseline. |
| **FBCSP (filter-bank CSP) + LDA** | 70–80% | Multi-band CSP, mutual-info feature ranking. Ang et al. 2008. |
| **Riemannian + MDM** | 70–80% | Covariance manifold geodesic; calibration-free transfer. Barachant et al. 2012. |
| **EEGNet** | 70–80% | Compact CNN; strong on within-subject. Lawhern et al. 2018. |
| **ShallowConvNet / DeepConvNet** | 70–82% | Schirrmeister et al. 2017. Larger models, more data needed. |
| **EEG-Conformer / transformer baselines** | 75–85% | 2022–2024 papers; gain over EEGNet is typically < 5 pp on small datasets. |

For two-class motor imagery (left vs right hand), 80–90% is achievable
with FBCSP on well-curated subjects.  Generic "let me classify any new
subject" pipelines should expect 70–80% as the realistic ceiling.

## Public Datasets

| Dataset | Subjects | Channels | Rate | Classes | Link / DOI |
|---------|----------|----------|------|---------|-----------|
| BCI Competition IV-2a | 9 | 22 EEG + 3 EOG | 250 Hz | left hand, right hand, foot, tongue | doi:10.1109/TBME.2007.910646 |
| BCI Competition IV-2b | 9 | 3 EEG (C3, Cz, C4) + 3 EOG | 250 Hz | left vs right hand | doi:10.1109/TBME.2007.910646 |
| PhysioNet EEGMMI | 109 | 64 | 160 Hz | left fist, right fist, fists, feet | https://physionet.org/content/eegmmidb/ |
| HighGamma (Schirrmeister 2017) | 14 | 128 | 500 Hz | hand-l, hand-r, feet, rest | https://gin.g-node.org/robintibor/high-gamma-dataset |
| OpenBMI MI | 54 | 62 | 1000 Hz | left vs right hand | doi:10.1093/gigascience/giz002 |

## Pitfalls & Failure Modes

- **Forgetting baseline correction.** Per-trial ERD% requires the pre-cue
  reference window — without it, the ERD signal is buried in
  inter-trial baseline drift.
- **Cross-subject CSP transfer.** CSP filters do not transfer across
  subjects without alignment (Riemannian alignment, EA, or covariance
  whitening).  Pre-trained CSP on a different subject ~ chance.
- **Blink leakage into IC1.** When the data is too short (< 60 s),
  FastICA's most-variant component conflates blinks with
  sensorimotor activity; rejecting "the blink IC" then takes mu/beta
  down with it.
- **Bilateral ERD.** Normal for foot imagery; for hand imagery, suggests
  poor lateralization or subject fatigue.  Re-train rather than reject
  outright.
- **EMG contamination disguised as gamma.** Broadband > 30 Hz power
  increase during imagery is almost always jaw / neck EMG; cap the
  upper bandpass at 30 Hz when uncertain.
- **Class imbalance from trial rejection.** Aggressive artefact rejection
  often biases toward one class (subjects clench more during a difficult
  condition).  Re-balance or stratify before decoding.

## Boundary with Related Paradigms

- **vs `phase_amplitude_coupling`**: PAC is *cross-frequency, single-trial*
  coupling within an electrode; MI's ERD is *single-frequency,
  cross-trial* power modulation across electrodes.  When the analysis
  asks "does theta phase modulate beta amplitude during imagery?", the
  answer needs PAC pipeline (notch → narrow bandpass → no ICA → Hilbert
  → MI calculation) — not the MI pipeline above.
- **vs `connectivity`**: connectivity is *between-channel* over a fixed
  band; MI ERD is *single-channel* over the imagery window.  When the
  question becomes "does C3–C4 coherence change with imagery?", swap to
  the `connectivity` skill.
- **vs `online_inference`**: same paradigm, different latency budget.
  Online MI requires causal filtering, ASR-style adaptive cleaning, and
  incremental classifiers (online MDM).  The pipeline above is the
  offline analysis; for online, see `online_inference.md`.
- **vs `stroke_rehab_bci`**: clinical MI with lesion-induced asymmetry —
  ipsilesional ERD is often weaker than contralesional ERS, requiring
  modified spatial filters (MUR-CSP, ipsilesional ROI).

## Standalone End-to-End Pipeline

```python
"""Self-contained MI pipeline — synthetic data, but the same shape as a
real subject. Runs without EasyBCI in any python>=3.10 env with
mne/scikit-learn/scipy installed."""

from __future__ import annotations
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP


def make_synthetic_mi(seed: int = 0, n_trials: int = 60, sfreq: float = 256.0):
    """Two-class MI: amplified mu power on left or right central electrode."""
    rng = np.random.default_rng(seed)
    ch_names = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
    n_ch = len(ch_names)
    duration_s = 4.0
    n_t = int(sfreq * duration_s)
    X = []
    y = []
    t = np.arange(n_t) / sfreq
    for trial in range(n_trials):
        cls = rng.integers(0, 2)
        # baseline pink-ish noise
        x = rng.standard_normal((n_ch, n_t)) * 5e-6
        # 10 Hz mu suppression: more on contralateral channel
        target = 3 if cls == 1 else 5  # C3 for right-hand (cls=1), C4 for left
        x[target] += np.sin(2 * np.pi * 10 * t) * 1e-5  # boost rest period
        x[target, n_t // 2:] *= 0.3  # ERD during imagery half
        X.append(x); y.append(cls)
    return np.stack(X), np.array(y), ch_names, sfreq


def main():
    X, y, ch_names, sfreq = make_synthetic_mi()
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, verbose="ERROR")
    # Bandpass within mu+beta range
    epochs.filter(8, 30, verbose="ERROR")

    csp = CSP(n_components=4, reg="ledoit_wolf", log=True)
    clf = Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
    scores = cross_val_score(clf, epochs.get_data(), y, cv=5, scoring="accuracy")
    print(f"CSP+LDA 5-fold accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


if __name__ == "__main__":
    main()
```

Expected output on the synthetic data: `0.65–0.85` accuracy.

## EasyBCI Pipeline Spec

YAML form usable in `pipeline_record.json`:

```yaml
pipeline_name: eeg-motor_imagery-baseline
modality: eeg
paradigm: motor_imagery
analysis_goal: classification
steps:
  - "notch:50"
  - "bandpass:0.5,40"
  - "drop_bads:auto"
  - "resample:256"
  - "scale:robust"
segment:
  method: "event"
  pre_event_s: 1.0
  post_event_s: 5.0
  baseline_window_s: [-1.0, 0.0]
features:
  type: "csp+log_bandpower"
  n_components: 4
  bands_hz: [[8, 30]]
classifier:
  type: "lda"
  shrinkage: "auto"
```

LLM rendering: when the user says "preprocess this MI EEG with subject A",
the LLM should produce *exactly* the steps + segmentation contract above
(modulo subject-specific cohort tag) — the operator selection is the
recommendation, the segmentation is the contract.

## References

1. Pfurtscheller, G., & Aranibar, A. (1977). *Event-related cortical
   desynchronization detected by power measurements of scalp EEG*.
   Electroencephalography and Clinical Neurophysiology 42: 817–826. —
   first quantitative ERD measurement.
2. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Muller, K.-R.
   (2008). *Optimizing spatial filters for robust EEG single-trial
   analysis*. IEEE Signal Processing Magazine 25: 41–56.
   doi:10.1109/MSP.2008.4408441 — the canonical CSP reference.
3. Ang, K. K., Chin, Z. Y., Wang, C., Guan, C., & Zhang, H. (2008).
   *Filter Bank Common Spatial Pattern algorithm on BCI Competition IV
   Datasets 2a and 2b*. Frontiers in Neuroscience 6: 39.
   doi:10.3389/fnins.2012.00039 — FBCSP.
4. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
   C. P., & Lance, B. J. (2018). *EEGNet: a compact convolutional
   network for EEG-based brain–computer interfaces*. Journal of Neural
   Engineering 15: 056013. doi:10.1088/1741-2552/aace8c.
5. Schirrmeister, R. T. et al. (2017). *Deep learning with convolutional
   neural networks for EEG decoding and visualization*. Human Brain
   Mapping 38(11): 5391–5420. doi:10.1002/hbm.23730.
6. Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012).
   *Multiclass brain–computer interface classification by Riemannian
   geometry*. IEEE Trans. Biomed. Eng. 59(4): 920–928.
   doi:10.1109/TBME.2011.2172210.
7. Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., &
   Vaughan, T. M. (2002). *Brain–computer interfaces for communication
   and control*. Clinical Neurophysiology 113: 767–791.
   doi:10.1016/s1388-2457(02)00057-3 — the BCI-as-rehabilitation
   foundation.
