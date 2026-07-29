---
name: online_inference
description: 'Real-time BCI inference: causal filtering, ASR, incremental classifiers'
version: 1.0.0
layer: L2
group: online
metadata:
  tags:
  - online
  - realtime
  - bci
  - asr
  - incremental
  - causal
  modalities:
  - eeg
  - ecog
  - lfp
  paradigms:
  - online_inference
  - realtime_bci
  - closed_loop
  analysis_goal_allowed:
  - online_inference
  analysis_goal_forbidden:
  - source_localization
  - phase_amplitude_coupling
---
# Online Inference

> **Latency budget governs everything in this skill.**  Offline pipelines
> can spend a minute on ICA; online pipelines have ~50 ms per window.
> Many operators that are routine offline (FIR linear-phase bandpass,
> ICA, zero-phase filtering) are **forbidden** here because they look
> ahead in time.

## Neuroscience Background

Online BCI is the closed-loop variant of any offline paradigm: features
are extracted on a *sliding* window, classified on-the-fly, and the
decision is fed back to the user / actuator.  The signal physiology
(mu/beta for MI, P300 for ERP, SSVEP target frequencies) is unchanged
from the offline analysis — only the engineering envelope shifts.

Latency budget components (typical 100–250 ms total):

| Stage | Budget (ms) | Notes |
|-------|-------------|-------|
| ADC + transport (amplifier → host) | 5–30 | LSL adds ~5 ms; USB-1 ≥ 8 ms; Bluetooth ≥ 30. |
| Buffer + window framing | 10–20 | Sliding window step ≥ 1 sample; usually 50–100 ms. |
| Causal filtering | 5–15 | IIR Butterworth or causal FIR; **no zero-phase**. |
| Adaptive cleaning (ASR) | 10–50 | Only when needed; bounded-window. |
| Feature extraction (CSP, log-bandpower) | 5–10 | Pre-computed projection matrices. |
| Classifier inference | 1–5 | Online MDM / online LDA; pre-fit. |
| Decision smoothing + actuation | 5–20 | Hysteresis to avoid jittery output. |

## Channel Selection

Same as the offline paradigm being run.  The only online-specific guidance:

- **Re-reference once at session start.**  Re-referencing inside the
  per-window pipeline introduces flicker artefacts.
- **Use a fixed channel set.**  Adding / removing channels per window
  invalidates the pre-fit CSP / classifier.

## Frequency Bands of Interest

Same bands as the offline paradigm. Two online-specific notes:

- **Causal filtering shifts phase.**  ERP-style paradigms (P300) are
  more vulnerable than power-based ones (MI, SSVEP).
- **Heavy bandpass narrows usable info.**  Below 8 Hz with a 4th-order
  IIR causal, the group delay is comparable to the window length;
  prefer first-order or skip the high-pass for stationary subjects.

## Recommended Pipeline

```
buffer:50ms_step → bandpass_iir:1,40 → asr → log_bandpower → online_mdm
```

Step rationale:

1. **Sliding buffer.**  Typical: 1.0 s window, 50 ms step (20 Hz update).
   Set window length such that the slowest informative oscillation has
   ≥ 4 cycles in-window.
2. **Causal IIR bandpass.**  4th-order Butterworth (`bandpass_filter`
   with `method="iir"`).  Forbidden: FIR linear-phase, zero-phase
   `filtfilt`.
3. **ASR (Artifact Subspace Reconstruction).**  Mullen et al. 2015 /
   Kothe-Makeig.  Identifies bursts that exceed `k=20` standard
   deviations of a calibration period and reconstructs them from clean
   subspace.  Bounded-window: typical 0.5 s overlap with 1 s window.
4. **No ICA.**  Online ICA (incremental ICA, RICA) exists but is
   under-determined on short windows and trades cleaning for stability.
   **REGISTRY enforces `allow_ica=False` for `online_inference`.**
5. **Pre-fit feature extractor.**  CSP / log-bandpower / Riemannian
   covariance projection matrices are fit on the calibration phase and
   frozen.
6. **Incremental classifier.**  Online MDM (Riemannian centroid update),
   online LDA (Welford-style mean / covariance), online EEGNet
   (mini-batch SGD with low learning rate).

## Common Artifacts

Same as offline + online-specific:

- **Window-boundary discontinuities.**  Per-window detrending creates
  sharp edges between windows.  Use overlap-add or skip detrending
  (rely on the high-pass).
- **Latency drift.**  Network jitter on streaming amplifiers can offset
  successive windows by 5–10 ms; small enough that simple causal
  filtering masks it but accumulates over a session.
- **Subject micro-movements.**  Constantly active EMG; ASR is the
  primary defence.

## Quality Metrics

Online quality is measured continuously, not per-trial:

- **Decision latency** (window start to actuator command): 50th and
  95th percentile.  Target: < 250 ms median, < 500 ms p95.
- **Decision rate**: decisions per second.  Should match the configured
  step size; lower means windows are being dropped.
- **Confidence stability**: rolling variance of classifier confidence.
  High variance → adapt classifier or surface the unstable state to the
  user.
- **ASR rejection rate**: fraction of samples reconstructed.
  > 30% → recalibrate (subject moved out of calibration distribution).

Recommended grade thresholds:

| Grade | p95 latency | Decision rate | Cumulative ASR rejection |
|-------|-------------|---------------|--------------------------|
| PASS  | < 500 ms | ≥ 90% target | < 20% |
| WARN  | 500–1000 ms | 70–90% | 20–40% |
| FAIL  | > 1000 ms | < 70% | > 40% |

## Classification / Decoding Baselines

| Method | Typical accuracy | Notes |
|--------|-----------------|-------|
| **Online MDM** (Riemannian) | 70–80% on MI | Calibration-light; centroid update is O(k) per window. |
| **Online CSP + LDA** | 65–80% on MI | Pre-fit CSP; sliding LDA mean / cov via Welford. |
| **Online EEGNet (low-lr SGD)** | 70–82% on MI | More robust than offline EEGNet to subject drift, with proper warm-up. |
| **Step detection (P300 speller)** | 75–95% on letter level | xDAWN pre-fit; threshold-based on accumulated evidence. |
| **CCA / FBCCA (SSVEP)** | 80–95% on 12-target | Pre-computed reference signals; ~100 ms decision. |

## Public Datasets

| Dataset | Task | Latency-relevant |
|---------|------|------------------|
| BCI Competition IV-2b | MI online | LSL-formatted online streaming files |
| Tsinghua SSVEP Benchmark | SSVEP | 12-class; canonical online benchmark |
| OpenBMI online MI | MI | Recorded with online stack metadata |
| BNCI Horizon 2020 | MI / P300 | Multiple online datasets |

## Pitfalls & Failure Modes

- **Using zero-phase filtering offline → online.**  Pipelines tested
  with `filtfilt` look great offline and break online (300 ms group
  delay).  Tag every filter `causal: true` in the pipeline_record.
- **Re-fitting CSP per-window.**  Each window has too few samples;
  CSP projection matrices oscillate and the classifier sees garbage.
  Pre-fit CSP on calibration; freeze.
- **Classifier overfitting to calibration session.**  Two sessions
  apart, the subject drifts.  Online incremental updates or
  Riemannian alignment are needed for cross-session deployment.
- **No safety guard.**  A motorised actuator following classifier output
  must hard-stop on `confidence < threshold` or `ASR rejection > 50%`.
  Encode the guard explicitly; do not rely on classifier accuracy
  alone.
- **Skipping the calibration phase.**  Online inference without ≥ 5
  minutes of calibration data is *guessing*.  Make the calibration
  duration explicit in the pipeline contract.

## Boundary with Related Paradigms

- **vs `motor_imagery` / `p300_erp` / `ssvep` / etc.**: Same physiology,
  different latency budget.  The paradigm skill defines what to look
  for; this skill defines how to look for it under real-time
  constraints.  Pair them: `motor_imagery` for the band selection,
  `online_inference` for the engineering.
- **vs `closed_loop_bci`**: closed-loop is the system that *also
  stimulates* in response to the inference; online_inference is just
  the read-out.  Closed-loop has additional safety + latency budget on
  the stimulation arm.
- **vs `phase_amplitude_coupling`**: PAC requires zero-phase filtering
  and Hilbert; cannot be computed online with the standard recipe.
  Online PAC is a research topic (causal Hilbert approximations,
  Carlqvist 2005), not a stable production pattern.

## Standalone End-to-End Pipeline

```python
"""Standalone online-style MI demo with sliding-window inference."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def causal_iir_bandpass_init(low: float, high: float, sfreq: float, order: int = 4):
    sos = butter(order, [low / (sfreq / 2), high / (sfreq / 2)], btype="band", output="sos")
    zi = sosfilt_zi(sos)
    return sos, zi


def stream_classify(
    samples: np.ndarray,        # (n_channels, n_total) — entire session
    sfreq: float,
    window_s: float = 1.0,
    step_s: float = 0.05,
):
    """Yield (decision_index, label) tuples — fully causal."""
    n_ch, n_t = samples.shape
    win = int(window_s * sfreq)
    step = int(step_s * sfreq)

    sos, zi = causal_iir_bandpass_init(8, 30, sfreq)
    zi = np.tile(zi[..., None], n_ch).T  # per-channel state

    # Toy classifier: log-bandpower difference between two channels (C3 vs C4).
    def predict(window: np.ndarray) -> int:
        bp_c3 = np.log(np.var(window[0]) + 1e-12)
        bp_c4 = np.log(np.var(window[1]) + 1e-12)
        return 0 if bp_c3 < bp_c4 else 1  # left-hand vs right-hand

    buffer = np.zeros((n_ch, win), dtype=samples.dtype)
    pos = 0
    for chunk_end in range(step, n_t + 1, step):
        chunk_start = max(0, chunk_end - step)
        chunk = samples[:, chunk_start:chunk_end]
        # Causal filter the chunk only — no look-ahead
        filtered, zi = sosfilt(sos, chunk, axis=1, zi=zi)
        # Slide buffer: drop oldest step, append new
        if filtered.shape[1] >= win:
            buffer = filtered[:, -win:]
        else:
            buffer = np.concatenate([buffer[:, filtered.shape[1]:], filtered], axis=1)
        if chunk_end >= win:
            yield chunk_end / sfreq, predict(buffer)


def main():
    rng = np.random.default_rng(0)
    sfreq = 256.0
    duration_s = 10.0
    t = np.arange(int(sfreq * duration_s)) / sfreq
    n_ch = 3
    samples = rng.standard_normal((n_ch, t.size)) * 1e-5
    # Inject right-hand MI: drop C3 power 0.4× during 4–6 s
    samples[0, int(4 * sfreq):int(6 * sfreq)] *= 0.4

    decisions = list(stream_classify(samples, sfreq))
    print(f"{len(decisions)} decisions over {duration_s:.0f}s")
    print(f"first 5: {decisions[:5]}")
    print(f"around 4–6s (right-hand expected): "
          f"{[d for d in decisions if 4.5 <= d[0] <= 5.5][:5]}")
    return decisions


if __name__ == "__main__":
    main()
```

Expected output: ≥ 100 decisions; right-hand window decisions skew
towards label 1.

## EasyBCI Pipeline Spec

```yaml
pipeline_name: eeg-online-mi
modality: eeg
paradigm: motor_imagery
analysis_goal: online_inference
steps:
  - "bandpass:8,30:method=iir,order=4"
  - "asr:k=20,window_s=0.5"
  - "log_bandpower:bands=[[8,30]]"
features:
  type: "online_mdm"
  pre_fit_calibration_min: 5
  decision_window_s: 1.0
  decision_step_s: 0.05
contract:
  produces_figures: false  # REGISTRY[online_inference].produces_figures = False
  causal_only: true
  allow_ica: false           # REGISTRY[online_inference].allow_ica = False
safety:
  min_confidence: 0.6
  max_asr_rejection_ratio: 0.5
```

## References

1. Müller-Putz, G. R. et al. (2015). *Towards noninvasive hybrid
   brain–computer interfaces: framework, practice, clinical
   application, and beyond*. Proceedings of the IEEE 103: 926–943.
   doi:10.1109/JPROC.2015.2411333.
2. Mullen, T. R. et al. (2015). *Real-time neuroimaging and cognitive
   monitoring using wearable dry EEG*. IEEE Transactions on Biomedical
   Engineering 62: 2553–2567.
   doi:10.1109/TBME.2015.2481482 — ASR.
3. Kothe, C. A. & Makeig, S. (2013). *BCILAB: a platform for
   brain–computer interface development*. Journal of Neural Engineering
   10: 056014. doi:10.1088/1741-2560/10/5/056014.
4. Barachant, A. & Bonnet, S. (2011). *Channel selection procedure
   using Riemannian distance for BCI applications*. NER 2011.
   doi:10.1109/NER.2011.5910558 — online MDM design.
5. Carlqvist, H. et al. (2005). *Amplitude and phase relationship
   between alpha and beta oscillations in the human
   electroencephalogram*. Medical & Biological Engineering & Computing
   43: 599–607. doi:10.1007/BF02351034 — causal Hilbert
   approximation note.

## Strengthened Pipeline (Phase 3-4 expansion)

### Why ASR replaces ICA for online use

ICA is offline-only — it needs the full recording for component
separation, and the separation matrix is not stable across short
windows. For online inference, **ASR** (Kothe & Makeig 2013) is the
canonical replacement: it learns its calibration matrix once at the
session start, then applies a streaming projection to every incoming
window. The ASR `k` parameter is typically 20 for online (less aggressive
than offline default).

```
load → bandpass:1,40,method=iir,causal=True → asr:20 → online_classifier
```

### Causal IIR filters

Offline pipelines use FIR + filtfilt (zero-phase). Online cannot —
filtfilt requires both directions. Use:

| Operation | Offline | Online |
|---|---|---|
| Bandpass | FIR / filtfilt | 4th-order Butterworth IIR via lfilter (causal) |
| Notch | FIR / filtfilt | 2nd-order IIR notch |
| Hilbert | Standard | Causal-truncated kernel; tolerate 5 ms group delay |

### Incremental classifier choices

Three viable families:

| Classifier | Update cost | Memory | Notes |
|---|---|---|---|
| **Riemannian MDM** | O(n_ch²) per update | O(n_classes · n_ch²) | Best default; robust to drift; few trials. |
| **Online CSP** | O(n_ch² · n_trials_window) | O(n_ch² · n_components) | Sliding-window cov update. |
| **Online EEGNet** | O(n_params) backprop | O(n_params) | Higher capacity; needs more trials; risk of overfit on fast loops. |

For chronic clinical BCI (BrainGate-class), the workhorse is Riemannian
MDM with weekly recalibration; new-paper deep models are reserved for
research.

### Latency Budget (per inference cycle)

```
Acquire window       :  10 ms   (200 ms window @ 50% overlap → 100 ms hop)
Causal IIR bandpass  :   5 ms
ASR projection       :  10 ms
Feature (Riemann)    :  10 ms
Classifier (MDM)     :  10 ms
Total                :  ~50 ms
```

A 50 ms total + 100 ms hop = 150 ms end-to-end latency; appropriate for
cursor control. Larger windows (500 ms) are OK for slower decoders
(typing) at the cost of action latency.

### Mini-Repo Contract Relaxation (online_inference)

Per `analysis_goals.py:REGISTRY`, `online_inference` sets
`produces_figures=False`. `contract_check.py` therefore skips the
`figures_missing` rule for this goal. The mini-repo output is still
required (`preprocessed_output/preprocessed`, `pipeline_record.json`,
`README.md`, etc.) — only the figures directory is optional.

### Boundary with `closed_loop_bci.md`

`online_inference` covers **inference-only** — read brain, output decoder
label. `closed_loop_bci` covers **inference + stimulation feedback** —
read brain, output stim. The two paradigms share the upstream (causal
filtering, ASR, incremental classifier) but `closed_loop_bci` adds
stim-trigger logic + safety guards (duty cap, impedance check).
