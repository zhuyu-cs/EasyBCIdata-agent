---
name: epoch_qc_sleep
description: "30-second epoch-level multi-channel quality scoring with hypnogram-aware thresholds"
layer: L3
group: psg
metadata:
  tags: [operator, psg, qc, epoch, sleep, hypnogram, artifact, quality]
  modalities: [eeg]
  step_string: "epoch_qc_sleep"
  analysis_goal_allowed: [sleep_staging, clinical_screening, feature_extraction, exploratory]
  analysis_goal_forbidden: [online_inference, source_localization, connectivity, phase_amplitude_coupling]
---
# Epoch QC (Sleep)

## Function

Scores quality of each 30-second epoch across all channels simultaneously.
When a hypnogram is available, adapts thresholds per sleep stage (e.g., high
delta amplitude is normal in N3, high EMG is normal in Wake). Produces a
per-epoch quality mask for downstream staging or feature extraction.

Input / Output: resampled multi-channel data + optional hypnogram →
`meta["epoch_scores"]` (quality array) + `meta["usable_epochs"]` (bool mask)
+ `meta["epoch_reject_reasons"]` (per-epoch string list).

## Algorithm & Math

### Per-Epoch Metrics (computed per channel, then aggregated)

| Metric | Formula | Purpose |
|---|---|---|
| Amplitude range | `np.ptp(epoch)` | Detect saturated/railed signals. |
| Flatness | `np.std(epoch) < flat_threshold` | Detect electrode dropout. |
| Gradient | `np.max(np.abs(np.diff(epoch)))` | Detect abrupt jumps/pops. |
| HF power ratio | `power(30-45 Hz) / power(0.5-45 Hz)` | Detect muscle artifact in EEG. |

### Aggregation

1. Each metric → z-score across all epochs of same channel.
2. Composite score per epoch: `1.0 - max(clipped_z_scores) / z_limit`.
3. Clamped to [0, 1]; threshold default = 0.5.

### Hypnogram-Aware Mode

When `meta["stages"]` or hypnogram labels are available:

| Stage | Adjustment |
|---|---|
| N3 (SWS) | Amplitude threshold ×2 (high delta is expected). |
| REM | EOG amplitude not penalized; EMG flatness not penalized. |
| Wake | EMG power not penalized (muscle tone normal). |
| N1/N2 | No adjustment (default thresholds). |

### No-Hypnogram Mode

All thresholds use global defaults. Usable as pure signal QC without
staging information.

## Parameter Format & Defaults

`epoch_qc_sleep` or `epoch_qc_sleep:{threshold}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | float | 0.5 | Minimum quality score to keep epoch. |
| `epoch_s` (kw) | float | 30.0 | Epoch duration (AASM: 30 s). |
| `flat_uv` (kw) | float | 1.0 | Flatness threshold in µV (std below this = flat). |
| `gradient_uv` (kw) | float | 200.0 | Max sample-to-sample jump in µV. |
| `hf_ratio_limit` (kw) | float | 0.4 | Max HF power ratio before penalty. |
| `z_limit` (kw) | float | 4.0 | Z-score clamp for composite scoring. |

## Modality-Specific Considerations

PSG (EEG modality): applies to all channel types kept in the array (EEG,
EOG, EMG, respiratory). Respiratory channels use only amplitude/flatness
metrics (HF ratio irrelevant). Channel routing based on `psg_aux` metadata.

## When to Use / NOT to Use

**Use** when: PSG sleep study after filtering/resampling; need epoch-level
artifact annotation; preparing data for staging or feature extraction.

**Don't use** when: continuous (non-epoched) analysis; raw unfiltered data
(metrics will be unreliable); non-sleep paradigm (use standard QC operators).

## Constraints & Ordering

- Apply AFTER resample (needs uniform sample rate for consistent metrics).
- Apply AFTER bandpass (HF ratio metric meaningless on unfiltered data).
- Apply AFTER respiratory_events and plm_detect (those need original rate).
- Does NOT modify data array; only writes meta.
- Complementary to `drop_bads` (channel-level) — this operates at
  epoch-level and is finer-grained.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| All epochs rejected | threshold too strict. | Warn if > 80% rejected; suggest lowering threshold. |
| No epochs rejected | threshold too lenient or data pristine. | Info-level, not error. |
| Hypnogram length mismatch | Stages shorter/longer than epochs. | Truncate or pad with "unscored"; warn. |
| Short recording | < 60 epochs (30 min). | Warn; metrics less reliable. |

## Common Issues

- **"N3 epochs all rejected."** SWS has high delta amplitude by definition;
  enable hypnogram-aware mode or increase amplitude threshold.
- **"REM epochs rejected for EOG."** EOG rapid eye movements are
  physiological in REM; ensure hypnogram-aware exemption is active.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import welch


def epoch_qc_sleep(
    data: np.ndarray, sfreq: float,
    stages: list[str] | None = None,
    threshold: float = 0.5, epoch_s: float = 30.0,
    flat_uv: float = 1.0, gradient_uv: float = 200.0,
    hf_ratio_limit: float = 0.4, z_limit: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Epoch-level QC. Returns (scores, usable_mask, reject_reasons)."""
    n_ch, n_samples = data.shape
    epoch_samples = int(epoch_s * sfreq)
    n_epochs = n_samples // epoch_samples
    data_cut = data[:, :n_epochs * epoch_samples].reshape(n_ch, n_epochs, epoch_samples)

    # Compute metrics per epoch per channel
    amp_range = np.ptp(data_cut, axis=-1)         # (n_ch, n_epochs)
    flatness = np.std(data_cut, axis=-1)          # (n_ch, n_epochs)
    gradient = np.max(np.abs(np.diff(data_cut, axis=-1)), axis=-1)

    # HF power ratio per epoch per channel
    hf_ratio = np.zeros((n_ch, n_epochs))
    for ep in range(n_epochs):
        for ch in range(n_ch):
            f, psd = welch(data_cut[ch, ep], fs=sfreq, nperseg=min(int(4 * sfreq), epoch_samples))
            total = psd[(f >= 0.5) & (f <= 45)].sum()
            hf = psd[(f >= 30) & (f <= 45)].sum()
            hf_ratio[ch, ep] = hf / total if total > 0 else 0.0

    # Z-score each metric across epochs (per channel)
    def _zscore(arr):
        mu = arr.mean(axis=-1, keepdims=True)
        sd = arr.std(axis=-1, keepdims=True)
        sd = np.maximum(sd, 1e-10)
        return (arr - mu) / sd

    z_amp = _zscore(amp_range)
    z_grad = _zscore(gradient)
    z_hf = _zscore(hf_ratio)

    # Flat detection (binary)
    is_flat = flatness < flat_uv

    # Stage-aware threshold adjustment
    stage_factor = np.ones(n_epochs)
    if stages:
        for ep in range(min(n_epochs, len(stages))):
            if stages[ep] in ("N3", "SWS"):
                stage_factor[ep] = 2.0  # amplitude tolerance doubled

    # Composite score
    z_combined = np.maximum(np.maximum(np.abs(z_amp), np.abs(z_grad)), np.abs(z_hf))
    z_worst = z_combined.max(axis=0)  # worst channel per epoch
    z_worst = np.minimum(z_worst / stage_factor, z_limit)
    scores = 1.0 - z_worst / z_limit
    scores = np.clip(scores, 0.0, 1.0)

    # Flat epochs get score 0
    any_flat = is_flat.any(axis=0)
    scores[any_flat] = 0.0

    usable = scores >= threshold
    reasons = []
    for ep in range(n_epochs):
        if usable[ep]:
            reasons.append("")
        elif any_flat[ep]:
            reasons.append("flat_channel")
        elif z_amp[:, ep].max() > z_limit:
            reasons.append("high_amplitude")
        elif z_grad[:, ep].max() > z_limit:
            reasons.append("gradient_jump")
        elif z_hf[:, ep].max() > z_limit:
            reasons.append("muscle_artifact")
        else:
            reasons.append("low_quality")
    return scores, usable, reasons
```

### EasyBCI-Adapted

```python
from typing import Any, Dict, List
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_epoch_qc_sleep(
    data_dict: Dict[str, Any], *,
    threshold: float = 0.5, epoch_s: float = 30.0,
    flat_uv: float = 1.0, gradient_uv: float = 200.0,
    hf_ratio_limit: float = 0.4, z_limit: float = 4.0,
) -> Dict[str, Any]:
    """Epoch-level QC for PSG sleep data.

    Parameters
    ----------
    data_dict : dict
    threshold : float
    epoch_s : float
    flat_uv, gradient_uv : float
    hf_ratio_limit : float
    z_limit : float

    Returns
    -------
    dict — `meta["epoch_scores"]`, `meta["usable_epochs"]`,
    `meta["epoch_reject_reasons"]`, `meta["epoch_qc_summary"]`.

    Modality coverage
    -----------------
    PSG (EEG with auxiliary channels): yes.

    References
    ----------
    AASM Manual for Scoring Sleep v2.6 (2020).
    """
    sfreq = float(data_dict["frequency"])
    data = data_dict["data"]
    n_ch, n_samples = data.shape
    epoch_samples = int(epoch_s * sfreq)
    n_epochs = n_samples // epoch_samples

    if n_epochs < 1:
        raise EasyBCIOperatorError(
            operator="epoch_qc_sleep",
            reason=f"recording too short for {epoch_s}s epochs",
            recoverable=True, fallback_step="skip epoch_qc_sleep",
        )

    t0 = time.monotonic()
    from scipy.signal import welch as _welch

    data_cut = data[:, :n_epochs * epoch_samples].reshape(n_ch, n_epochs, epoch_samples)

    amp_range = np.ptp(data_cut, axis=-1)
    flatness = np.std(data_cut, axis=-1)
    gradient = np.max(np.abs(np.diff(data_cut, axis=-1)), axis=-1)

    hf_ratio = np.zeros((n_ch, n_epochs), dtype=np.float32)
    nperseg = min(int(4 * sfreq), epoch_samples)
    for ep in range(n_epochs):
        for ch in range(n_ch):
            f, psd = _welch(data_cut[ch, ep], fs=sfreq, nperseg=nperseg)
            total = psd[(f >= 0.5) & (f <= 45)].sum()
            hf = psd[(f >= 30) & (f <= 45)].sum()
            hf_ratio[ch, ep] = hf / total if total > 0 else 0.0

    def _zscore(arr):
        mu = arr.mean(axis=-1, keepdims=True)
        sd = np.maximum(arr.std(axis=-1, keepdims=True), 1e-10)
        return (arr - mu) / sd

    z_amp = _zscore(amp_range)
    z_grad = _zscore(gradient)
    z_hf = _zscore(hf_ratio)

    is_flat = flatness < flat_uv

    # Hypnogram-aware adjustments
    stages: List[str] = []
    meta = data_dict.get("meta", {})
    if "stages" in meta:
        stages = meta["stages"]
    elif "hypnogram_path" in meta:
        from easybci_lib.tools.neural_processing.io.psg_annotations import parse_hypnogram
        stages = parse_hypnogram(meta["hypnogram_path"])

    stage_factor = np.ones(n_epochs)
    for ep in range(min(n_epochs, len(stages))):
        s = stages[ep]
        if s in ("N3", "SWS"):
            stage_factor[ep] = 2.0
        elif s == "REM":
            stage_factor[ep] = 1.5

    z_combined = np.maximum(np.maximum(np.abs(z_amp), np.abs(z_grad)), np.abs(z_hf))
    z_worst = z_combined.max(axis=0)
    z_worst = np.minimum(z_worst / stage_factor, z_limit)
    scores = np.clip(1.0 - z_worst / z_limit, 0.0, 1.0).astype(np.float32)
    scores[is_flat.any(axis=0)] = 0.0

    usable = scores >= threshold
    reasons: List[str] = []
    for ep in range(n_epochs):
        if usable[ep]:
            reasons.append("")
        elif is_flat[:, ep].any():
            reasons.append("flat_channel")
        elif np.abs(z_amp[:, ep]).max() > z_limit:
            reasons.append("high_amplitude")
        elif np.abs(z_grad[:, ep]).max() > z_limit:
            reasons.append("gradient_jump")
        elif np.abs(z_hf[:, ep]).max() > z_limit:
            reasons.append("muscle_artifact")
        else:
            reasons.append("low_quality")

    usable_pct = float(usable.sum()) / n_epochs * 100
    elapsed = time.monotonic() - t0

    if usable_pct < 20:
        import logging
        logging.getLogger(__name__).warning(
            "epoch_qc_sleep: only %.0f%% epochs usable — consider relaxing threshold", usable_pct)

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **meta,
        "epoch_scores": scores.tolist(),
        "usable_epochs": usable.tolist(),
        "epoch_reject_reasons": reasons,
        "epoch_qc_summary": {
            "n_epochs": n_epochs, "n_usable": int(usable.sum()),
            "usable_pct": round(usable_pct, 1),
            "hypnogram_aware": len(stages) > 0,
        },
        "epoch_qc_sleep": {"threshold": threshold, "epoch_s": epoch_s,
                           "z_limit": z_limit},
    }
    record_step_elapsed("epoch_qc_sleep", elapsed,
                        (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Berry, R. B. et al. (2020). *The AASM manual for the scoring of
   sleep and associated events: rules, terminology and technical
   specifications*, version 2.6. American Academy of Sleep Medicine.
2. Iber, C. et al. (2007). *The AASM manual for the scoring of sleep
   and associated events*, 1st edition. AASM.
3. Himanen, S. L. & Hasan, J. (2000). *Limitations of Rechtschaffen
   and Kales*. Sleep Med. Rev. 4(2): 149–167.
   doi:10.1053/smrv.1999.0086.
