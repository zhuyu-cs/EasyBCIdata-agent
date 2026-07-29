---
name: hfo_detector
description: "High-Frequency Oscillation (80-500 Hz) detector — sEEG epilepsy biomarker"
layer: L3
group: qc_operator
metadata:
  tags: [operator, qc, hfo, ripple, fast_ripple, seeg, epilepsy]
  modalities: [seeg, ecog]
  step_string: "hfo_detector"
  analysis_goal_allowed: [feature_extraction, clinical_screening, exploratory]
  analysis_goal_forbidden: [online_inference, source_localization]
---
# HFO Detector

## Function

Detects High-Frequency Oscillations (HFO) — sub-events typically 80–500 Hz
oscillating bursts that are an established biomarker for the epileptogenic
zone in sEEG (Worrell 2008; Jiruska 2017).

Two sub-classes:
- **Ripples** (80–250 Hz): physiologic + pathologic.
- **Fast ripples** (250–500 Hz): strongly pathologic.

Input / Output: continuous sEEG/ECoG → `meta["hfo_events"]: list[(channel, start_s, end_s, kind)]`.

## Algorithm & Math

Standard pipeline (Staba 2002):
1. Bandpass 80–250 Hz (ripple) or 250–500 Hz (fast ripple).
2. Hilbert envelope.
3. Threshold (3–5 σ of session-wide MAD).
4. Cluster oscillation cycles > 6 per event.
5. Reject muscle artifacts (broadband > 600 Hz energy filter).

## Parameter Format & Defaults

`hfo_detector:{band}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `band` | str | "ripple" | "ripple" / "fast_ripple" / "both". |
| `k` (kw) | float | 5.0 | σ-multiplier threshold. |
| `min_cycles` (kw) | int | 6 | Minimum oscillation cycles. |

## Modality-Specific Considerations

sEEG / ECoG with sample rate ≥ 1024 Hz.

## When to Use / NOT to Use

**Use** when: epilepsy localization; clinical seizure-zone identification.

**Don't use** when: < 1 kHz sample rate (band too narrow); online; EEG scalp
(HFO not reliably detectable at scalp).

## Constraints & Ordering

- Apply after notch / bandpass:1,500 / drop_bads.
- sample rate ≥ 1024 Hz for ripple; ≥ 2048 Hz for fast ripple.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| Sample rate too low | Aliased band. | Pre-check; raise recoverable. |
| Muscle contamination | False positives at high amplitude. | Pre-check broadband > 600 Hz energy. |

## Common Issues

- **"Too many false positives."** Raise `k` to 6–7 or tighten min_cycles.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def hfo_detector(
    data: np.ndarray, sfreq: float, band: str = "ripple",
    k: float = 5.0, min_cycles: int = 6,
) -> list:
    """Returns list of (channel, start_s, end_s, kind)."""
    bands = {"ripple": (80, 250), "fast_ripple": (250, 500)}
    lo, hi = bands.get(band, (80, 250))
    if hi >= sfreq / 2:
        raise ValueError(f"hfo: band {hi} >= Nyquist {sfreq/2}")
    b, a = butter(4, [lo / (sfreq/2), hi / (sfreq/2)], btype="bandpass")
    filtered = filtfilt(b, a, data, axis=-1)
    env = np.abs(hilbert(filtered, axis=-1))
    threshold = k * (np.median(np.abs(env), axis=-1, keepdims=True) / 0.6745)
    events = []
    min_samples = int(min_cycles * sfreq / ((lo + hi) / 2))
    for c in range(data.shape[0]):
        above = env[c] > threshold[c]
        in_event = False; start = 0
        for t in range(len(above)):
            if above[t] and not in_event:
                start = t; in_event = True
            elif not above[t] and in_event:
                if t - start >= min_samples:
                    events.append((c, start / sfreq, t / sfreq, band))
                in_event = False
    return events
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_hfo_detector(
    data_dict: Dict[str, Any], *, band: str = "ripple",
    k: float = 5.0, min_cycles: int = 6,
) -> Dict[str, Any]:
    """HFO detection.

    Parameters
    ----------
    data_dict : dict
    band : str
        "ripple" / "fast_ripple" / "both".
    k : float
    min_cycles : int

    Returns
    -------
    dict — `meta["hfo_events"]: list of tuples`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if sample rate inadequate.

    Modality coverage
    -----------------
    sEEG / ECoG: yes. EEG scalp / others: forbidden.

    References
    ----------
    Staba et al. 2002; Worrell et al. 2008; Jiruska et al. 2017.
    """
    sfreq = float(data_dict["frequency"])
    bands_to_check = []
    if band in ("ripple", "both"):
        bands_to_check.append(("ripple", 80, 250))
    if band in ("fast_ripple", "both"):
        bands_to_check.append(("fast_ripple", 250, 500))
    for kind, lo, hi in bands_to_check:
        if hi >= sfreq / 2:
            raise EasyBCIOperatorError(
                operator="hfo_detector",
                reason=f"band {kind} ({hi} Hz) >= Nyquist {sfreq/2}",
                recoverable=True, fallback_step="resample then hfo_detector",
            )

    t0 = time.monotonic()
    from scipy.signal import butter, filtfilt, hilbert
    events = []
    for kind, lo, hi in bands_to_check:
        b, a = butter(4, [lo / (sfreq/2), hi / (sfreq/2)], btype="bandpass")
        filtered = filtfilt(b, a, data_dict["data"], axis=-1)
        env = np.abs(hilbert(filtered, axis=-1))
        threshold = k * (np.median(np.abs(env), axis=-1, keepdims=True) / 0.6745)
        min_samples = max(1, int(min_cycles * sfreq / ((lo + hi) / 2)))
        for c in range(data_dict["data"].shape[0]):
            above = env[c] > threshold[c, 0]
            in_event = False; start = 0
            for ti in range(len(above)):
                if above[ti] and not in_event:
                    start = ti; in_event = True
                elif not above[ti] and in_event:
                    if ti - start >= min_samples:
                        events.append((c, start / sfreq, ti / sfreq, kind))
                    in_event = False
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {**out.get("meta", {}), "hfo_events": events,
                   "hfo_detector": {"band": band, "k": k, "min_cycles": min_cycles}}
    record_step_elapsed("hfo_detector", elapsed, (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Staba, R. J. et al. (2002). *Quantitative analysis of high-frequency
   oscillations (80–500 Hz) recorded in human epileptic hippocampus
   and entorhinal cortex*. J. Neurophysiol. 88(4): 1743–1752.
   doi:10.1152/jn.2002.88.4.1743.
2. Worrell, G. A. et al. (2008). *High-frequency oscillations in human
   temporal lobe: simultaneous microwire and clinical macroelectrode
   recordings*. Brain 131(4): 928–937. doi:10.1093/brain/awn006.
3. Jiruska, P. et al. (2017). *Update on the mechanisms and roles of
   high-frequency oscillations in seizures and epileptic disorders*.
   Epilepsia 58(8): 1330–1339. doi:10.1111/epi.13830.
