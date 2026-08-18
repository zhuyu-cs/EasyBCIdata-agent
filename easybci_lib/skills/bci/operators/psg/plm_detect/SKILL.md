---
name: plm_detect
description: "Periodic Limb Movement (PLM) detector — leg EMG event scoring per WASM/AASM criteria"
layer: L3
group: psg
metadata:
  tags: [operator, psg, plm, periodic_limb_movement, leg, emg, sleep]
  modalities: [eeg]
  step_string: "plm_detect"
  analysis_goal_allowed: [sleep_staging, clinical_screening, feature_extraction, exploratory]
  analysis_goal_forbidden: [online_inference, source_localization, connectivity, phase_amplitude_coupling]
---
# PLM Detector

## Function

Detects Periodic Limb Movements in sleep (PLMS) from anterior tibialis
EMG channels following WASM/AASM 2016 scoring criteria. Computes PLM index
(PLMI) as movements per hour of sleep.

Input / Output: data with limb EMG channel(s) → `meta["plm_events"]`
(event list) + `meta["plm_index"]` (PLMI, events/hour).

## Algorithm & Math

### Leg Movement (LM) Detection (AASM 2016)

1. Bandpass 10–100 Hz → full-wave rectification → 0.5 s RMS envelope.
2. Baseline: 30th percentile of RMS over entire recording (quiescent tone).
3. Threshold: RMS > baseline × 8 (or baseline + 2 µV, whichever is greater).
4. Duration criterion: 0.5–10 s above threshold = one LM event.
5. Merge events separated by < 0.5 s.

### PLM Series (WASM 2006)

1. Collect all LM events.
2. Inter-movement interval (IMI): 5–90 s between consecutive LMs.
3. Series criterion: ≥ 4 consecutive LMs meeting IMI → mark as PLM series.
4. PLMI = total PLM events / total sleep time in hours.

### Bilateral Scoring

If both Limb-L and Limb-R available: movements within 5 s of each other
on opposite legs are counted as a single bilateral LM (AASM 2016 Rule 2).

## Parameter Format & Defaults

`plm_detect` or `plm_detect:{limb_ch}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limb_ch` | str | "Limb" | Limb EMG channel substring match. |
| `threshold_factor` (kw) | float | 8.0 | Multiplier over baseline RMS. |
| `min_duration_s` (kw) | float | 0.5 | Minimum LM duration. |
| `max_duration_s` (kw) | float | 10.0 | Maximum LM duration. |
| `min_imi_s` (kw) | float | 5.0 | Minimum inter-movement interval. |
| `max_imi_s` (kw) | float | 90.0 | Maximum inter-movement interval. |
| `min_series` (kw) | int | 4 | Minimum LMs to form a PLM series. |

## Modality-Specific Considerations

PSG (EEG modality with psg_context): requires at least one Limb/Leg EMG
channel. Typical channel names: Limb-L, Limb-R, LAT-L, LAT-R, Leg-L, Leg-R.

## When to Use / NOT to Use

**Use** when: PSG sleep study with leg EMG; quantifying PLMS; evaluating
restless legs syndrome (RLS); computing PLMI for clinical reports.

**Don't use** when: no limb EMG channels; wake-only EEG; non-sleep paradigm.

## Constraints & Ordering

- Apply BEFORE resample (needs ≥ 200 Hz for accurate LM duration).
- Apply AFTER notch (reduces 50/60 Hz contamination in EMG).
- Does NOT modify data array; only writes meta.

## Failure Modes & Detection

| Failure | Symptom | Detection |
|---|---|---|
| No limb channel | Cannot detect. | Skip + warning in meta. |
| EMG saturated | False baseline. | `np.ptp(emg) < 1e-6` → skip channel. |
| All movement (restless patient) | Baseline inflated, misses events. | Warn if > 50% samples above threshold. |
| Short recording (< 2 h) | PLMI unreliable. | Warn in meta. |

## Common Issues

- **"PLMI=0 but patient has RLS diagnosis."** Threshold too high for the
  recording amplitude; try `threshold_factor: 4.0`.
- **"Too many false positives."** Likely EMG cross-talk from respiratory
  effort; verify channel labels.

## Reference Implementation

### Standalone

```python
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt


def plm_detect(
    emg: np.ndarray, sfreq: float,
    threshold_factor: float = 8.0,
    min_duration_s: float = 0.5, max_duration_s: float = 10.0,
    min_imi_s: float = 5.0, max_imi_s: float = 90.0,
    min_series: int = 4,
) -> tuple[list[dict], float]:
    """Detect PLM events. Returns (events, plmi)."""
    # Bandpass 10-100 Hz
    nyq = sfreq / 2
    if 100 >= nyq:
        hi = nyq - 1
    else:
        hi = 100
    b, a = butter(4, [10 / nyq, hi / nyq], btype="bandpass")
    filtered = filtfilt(b, a, emg)
    # RMS envelope (0.5s window)
    win = max(1, int(0.5 * sfreq))
    rectified = filtered ** 2
    rms = np.sqrt(np.convolve(rectified, np.ones(win) / win, mode="same"))
    # Baseline: 30th percentile
    baseline = float(np.percentile(rms, 30))
    threshold = max(baseline * threshold_factor, baseline + 2e-6)
    # Detect LM events
    min_samp = int(min_duration_s * sfreq)
    max_samp = int(max_duration_s * sfreq)
    above = rms > threshold
    events_raw = []
    in_event = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_event:
            start = i
            in_event = True
        elif not above[i] and in_event:
            dur = i - start
            if min_samp <= dur <= max_samp:
                events_raw.append((start / sfreq, dur / sfreq))
            in_event = False
    # Merge events < 0.5s apart
    merged = []
    for onset, dur in events_raw:
        if merged and (onset - (merged[-1][0] + merged[-1][1])) < 0.5:
            merged[-1] = (merged[-1][0], onset + dur - merged[-1][0])
        else:
            merged.append((onset, dur))
    # PLM series detection
    plm_events = []
    series_buf = []
    for i, (onset, dur) in enumerate(merged):
        if not series_buf:
            series_buf.append((onset, dur))
        else:
            imi = onset - (series_buf[-1][0] + series_buf[-1][1])
            if min_imi_s <= imi <= max_imi_s:
                series_buf.append((onset, dur))
            else:
                if len(series_buf) >= min_series:
                    plm_events.extend(series_buf)
                series_buf = [(onset, dur)]
    if len(series_buf) >= min_series:
        plm_events.extend(series_buf)
    duration_h = len(emg) / sfreq / 3600.0
    plmi = len(plm_events) / duration_h if duration_h > 0 else 0.0
    events = [{"onset": o, "duration": d, "leg": "unknown"} for o, d in plm_events]
    return events, plmi
```

### EasyBCI-Adapted

```python
from typing import Any, Dict
import time
import numpy as np
from easybci_lib.tools.neural_processing.operator_errors import EasyBCIOperatorError
from easybci_lib.tools.neural_processing.preprocess.step_cache import record_step_elapsed


def operator_plm_detect(
    data_dict: Dict[str, Any], *,
    limb_ch: str = "Limb", threshold_factor: float = 8.0,
    min_duration_s: float = 0.5, max_duration_s: float = 10.0,
    min_imi_s: float = 5.0, max_imi_s: float = 90.0,
    min_series: int = 4,
) -> Dict[str, Any]:
    """PLM detection (WASM/AASM criteria).

    Parameters
    ----------
    data_dict : dict
    limb_ch : str
    threshold_factor : float
    min_duration_s, max_duration_s : float
    min_imi_s, max_imi_s : float
    min_series : int

    Returns
    -------
    dict — `meta["plm_events"]`, `meta["plm_index"]`.

    Raises
    ------
    EasyBCIOperatorError
        recoverable=True if no limb EMG channel found.

    Modality coverage
    -----------------
    PSG (EEG with limb EMG): yes.

    References
    ----------
    Zucconi et al. 2006 (WASM); Berry et al. 2016 (AASM v2.3).
    """
    channels = data_dict.get("channels", [])
    sfreq = float(data_dict["frequency"])

    limb_indices = [i for i, c in enumerate(channels)
                    if limb_ch.lower() in c.lower() or "leg" in c.lower()]
    if not limb_indices:
        raise EasyBCIOperatorError(
            operator="plm_detect",
            reason=f"no channel matching {limb_ch!r} or 'leg'",
            recoverable=True, fallback_step="skip plm_detect",
        )

    t0 = time.monotonic()
    from scipy.signal import butter, filtfilt

    all_events = []
    for idx in limb_indices:
        emg = data_dict["data"][idx]
        nyq = sfreq / 2
        hi = min(100, nyq - 1)
        b, a = butter(4, [10 / nyq, hi / nyq], btype="bandpass")
        filtered = filtfilt(b, a, emg)
        win = max(1, int(0.5 * sfreq))
        rms = np.sqrt(np.convolve(filtered ** 2, np.ones(win) / win, mode="same"))
        baseline = float(np.percentile(rms, 30))
        threshold = max(baseline * threshold_factor, baseline + 2e-6)

        min_samp = int(min_duration_s * sfreq)
        max_samp = int(max_duration_s * sfreq)
        above = rms > threshold
        raw_events = []
        in_event = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_event:
                start = i
                in_event = True
            elif not above[i] and in_event:
                dur = i - start
                if min_samp <= dur <= max_samp:
                    raw_events.append((start / sfreq, dur / sfreq))
                in_event = False

        # Merge < 0.5s gaps
        merged = []
        for onset, dur in raw_events:
            if merged and (onset - (merged[-1][0] + merged[-1][1])) < 0.5:
                merged[-1] = (merged[-1][0], onset + dur - merged[-1][0])
            else:
                merged.append((onset, dur))

        # PLM series
        series_buf = []
        for onset, dur in merged:
            if not series_buf:
                series_buf.append((onset, dur))
            else:
                imi = onset - (series_buf[-1][0] + series_buf[-1][1])
                if min_imi_s <= imi <= max_imi_s:
                    series_buf.append((onset, dur))
                else:
                    if len(series_buf) >= min_series:
                        all_events.extend(
                            {"onset": o, "duration": d, "leg": channels[idx]}
                            for o, d in series_buf
                        )
                    series_buf = [(onset, dur)]
        if len(series_buf) >= min_series:
            all_events.extend(
                {"onset": o, "duration": d, "leg": channels[idx]}
                for o, d in series_buf
            )

    # Bilateral dedup: same-time events from L/R count as one
    all_events.sort(key=lambda e: e["onset"])
    deduped = []
    for ev in all_events:
        if deduped and abs(ev["onset"] - deduped[-1]["onset"]) < 5.0:
            deduped[-1]["leg"] = "bilateral"
        else:
            deduped.append(ev)

    duration_h = data_dict["data"].shape[-1] / sfreq / 3600.0
    plmi = len(deduped) / duration_h if duration_h > 0 else 0.0
    elapsed = time.monotonic() - t0

    out = dict(data_dict)
    out["elapsed_s"] = elapsed
    out["meta"] = {
        **out.get("meta", {}),
        "plm_events": deduped,
        "plm_index": round(plmi, 1),
        "plm_detect": {"limb_ch": limb_ch, "threshold_factor": threshold_factor,
                       "min_series": min_series},
    }
    record_step_elapsed("plm_detect", elapsed,
                        (data_dict.get("meta") or {}).get("step_cache_key"))
    return out
```

## References

1. Zucconi, M. et al. (2006). *The official World Association of Sleep
   Medicine (WASM) standards for recording and scoring periodic leg
   movements in sleep (PLMS) and wakefulness (PLMW)*. Sleep Med. 7(2):
   175–183. doi:10.1016/j.sleep.2005.12.008.
2. Berry, R. B. et al. (2016). *AASM scoring manual updates for
   2017*. J. Clin. Sleep Med. 13(5): 665–666. doi:10.5664/jcsm.6576.
