---
name: notch_filter
description: "Notch filter to remove power line interference (50/60 Hz) and harmonics"
layer: L3
group: filter
metadata:
  tags: [operator, filter, notch, powerline]
  modalities: [eeg, seeg, ecog, meg]
  step_string: "notch"
  analysis_goal_allowed: [classification, source_localization, feature_extraction, clinical_screening, exploratory, generic, online_inference]
  analysis_goal_forbidden: []
---
# Notch Filter

## Function

Removes power line interference at specified frequencies and their harmonics up to Nyquist. Uses MNE zero-phase FIR notch filter.

## Parameter Format

`notch:{frequency}` or `notch:{freq1},{freq2}`

Examples:
- `notch:50` — 50 Hz + harmonics (100, 150, 200, ...)
- `notch:60` — 60 Hz + harmonics
- `notch:50,60` — Both 50 and 60 Hz lines + all harmonics

Default (no parameter): `notch:50`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| frequency | float or comma-separated floats | 50 | Base frequency/frequencies to notch |

Harmonics are automatically generated: `f, 2f, 3f, ...` up to `min(Nyquist, 300 Hz)`.

## When to Use

- PSD shows sharp peaks at 50 Hz (Europe/Asia) or 60 Hz (Americas)
- Data was recorded without hardware notch filter
- Line noise visible as regular oscillation in time domain

## When NOT to Use

- Data already has hardware notch applied (check recording metadata)
- Analysis focuses on frequencies near the notch (e.g., gamma band 40-60 Hz) — consider narrower custom filter
- Spike data (not applicable)

## Constraints

- Frequency must be < Nyquist (sfreq/2). Harmonics above Nyquist are automatically excluded.
- Zero-phase filter: no temporal distortion but requires sufficient signal length
- Transition bandwidth: ~1 Hz. Frequencies within ±0.5 Hz of notch center are affected.

## Ordering

- Typically **first step** in the pipeline
- Must be applied BEFORE bandpass if the bandpass upper bound excludes line noise
- Apply BEFORE ICA (line noise confuses ICA decomposition)

## Recommended Parameters

| Region | Frequency | Notes |
|--------|-----------|-------|
| Europe, Asia, Africa | 50 | 50 Hz power grid |
| Americas, Japan (East) | 60 | 60 Hz power grid |
| Unknown/mixed | 50,60 | Safe default for both |

## Reference Code

### MNE Chain (on existing Raw object)

```python
# Single frequency
raw.notch_filter(50.0, verbose=False)
# Multiple frequencies
raw.notch_filter([50.0, 60.0], verbose=False)
```

### Standalone with Auto-Harmonics

```python
import mne
import numpy as np

sfreq = 256.0  # from data
base_freq = 50.0
freqs = np.arange(base_freq, min(sfreq / 2, 301), base_freq).tolist()

info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info, verbose=False)
raw.notch_filter(freqs, phase='zero', verbose=False)
data = raw.get_data().astype(np.float32)
```

### Key API

- `raw.notch_filter(freqs, phase='zero', verbose=False)`
- Harmonics generation: `np.arange(base_freq, min(nyquist, 301), base_freq)`

## Narrow Q vs Wide Q — PAC / Connectivity Cautions

The notch's quality factor `Q = f0 / Δf_3dB` controls the notch
width: wider notches drink more energy from neighbouring bands. For
spectral / band-power analysis this is a non-issue; for **PAC and
phase-coupling** analyses the notch must be selected with care.

| Q regime | Width @ 50 Hz | When safe | When forbidden |
|---|---|---|---|
| Narrow (Q ≥ 50) | ≤ 1 Hz | All goals incl. PAC / connectivity (preserves phase). | — |
| Medium (Q 20–50) | 1–2.5 Hz | Classification, ERP, clinical screening. | PAC where carrier band touches 50 Hz ± 2 Hz. |
| Wide (Q < 20) | > 2.5 Hz | One-pass line-noise scrub on dirty data. | Connectivity / PAC analyses (phase distortion + bandwidth loss). |

For PAC / connectivity tasks, the safer choice is **`zapline`**
(spectral-PCA line-noise removal — `bci/operators/filter/zapline/`),
which preserves wide-band phase. When you must use `notch_filter` in
PAC analyses, set `Q >= 50` and verify the PSD: any visible dip wider
than ±0.5 Hz around the notch frequency is too wide for PAC.

The `connectivity` and `phase_amplitude_coupling` goals are therefore
excluded from this operator's `analysis_goal_allowed` list — they are
not strictly forbidden (a narrow-Q notch is safe), but the recommended
operator for those goals is `zapline`. Use this notch operator only
when zapline is not viable (low channel count, real-time constraints).

## Constraints — Ordering with PAC / Connectivity

When the downstream `analysis_goal ∈ {phase_amplitude_coupling, connectivity}`:

1. **Prefer `zapline`** — see above.
2. If you must use `notch_filter`: pin `Q >= 50`, validate PSD width
   ≤ 1 Hz, and document the choice in `plan/reasoning.md`.
3. **Forbidden** when the PAC carrier band overlaps the notch ± 2 Hz
   (e.g., 50 Hz PAC carrier with US-mains 60 Hz notch is OK; with
   EU-mains 50 Hz it is forbidden — switch to zapline).
