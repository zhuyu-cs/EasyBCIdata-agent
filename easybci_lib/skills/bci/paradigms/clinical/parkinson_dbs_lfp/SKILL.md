---
name: parkinson_dbs_lfp
description: 'Parkinson DBS LFP — STN/GPi beta-band pathological signature detection'
layer: L2
group: clinical
metadata:
  tags: [dbs, parkinson, stn, gpi, beta_oscillation, lfp]
  modalities: [lfp]
  paradigms: [dbs_lfp_offline, parkinson_disease]
  analysis_goal_allowed:
    - classification
    - feature_extraction
    - clinical_screening
    - exploratory
    - connectivity
    - phase_amplitude_coupling
    - online_inference
  analysis_goal_forbidden:
    - source_localization
---
# Parkinson DBS LFP

## Neuroscience Background

Deep brain stimulation (DBS) electrodes implanted in the subthalamic
nucleus (STN) or globus pallidus interna (GPi) record local field
potentials (LFP) as a side-effect of their primary therapeutic role.
The pathological signature of Parkinson's disease in these LFP is
**exaggerated beta-band (13–30 Hz) oscillations**, especially the
**low-beta sub-band (13–20 Hz)** in OFF-medication state (Kühn 2006;
Little 2013).

Beta power suppression correlates with motor improvement under both
L-DOPA and DBS. This has motivated **adaptive / closed-loop DBS** —
triggering stimulation only during beta bursts (`dbs_lfp_closedloop.md`).

Additional signatures:
- **β–γ phase-amplitude coupling** (PAC): pathological in PD, normalized
  with stimulation (de Hemptinne 2015).
- **β bursts**: short (<1 s) transient amplitude bursts; duration of
  bursts correlates with symptom severity (Tinkhauser 2017).

## Electrode / Channel Selection

| Electrode kind | Channels | Notes |
|---|---|---|
| Medtronic 3389 | 4 (0, 1, 2, 3) | Standard chronic DBS lead. |
| Boston Sci Vercise / Abbott Infinity | 8 (segmented) | Directional steering. |
| Bipolar recording | (1-2, 2-3) typically | Reduces stim artefact. |

Bipolar pairs preferred for clinical analysis; monopolar useful for
research (more channels, no preprocessing artefact).

## Frequency Bands

| Band | Hz | Pathology / interpretation |
|---|---|---|
| Theta | 4–8 | Cognitive load; not PD-specific. |
| Alpha | 8–13 | Subjective; tremor-band overlap. |
| **Low-beta** | **13–20** | **Pathological PD signature; OFF >> ON medication.** |
| High-beta | 20–30 | Less robust PD marker; movement-modulated. |
| Gamma | 60–100 | Pro-kinetic; appears during voluntary movement. |
| High-gamma / HFO | 200–400 | Less common in chronic DBS LFP. |

## Recommended Pipeline

### Offline analysis

```
load → bandpass:1,300 → notch:50/60 (+stim artefact removal) → bipolar_ref →
drop_bads → multitaper_psd → beta_power_feature
```

### Closed-loop (online) variant — see `dbs_lfp_closedloop.md`

## Common Artifacts

| Artifact | Cause | Mitigation |
|---|---|---|
| **Stimulation artefact** | DBS pulse 130 Hz typical. | Notch at stim freq + harmonics; or template-subtract. |
| **Cardiac pulsatility** | LFP electrode near vasculature. | Regress ECG (if recorded). |
| **Movement / dystonia** | Mechanical motion of electrode. | Reject epochs via auto-detector. |
| **Drift / impedance shift** | Tissue encapsulation over months. | Re-calibrate baseline. |

## Quality Metrics

| Metric | Healthy patient (OFF med) | Reduced (ON med / DBS) |
|---|---|---|
| Low-β power (13–20 Hz, normalized) | > 0.3 (peak) | < 0.15 |
| β burst duration | > 300 ms | < 150 ms |
| β–γ PAC modulation index | > 0.2 | < 0.1 |

## Classification / Decoding Baselines

| Task | Decoder | Accuracy |
|---|---|---|
| Motor state classification (ON / OFF med) | β-power LR | 75–85% |
| Movement-vs-rest detection | β-power threshold | ~80% |
| Burst detection (online) | Hilbert envelope > 75% percentile | gating-quality dependent |

## Public Datasets

| Dataset | Notes |
|---|---|
| OpenNeuro `ds002778` (Schroll 2020) | STN-LFP, PD patients, OFF/ON med. |
| BIDS-iEEG / DBS section | Variable per lab. |

## Pitfalls & Failure Modes

- **Stim-artefact survival.** Even after notch + bipolar, low-frequency
  stim envelope can survive into the beta band. Validate with rest-only
  segments.
- **Single bipolar ≠ omnibus.** Different bipolar pairs see different
  STN sub-regions; use the contact pair with the largest β peak.
- **Cross-patient β-frequency drift.** Some patients have β peak at
  18 Hz, others 24 Hz. Per-patient peak detection > generic 13–20 Hz.

## Boundary with Related Paradigms

- **`dbs_lfp_closedloop.md`** — the online / closed-loop variant of this
  paradigm with real-time β burst detection + stim triggering.
- **`seeg_epilepsy.md`** — seEEG depth electrodes in different anatomy
  (cortex / hippocampus) for epilepsy, not basal ganglia DBS.

## Standalone Pipeline Example

```python
import mne
import numpy as np

raw = mne.io.read_raw_brainvision("pd_dbs_off.vhdr", preload=True)
raw.filter(1, 300)
raw.notch_filter([50, 100, 150, 130, 260])    # line + stim @ 130 Hz
raw.set_eeg_reference("ref_channel", ch_type="dbs")
psd_data, freqs = mne.time_frequency.psd_array_welch(
    raw.get_data(), sfreq=raw.info["sfreq"], fmin=1, fmax=45, n_fft=4096,
)
beta_band = (freqs >= 13) & (freqs <= 20)
beta_power = psd_data[..., beta_band].mean(axis=-1)
print(f"Bipolar β power: {beta_power}")
```

## EasyBCI Pipeline Spec

```yaml
modality: lfp
paradigm: parkinson_dbs_lfp
analysis_goal: clinical_screening
steps:
  - load:brainvision
  - bandpass:1,300
  - notch:50,100,150,130,260
  - bipolar_ref:contacts
  - drop_bads:peak_to_peak
  - multitaper_psd:NW=3,fmin=1,fmax=45
```

## References

1. Kühn, A. A. et al. (2006). *Reduction in subthalamic 8–35 Hz
   oscillatory activity correlates with clinical improvement in
   Parkinson's disease*. Eur. J. Neurosci. 23(7): 1956–1960.
   doi:10.1111/j.1460-9568.2006.04717.x.
2. Little, S. et al. (2013). *Adaptive deep brain stimulation in advanced
   Parkinson disease*. Annals of Neurology 74(3): 449–457.
   doi:10.1002/ana.23951.
3. de Hemptinne, C. et al. (2015). *Therapeutic deep brain stimulation
   reduces cortical phase-amplitude coupling in Parkinson's disease*.
   Nature Neuroscience 18(5): 779–786. doi:10.1038/nn.3997.
4. Tinkhauser, G. et al. (2017). *The modulatory effect of adaptive deep
   brain stimulation on beta bursts in Parkinson's disease*. Brain
   140(4): 1053–1067. doi:10.1093/brain/awx010.
