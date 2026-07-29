---
name: spike_lfp
layer: L2
group: modality
metadata:
  analysis_goal_allowed:
  - classification
  - feature_extraction
  - exploratory
  - generic
  analysis_goal_forbidden:
  - source_localization
  - online_inference
tags:
- spike
- lfp
- neural_spike
- extracellular
- neuropixels
- utah_array
- single_unit
- multi_unit
- spike_train
- firing_rate
- tuning_curve
modality: spike
---
# Spike Trains and LFP — Extracellular Electrophysiology

## Signal Characteristics

| Property | Single Unit | Multi Unit | LFP |
|----------|-------------|------------|-----|
| Frequency range | 300–6000 Hz | 300–6000 Hz | 0.5–300 Hz |
| Amplitude | 50–500 µV | 20–200 µV | 0.1–5 mV |
| Sampling rate | 20–40 kHz | 20–40 kHz | 1–2.5 kHz |
| Spatial extent | ~50 µm | ~100–200 µm | ~mm |

## Recording Systems

| System | Channels | Electrode Type | Species |
|--------|----------|---------------|---------|
| Neuropixels 1.0 | 384 | Linear silicon probe | Mouse, rat, primate |
| Neuropixels 2.0 | 5120 (384 active) | 4-shank | Mouse, rat |
| Utah array | 96–128 | Microelectrode array | Primate, human |
| Tetrode | 4 per bundle | Twisted wire | Rodent |
| Microwire | 16–64 | Thin wire bundles | Primate |
| Laminar probe | 16–32 | Linear array | Layer analysis |

## Spike Processing Steps

| Step | Method | Purpose |
|------|--------|---------|
| 1. Bandpass | 300–6000 Hz | Isolate spikes from LFP |
| 2. Threshold | ±4–5 × MAD | Detect spike events |
| 3. Extract waveforms | -0.5 to +1.5 ms | Clip spike shapes |
| 4. Feature extraction | PCA, wavelets | Reduce dimensionality |
| 5. Clustering | KiloSort, MountainSort | Assign spikes to units |
| 6. Quality control | ISI violations, SNR | Validate single units |

## Recommended Pipeline

```yaml
# For pre-sorted spike times (most common input)
pipeline:
  - bin_spikes:10          # 10ms bins → firing rate
  - scale:standard         # Normalize across neurons
```

### For Raw Continuous Data
```yaml
pipeline:
  - bandpass:300,6000      # Spike band
  - scale:robust           # Median absolute deviation normalization
```

### Notes
- Most users will have PRE-SORTED spike times (from KiloSort/MountainSort)
- bin_spikes converts spike times → firing rate matrix (neurons × time bins)
- Bin sizes: 1ms (fast dynamics), 10ms (standard), 50ms (slow signals), 100ms+ (behavior)
- Gaussian smoothing after binning: σ = 20–50 ms for smooth firing rates
- Trial alignment: align to behavior events (cue, movement onset, reward)
- Rate vs. timing: some analyses need precise spike times, not binned rates

## LFP Analysis

| Analysis | Frequency | Application |
|----------|-----------|-------------|
| Theta oscillations | 6–12 Hz | Hippocampal navigation, memory |
| Gamma oscillations | 30–100 Hz | Local computation, attention |
| Sharp-wave ripples | 150–250 Hz | Memory replay (offline) |
| Phase coding | Theta phase | Position encoding |
| Spike-LFP coupling | Theta/gamma | Phase locking of spikes |
| Current source density | All | Laminar source identification |

## BCI Applications

| Application | Signal | Decoder |
|-------------|--------|---------|
| Cursor/prosthetic control | Population firing rates | Kalman filter, ReFIT |
| Speech prosthesis | Motor cortex ensemble | RNN, Transformer |
| Handwriting BCI | Motor cortex | RNN character recognizer |
| Point-and-click | Motor + parietal | Two-stage (point + click) |
| Grasp type | Motor/premotor | SVM, neural network |

## Quality Metrics

| Metric | Good | Acceptable | Reject |
|--------|------|------------|--------|
| ISI violations (< 2ms) | < 0.5% | < 2% | > 5% |
| Presence ratio | > 90% | > 70% | < 50% |
| Amplitude cutoff | < 0.1 | < 0.2 | > 0.3 |
| SNR (peak/noise) | > 5 | > 2 | < 1.5 |
| Drift (µm/hr) | < 5 | < 20 | > 50 |
| Firing rate stability | CV < 0.3 | CV < 0.5 | CV > 1.0 |

## File Formats

| Format | Source | Contains |
|--------|--------|----------|
| NWB (.nwb) | Standard | Everything (spikes + LFP + behavior) |
| .nev + .ns6 | Blackrock | Spikes + continuous |
| .bin + .meta | SpikeGLX (Neuropixels) | Raw continuous |
| .kwik/.kwd | KlustaKwik | Sorted spikes |
| .phy | Phy/KiloSort | Sorted spikes (manual curation) |

## Spike–LFP Coupling Analyses

The hallmark of spike–LFP paradigms is that the **two streams are
analyzed jointly** rather than separately. Five canonical analyses:

### 1. Spike-Field Coherence (SFC)

Coherence between the binned spike train of one unit and the LFP at the
same (or nearby) electrode. Quantifies how rhythmically a neuron locks
to ongoing oscillations.

```
SFC(f) = |S_spike,LFP(f)|² / (S_spike,spike(f) · S_LFP,LFP(f))
```

Typical bands: theta (4–8) for hippocampal place cells; gamma (30–80)
for cortical interneurons.

### 2. Phase-Locking Spike Rate (PLV-Locked Rate)

For each spike, record the LFP instantaneous phase φ(t_spike) at the
band of interest. Plot the distribution of phases (circular histogram).
A non-uniform distribution → preferred phase. Measured by mean resultant
vector length:

```
PLV_unit = | (1/N) · Σ_i exp(j · φ(t_spike_i)) |
```

Range [0, 1]; statistically tested via Rayleigh test.

### 3. Theta-Modulated Spiking

Hippocampal pyramidal cells / interneurons spike preferentially at
specific theta phases (Buzsáki 2002). The standard analysis:

```
phase_pref = circular_mean(LFP_theta_phase[at each spike])
mvl = mean resultant length
```

Phase-modulation maps reveal place-cell-like rhythm-encoding.

### 4. Cross-Frequency Coupling Between Spike Rate and LFP Power

Compute spike rate (e.g. 50 ms bin) and LFP gamma envelope; cross-
correlate. Relevant for gamma-rhythm-bound spiking (Fries 2007).

### 5. Granger / Directed Spike↔LFP Influence

Treat binned spike rate and LFP envelope as a multivariate time series;
apply Granger / DTF. Reveals whether spike rate "leads" LFP or vice
versa — relevant for top-down vs bottom-up oscillation theories.

## Recommended Joint Pipeline

```
load_continuous_30kHz → split → ap_band (300-6000) → spike_sort → unit_table
                            └→ lfp_band (1-300) → resample:1000 → save as separate stream

per analysis:
  unit_table + lfp_band → spike_field_coherence
  unit_table + lfp_band → phase_locked_rate
  unit_table → autocorrelogram (cell type)
  unit_table × unit_table → cross_correlogram (connectivity)
```

## Theta-Phase Spike Locking — Worked Example

```python
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# 1. Bandpass LFP at theta (4-8 Hz)
def theta_phase(lfp, sfreq):
    b, a = butter(4, [4 / (sfreq/2), 8 / (sfreq/2)], btype="bandpass")
    return np.angle(hilbert(filtfilt(b, a, lfp)))

# 2. Sample phase at each spike time
def phase_locked_metrics(spike_times_s, lfp, sfreq):
    phases = theta_phase(lfp, sfreq)
    sample_idx = (np.asarray(spike_times_s) * sfreq).astype(int)
    sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < len(phases))]
    spike_phases = phases[sample_idx]
    mvl = float(np.abs(np.mean(np.exp(1j * spike_phases))))
    pref_phase = float(np.angle(np.mean(np.exp(1j * spike_phases))))
    return {"mvl": mvl, "pref_phase_rad": pref_phase, "n_spikes": len(spike_phases)}
```

## Pitfalls & Failure Modes

- **AP-band leakage into LFP.** When using a single broadband recording
  for both, ensure AP and LFP are independently bandpassed first.
- **Spike-LFP contamination.** Spikes themselves contribute to LFP at
  the same channel; for clean coupling analyses, use **the LFP from a
  neighbouring channel** (not the one carrying the unit).
- **Phase-locking with few spikes.** PLV is biased upward at small N;
  always compute Rayleigh p-value.
- **Drift between AP and LF streams.** Sample-rate drift over hours can
  desynchronize the two — use `sample_numbers` for alignment.

## Boundary with Related Paradigms

| Related paradigm | Boundary |
|---|---|
| **`neuropixel_population.md`** | Use Neuropixels paradigm when the focus is **population decoding** (joint firing dynamics). This paradigm focuses on **spike-LFP coupling** per unit / pair. |
| **`utah_array_motor.md`** | Utah array clinical implant; this paradigm is more general (rodent / NHP / human research). |
| **`connectivity.md`** | Macroscale-LFP connectivity only; this paradigm includes spike-level features. |
| **`phase_amplitude_coupling.md`** | PAC is LFP-only (slow phase × fast amplitude); this paradigm couples spike timing to LFP phase. |

## References (Spike-LFP Coupling)

1. Buzsáki, G. (2002). *Theta oscillations in the hippocampus*. Neuron
   33(3): 325–340. doi:10.1016/S0896-6273(02)00586-X.
2. Fries, P. (2007). *Neuronal gamma-band synchronization as a
   fundamental process in cortical computation*. Annual Review of
   Neuroscience 32: 209–224. doi:10.1146/annurev.neuro.051508.135603.
3. Mitchell, J. F. et al. (2009). *Spatial attention decorrelates intrinsic
   activity fluctuations in macaque area V4*. Neuron 63(6): 879–888.
   doi:10.1016/j.neuron.2009.09.013 — spike-field coherence in attention.
4. Womelsdorf, T. et al. (2007). *Modulation of neuronal interactions
   through neuronal synchronization*. Science 316(5831): 1609–1612.
   doi:10.1126/science.1139597 — gamma rhythm and spike timing.
5. Buzsáki, G., & Schomburg, E. W. (2015). *What does gamma coherence
   tell us about inter-regional neural communication?* Nature Neuroscience
   18(4): 484–489. doi:10.1038/nn.3952.
