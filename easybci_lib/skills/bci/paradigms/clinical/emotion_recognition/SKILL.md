---
name: emotion_recognition
layer: L2
group: clinical
metadata:
  analysis_goal_allowed:
  - classification
  - feature_extraction
  - exploratory
  - generic
  analysis_goal_forbidden:
  - clinical_screening
  - source_localization
tags:
- eeg
- emotion
- affective
- valence
- arousal
- asymmetry
- frontal_alpha
- eeg_emotion
- affective_computing
modality: eeg
---
# Emotion Recognition — Affective EEG Processing

## Signal Characteristics

| Property | Typical Value |
|----------|--------------|
| Sampling rate | 128–512 Hz |
| Channels | 14–64 (EMOTIV: 14, research: 32–62) |
| Key regions | Prefrontal (Fp1/Fp2, F3/F4, F7/F8), Temporal (T7/T8) |
| Trial duration | 5–60 seconds per stimulus |
| Emotion models | Valence-Arousal (2D), Discrete (happy/sad/anger/fear/neutral) |

## Theoretical Basis

| Theory | Feature | Interpretation |
|--------|---------|----------------|
| Frontal asymmetry | F4_alpha - F3_alpha | Positive = approach (positive valence) |
| Prefrontal theta | Fp1/Fp2 theta power | Working memory load, frustration |
| Parietal alpha | P3/P4 alpha | Arousal (lower alpha = higher arousal) |
| Gamma binding | Distributed gamma | Emotional intensity, feature integration |
| Beta activity | Frontal beta | Anxiety, active engagement |

## Common Datasets

| Dataset | Channels | Subjects | Stimuli | Classes |
|---------|----------|----------|---------|---------|
| DEAP | 32 | 32 | Music videos | Valence/Arousal (1–9) |
| SEED | 62 | 15 | Film clips | Positive/Neutral/Negative |
| DREAMER | 14 | 23 | Film clips | Valence/Arousal/Dominance |
| MAHNOB-HCI | 32 | 27 | Film clips | Valence/Arousal (1–9) |
| AMIGOS | 14 | 40 | Film clips | Valence/Arousal |

## Recommended Pipeline

```yaml
pipeline:
  - notch:50              # Line noise
  - bandpass:1,50         # Include gamma, exclude DC drift
  - resample:256          # Preserve gamma up to 50 Hz
  - drop_bads             # Eye/muscle-heavy channels (Fp1/Fp2 keep if frontal asymmetry)
  - scale:robust          # Handle blink artifacts in frontal channels
```

### Notes
- Frontal channels (Fp1/Fp2) are critical for asymmetry but also most contaminated by EOG
- ICA artifact removal is essential BEFORE feature extraction (eye blinks corrupt frontal alpha)
- Baseline correction: subtract pre-stimulus period (1–5s) from each trial
- Feature extraction typically per-band: delta, theta, alpha, beta, gamma power
- Differential entropy (DE) outperforms band power in most benchmarks
- Cross-subject generalization is the main challenge — domain adaptation helps
- Consider 1-second non-overlapping windows for feature extraction (SEED convention)

## Feature Extraction

| Feature | Bands | Computation |
|---------|-------|-------------|
| Band power (PSD) | All 5 bands | Welch's method per channel |
| Differential entropy | All 5 bands | 0.5 × log(2πe × σ²) |
| Frontal asymmetry | Alpha, beta | Right - Left hemisphere power |
| Coherence | Theta, alpha | Inter-channel phase coupling |
| Hjorth parameters | Broadband | Activity, mobility, complexity |
| Fractal dimension | Broadband | Higuchi or Katz method |

## Quality Metrics

- Trial rejection: exclude if > 30% timepoints exceed ±100 µV
- Minimum usable trials per emotion: ≥ 10 per class
- Class balance: check for stimulus-induced bias
- Baseline stability: verify pre-stimulus alpha across trials
