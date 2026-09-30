# Preprocessing

`musyn.preprocessing` covers general sEMG conditioning that isn't specific
to any of the three papers: filtering, ECG artifact removal, and trial
segmentation/normalization. Unlike the analysis functions, these use
`(n_samples, n_channels)` layout (rows = time) — the transpose of the
`(n_muscles, n_samples)` convention used by {func}`musyn.extract_envelope`,
{func}`musyn.extract_synergies`, and {func}`musyn.select_synergy_number`.
Transpose (`.T`) when moving between the two.

## Filtering

{func}`musyn.preprocessing.condition_emg` applies, in order:

1. A 3rd-order Butterworth bandpass (default 20–450 Hz).
2. 3rd-order Butterworth bandstop notches at the mains-hum fundamental and
   its harmonics (default 50 Hz × 8 harmonics, i.e. 50, 100, ..., 400 Hz,
   each ±0.5 Hz wide).
3. Optionally, single-channel ECG removal on specified columns.

Every stage is independently toggleable and configurable
(`apply_bandpass`/`bandpass_range`, `apply_notch`/`notch_fundamental_hz`/
`notch_harmonics`) — the defaults reproduce the pipeline's original
always-on, fixed-range behavior exactly, so passing no extra arguments is
safe for existing code. Pass `notch_fundamental_hz=60.0` on a 60 Hz mains
grid, or `apply_notch=False` if your acquisition hardware already
hardware-notches mains hum.

```{note}
A single narrow (±0.5 Hz) `filtfilt` notch's edge transients can dominate a
*short* signal's overall amplitude even when steady-state attenuation is
good — this only matters if you're writing tests against very short (few
thousand sample) synthetic signals; real recordings are unaffected.
```

{func}`musyn.preprocessing.linear_envelope` (rectify + low-pass) is the
standard non-adaptive envelope — see {doc}`envelope` for when to prefer the
adaptive algorithm instead.

## ECG artifact removal

Upper-body and trunk EMG recordings (especially near the chest, shoulder,
or upper back) can pick up cardiac electrical activity. `musyn.preprocessing.remove_ecg_scica`
removes it per-channel via single-channel ICA (SC-ICA):

1. Embed the 1-D signal into a Hankel (time-delay) matrix.
2. Decompose with FastICA.
3. Identify the ECG component as the one with maximum excess kurtosis (the
   QRS complex is sharply peaked relative to EMG).
4. Zero that component and reconstruct via diagonal averaging.

Requires `musyn[ecg]` (scikit-learn). Rule of thumb: set `embedding_dim /
fs` to cover one QRS complex (≈4–20 ms) — e.g. `embedding_dim=40` at 1 kHz
covers 40 ms. Run this *after* bandpass/notch filtering, either standalone
or via `condition_emg(..., ecg_channels=[...])`.

## Segmentation and normalization

For cyclic or trial-based paradigms (gait, repeated reaching movements),
three utilities handle the common cross-trial analysis workflow:

- {func}`musyn.preprocessing.segment_envelope` — concatenate envelope
  segments around `(start, stop)` event pairs, with optional pre/post
  padding (`delay_on`/`delay_off`), returning per-trial maxima alongside the
  concatenated data.
- {func}`musyn.preprocessing.time_normalize_envelope` — interpolate each
  trial to a fixed sample count, so trials of different durations (e.g.
  variable gait-cycle length) can be averaged sample-by-sample.
- {func}`musyn.preprocessing.normalize_envelope` — amplitude-normalize by a
  percentile (default 80th) of per-trial maxima, with a numerical floor to
  avoid issues in downstream log-based computations.

A typical flow: `segment_envelope` or `time_normalize_envelope` first (to
get `maxima`), then `normalize_envelope(env, maxima)`.
