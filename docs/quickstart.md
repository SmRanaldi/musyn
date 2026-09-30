# Quickstart

This walks through the full pipeline on synthetic data: preprocessing →
adaptive envelope → synergy number selection → NMF synergy extraction →
quality metrics. It mirrors `examples/04_full_pipeline.py` in the repository
— see {doc}`examples` to run it directly.

## 1. Preprocess raw sEMG

Raw EMG arrays use `(n_samples, n_channels)` layout (rows = time):

```python
import numpy as np
from musyn.preprocessing import condition_emg

raw_emg = np.random.randn(20000, 8)  # 8 channels, 20000 samples
emg_clean = condition_emg(raw_emg, fs=2000.0)  # bandpass [20-450 Hz] + 50 Hz notch x8 harmonics
```

Pass `ecg_channels=[...]` to also run single-channel ICA ECG removal on
specific columns (requires `musyn[ecg]`); see
{func}`musyn.preprocessing.remove_ecg_scica`.

## 2. Extract the envelope

The analysis functions (`extract_synergies`, `select_synergy_number`) use
the *transposed* `(n_muscles, n_samples)` convention — muscles as rows.

```python
import musyn

envelope = musyn.extract_envelope(emg_clean.T, fs=2000.0)  # (8, 20000)
```

`extract_envelope` also accepts a 1-D single-channel signal. Pass
`return_info=True` to inspect which backend ran (`'cython'`, `'numba'`, or
`'numpy'`) and per-sample adaptive window lengths — see
{doc}`user_guide/envelope`.

For a simpler (non-adaptive) envelope, `musyn.preprocessing.linear_envelope`
does rectification + low-pass filtering instead.

## 3. Select the number of synergies

```python
k = musyn.select_synergy_number(envelope, events=[0, 5000, 10000, 15000, 20000])
```

`events` marks segment boundaries (e.g. movement-cycle onsets/offsets) and
is **strongly recommended** — see {doc}`user_guide/selection` for why it
matters and how to pick a `method`.

## 4. Extract synergies

```python
W, C, info = musyn.extract_synergies(envelope, n_synergies=k)
# W: (8, k) synergy weights: how much each muscle contributes to each synergy
# C: (k, 20000) activation time courses
```

## 5. Evaluate reconstruction quality

```python
print(f"VAF = {musyn.vaf(envelope, W, C):.3f}")
print(f"R²  = {musyn.r_squared(envelope, W, C):.3f}")
```

If you're benchmarking against a known ground truth `W_true` (e.g. on
synthetic data), use {func}`musyn.quality_ratio` to measure how well the
extracted synergies match, independent of column ordering.

## Next steps

- {doc}`user_guide/index` for how each algorithm works and its parameters
- {doc}`api/index` for the full function reference
- {doc}`examples` to run the complete scripts, including plots
