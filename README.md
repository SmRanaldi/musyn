# musyn — Muscle Synergy Analysis

[![Tests](https://github.com/SmRanaldi/musyn/actions/workflows/tests.yml/badge.svg)](https://github.com/SmRanaldi/musyn/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/musyn/badge/?version=latest)](https://musyn.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/musyn.svg)](https://pypi.org/project/musyn/)
[![Python versions](https://img.shields.io/pypi/pyversions/musyn.svg)](https://pypi.org/project/musyn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python package for muscle synergy analysis from surface EMG (sEMG). Implements
three peer-reviewed algorithms with high-performance backends, plus standard
preprocessing utilities.

**Full documentation: [musyn.readthedocs.io](https://musyn.readthedocs.io)**
— installation, a user guide for each algorithm, the complete API reference,
and annotated examples. This README covers the same ground more tersely.

## Algorithms

| # | Algorithm | Paper | Performance |
|---|-----------|-------|-------------|
| 1 | Adaptive envelope extraction | Ranaldi et al. 2018 | Cython / Numba / NumPy |
| 2 | NMF synergy extraction | Soomro et al. 2018 | BLAS-backed NumPy |
| 3 | AIC synergy number selection | Ranaldi et al. 2021 | Parallel k-sweep |

## Installation

```bash
pip install musyn                  # core (envelope + NMF; AIC methods 'vaf'/'r2'/'plateau'/'surrogate')
pip install musyn[wavelet]         # + PyWavelets, for AIC methods 'min'/'der'/'firstpeak'/'lastpeak' (default method='min')
pip install musyn[ecg]             # + ECG removal (requires scikit-learn)
pip install musyn[jit]             # + Numba JIT backend for envelope
pip install musyn[plot]            # + matplotlib for examples
pip install musyn[all]             # everything + dev tools
```

Only `numpy`, `scipy`, and `joblib` are hard dependencies — everything else
above is an optional extra. Since `select_synergy_number`'s default
`method='min'` needs a wavelet-based AIC curve, most users will want
`musyn[wavelet]`.

A C compiler is required to build the Cython extension (100–500× speedup on the
adaptive envelope loop). If unavailable the package falls back to Numba JIT or
NumPy automatically.

---

## Quick Start

```python
import numpy as np
import musyn
from musyn.preprocessing import condition_emg, linear_envelope

# --- Preprocessing (raw EMG, shape (n_samples, n_channels)) ---
emg_clean = condition_emg(raw_emg, fs=2000.0)             # bandpass + notch
envelope  = linear_envelope(emg_clean, fs=2000.0)         # rectify + lowpass

# or, for adaptive envelope (Ranaldi 2018):
envelope_adaptive = musyn.extract_envelope(raw_emg.T, fs=2000.0)  # (M,N) → (M,N)

# --- Synergy number selection (Ranaldi 2021) ---
# envelope for analysis: shape (n_muscles, n_samples)
D = envelope.T                         # (n_muscles, n_samples)
k = musyn.select_synergy_number(D, events=[0, 1200, 2400, 3600])

# --- Synergy extraction (Soomro 2018) ---
W, C, info = musyn.extract_synergies(D, n_synergies=k)

# --- Quality metrics ---
print(f"VAF  = {musyn.vaf(D, W, C):.3f}")
print(f"R²   = {musyn.r_squared(D, W, C):.3f}")
```

---

## API Reference

### musyn.extract_envelope

```python
envelope = musyn.extract_envelope(signal, fs=1000.0)
```

Adaptive sEMG envelope extraction (Ranaldi et al. 2018). Uses an AR(12)
prewhitening step followed by an adaptive window ν-order detector.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `signal` | array (N,) or (M, N) | — | Raw sEMG, 1-D or multi-channel |
| `fs` | float | — | Sampling frequency in Hz |
| `alpha` | float | 1.0 | Shape parameter |
| `nu` | int | 2 | Detection order |
| `n_jobs` | int | -1 | Parallel channels (-1 = all CPUs) |

Returns `envelope` with the same shape as `signal`, and an `info` dict
reporting the active backend (`'cython'`, `'numba'`, or `'numpy'`).

---

### musyn.extract_synergies

```python
W, C, info = musyn.extract_synergies(envelope, n_synergies=4)
```

NMF-based muscle synergy extraction (Soomro et al. 2018).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `envelope` | array (M, N) | — | Non-negative envelope matrix |
| `n_synergies` | int | — | Number of synergies k |
| `init` | str | `'sparse'` | Initialisation: `'rand'`, `'nsvd'`, `'sparse'` |
| `n_runs` | int | 10 | NMF restarts (best solution returned) |
| `max_iter` | int | 5000 | Max multiplicative-update iterations |
| `tol` | float | 1e-4 | Convergence tolerance |

Returns:
- `W`: synergy matrix, shape (M, k)
- `C`: activation matrix, shape (k, N)
- `info`: dict with `'vaf'`, `'r_squared'`, `'n_iter'`, `'converged'`

---

### musyn.select_synergy_number

```python
k = musyn.select_synergy_number(envelope, events=[0, 1200, 2400])
```

AIC-based synergy number selection (Ranaldi et al. 2021).

**AIC formula:**

```
AIC(k) = L(k) + 2·k·N_M + 2·Σᵢ DoFᵢ(k)
```

where `L(k)` is the per-channel squared reconstruction error normalised by
signal variance + per-channel reconstruction noise, and `DoFᵢ` is the
Daubechies-5 wavelet-based effective degrees of freedom for synergy i.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `M_matrix` | array (M, N) | — | Envelope matrix |
| `k_range` | range or list | `range(1, M+1)` | Synergy counts to evaluate |
| `method` | str | `'min'` | Selection criterion (see table below) |
| `events` | array-like of int | None | Segment-boundary indices (strongly recommended) |
| `n_runs` | int | 5 | NMF restarts per k |
| `wavelet` | str | `'db5'` | Wavelet for DoF estimation |
| `n_jobs` | int | -1 | Parallel jobs for k-sweep |
| `return_full` | bool | False | Return full result dict |
| `seed` | int | None | Random seed |

#### Selection methods

| `method` | Input | Description |
|----------|-------|-------------|
| `'min'` | AIC curve | Global minimum (default, paper recommendation) |
| `'der'` | AIC curve | First stationary point of normalised derivative |
| `'firstpeak'` | AIC curve | First local minimum |
| `'lastpeak'` | AIC curve | Last local minimum |
| `'vaf'` | solutions | First k where VAF ≥ 0.97 |
| `'r2'` | solutions | First k where R² ≥ 0.95 |
| `'plateau'` | solutions | First k where ΔVAF ≤ 5% (N_5%) |
| `'surrogate'` | solutions | Surrogate-data baseline comparison (N_SURR) |

When `return_full=True`, returns a dict with keys `k_opt`, `aic_values`,
`k_range`, `solutions`, `method`.

---

### musyn.preprocessing

Standard sEMG preprocessing utilities. All functions use (n_samples, n_channels)
layout (rows = time).

```python
from musyn.preprocessing import (
    condition_emg,
    linear_envelope,
    remove_ecg_scica,
    segment_envelope,
    time_normalize_envelope,
    normalize_envelope,
)
```

#### condition_emg

```python
emg_clean = condition_emg(data, fs, ecg_channels=None, l_qrs=0.004)
```

Bandpass filter [20–450 Hz] + notch filters at 50–400 Hz harmonics (8 harmonics,
±0.5 Hz each), followed by optional per-channel ECG removal via SC-ICA.

| Argument | Default | Description |
|----------|---------|-------------|
| `ecg_channels` | `None` | List of channel indices for ECG removal |
| `l_qrs` | 0.004 s | QRS duration for ICA embedding window |

#### linear_envelope

```python
env = linear_envelope(data, fs, fc=2.0)
```

Full-wave rectification + 3rd-order Butterworth low-pass at `fc` Hz. Use
`musyn.extract_envelope` for the adaptive algorithm.

#### remove_ecg_scica

```python
clean = remove_ecg_scica(signal, fs=1000, embedding_dim=40)
```

Single-channel ICA ECG removal for a 1-D signal. Uses time-delay (Hankel)
embedding, FastICA decomposition, kurtosis-based ECG component identification,
and diagonal-averaging reconstruction. Requires `scikit-learn`.

Rule of thumb: `embedding_dim / fs` should cover one QRS complex (4–20 ms).

#### segment_envelope

```python
segments, maxima = segment_envelope(data, events, delay_on=500, delay_off=500)
```

Concatenate envelope segments around (start, stop) event pairs with optional
pre/post padding. Returns concatenated array and per-trial maxima.

#### time_normalize_envelope

```python
data_norm, maxima, boundaries = time_normalize_envelope(
    data, events, n_samples=100, delay_on=500, delay_off=500
)
```

Interpolate each trial to a fixed number of samples, enabling cross-trial
averaging for variable-duration movements.

#### normalize_envelope

```python
env_norm = normalize_envelope(env, maxima, percentile=80, floor=1e-4)
```

Amplitude-normalise by the 80th percentile of per-trial maxima. Floors values
at `1e-4` to avoid numerical issues with log-based computations.

---

### Quality metrics

```python
vaf_val = musyn.vaf(D, W, C)          # Variance Accounted For
r2_val  = musyn.r_squared(D, W, C)    # Mean per-channel R²
qr      = musyn.quality_ratio(W, W_ref)  # Synergy quality ratio (0–1)
```

---

## Performance

### Adaptive envelope (hot loop)

| Backend | Speedup | How to get it |
|---------|---------|---------------|
| Cython C extension | 100–500× | C compiler at `pip install` |
| Numba JIT | 100–200× | `pip install musyn[jit]` |
| NumPy vectorized | 5–20× | always available |

The active backend is reported in the `info` dict returned by `extract_envelope`.

### Parallelism

- Multi-channel envelope: `joblib.Parallel(n_jobs=-1)` over channels.
- AIC k-sweep: `joblib.Parallel(n_jobs=-1)` over synergy counts.

---

## Package structure

```
src/musyn/
  envelope/           Algorithm 1 — adaptive sEMG envelope
    _loop.pyx           Cython C extension (primary backend)
    _numba_loop.py      Numba JIT fallback
    _numpy_loop.py      NumPy fallback (always available)
    prewhiten.py        AR(12) Yule-Walker prewhitening
    detection.py        nu-order detector, window initialisation
    adaptive.py         backend dispatcher + iterative loop
    api.py              extract_envelope()

  decomposition/      Algorithm 2 — NMF synergy extraction
    init_strategies.py  RAND / NSVD / SPARSE initialisation
    updates.py          multiplicative update rules
    nnmf.py             NMF loop + multi-run
    api.py              extract_synergies()

  selection/          Algorithm 3 — AIC synergy number selection
    noise.py            noise estimation utilities (MAD + SDN)
    wavelet_dof.py      Daubechies-5 DWT effective DoF (PyWavelets)
    aic.py              likelihood + AIC per k (parallel)
    criteria.py         8 selection criteria
    api.py              select_synergy_number()

  preprocessing/      sEMG signal conditioning
    filtering.py        condition_emg(), linear_envelope()
    ecg.py              remove_ecg_scica() (SC-ICA, requires sklearn)
    segmentation.py     segment / time-normalise / amplitude-normalise

  metrics/quality.py  QR, VAF, R², cosine_similarity
  io/                 load_csv / load_mat / load_npz / save_npz
  utils/              validation, Numba support, type aliases

tests/                pytest suite
examples/             standalone scripts + synthetic sample data
```

---

## Development

```bash
# Install in editable mode (builds Cython extension if possible)
pip install -e ".[all]"

# Generate synthetic sample data (run once)
python examples/generate_sample_data.py

# Run tests
pytest tests/

# Run examples
python examples/01_basic_envelope.py
python examples/03_synergy_number_selection.py
```

---

## MATLAB equivalences

| MATLAB (SmRanaldi repos) | musyn |
|--------------------------|-------|
| `loopFunction.c` (MEX) | `envelope/_loop.pyx` (Cython) |
| `posAutoCorr.c` (MEX) | `scipy.signal.correlate` |
| `whiteningSignal.c` (MEX) | `scipy.linalg.solve_toeplitz` + `scipy.signal.lfilter` |
| `chiTable.mat` (17 MB) | `scipy.stats.chi2.ppf` (exact, no file) |
| `DoFWaveletEvents.m` | `selection/wavelet_dof.py` |
| `findNSynAIC.m` | `selection/criteria.py` |
| `nSyn5Perc.m` | `criteria.select_plateau` |
| `nSynRand.m` | `criteria.select_surrogate` |
| `loopFunction.c` / SC-ICA | `preprocessing/ecg.py` |

---

## References

- Ranaldi S, De Marchis C, Conforto S (2018) *An automatic, adaptive,
  information-based algorithm for the extraction of the sEMG envelope.*
  J Electromyogr Kinesiol.
- Soomro MH, Conforto S, Giunta G, Ranaldi S, De Marchis C (2018)
  *Comparison of Initialization Techniques for the Accurate Extraction of
  Muscle Synergies from Myoelectric Signals via Nonnegative Matrix
  Factorization.*
  Applied Bionics and Biomechanics.
- Ranaldi S, De Marchis C, Severini G, Conforto S (2021)
  *An Objective, Information-Based Approach for Selecting the Number of
  Muscle Synergies to be Extracted via Non-Negative Matrix Factorization.*
  IEEE Trans Neural Syst Rehabil Eng.

If musyn contributes to your research, please cite it — see
[`CITATION.cff`](CITATION.cff) (also used by GitHub's "Cite this repository"
button).

## Contributing

Bug reports, feature requests, and pull requests are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, test conventions
(synthetic data only — no real subject EMG), and code style. This project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT © Simone Ranaldi
