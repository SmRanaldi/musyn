# musyn — Muscle Synergy Analysis Python Package

## Project Overview

Python package implementing three algorithms from Simone Ranaldi's published
research on surface EMG (sEMG) amplitude and synergy analysis:

| # | Algorithm | Paper | File |
|---|-----------|-------|------|
| 1 | Adaptive sEMG envelope extraction | Ranaldi et al. 2018 | `biblio/18-envelope-paper.pdf` |
| 2 | NMF synergy extraction (sparse init) | Soomro et al. 2018 | `biblio/18-nmf-abb.pdf` |
| 3 | AIC-based synergy number selection | Ranaldi et al. 2021 | `biblio/21-aic-tnsre.pdf` |

The prior MATLAB and C implementations are at:
- https://github.com/SmRanaldi/NSyn_Criteria (NMF + AIC)
- https://github.com/SmRanaldi/EMG_envelope (adaptive envelope)

## Repository Layout (src-layout)

```
src/musyn/
  __init__.py            ← public API: extract_envelope, extract_synergies,
  _version.py              select_synergy_number, quality_ratio, vaf, r_squared
  envelope/
    prewhiten.py         ← AR(12) Yule-Walker, whitening filter
    detection.py         ← nu-order detection, window init
    adaptive.py          ← backend dispatcher (Cython > Numba > NumPy)
    _loop.pyx            ← Cython C extension (compiled at install)
    _numba_loop.py       ← Numba JIT fallback
    _numpy_loop.py       ← NumPy vectorized fallback
    api.py               ← extract_envelope() with joblib parallel channels
  decomposition/
    init_strategies.py   ← RAND, NSVD, SPARSE initializations
    updates.py           ← multiplicative update rules (W, C)
    nnmf.py              ← NMF loop + multi-run
    api.py               ← extract_synergies()
  selection/
    noise.py             ← σ²_M (MAD) + σ²_SDN estimation
    wavelet_dof.py       ← Daubechies-5 DWT, DoF computation (pywt)
    aic.py               ← modified log-likelihood, AIC per k (joblib parallel)
    criteria.py          ← 8 selection criteria (min/der/firstpeak/lastpeak/
                           vaf/r2/plateau/surrogate)
    api.py               ← select_synergy_number()
  io/
    loaders.py           ← load_csv(), load_mat(), load_npz()
    writers.py           ← save_npz(), save_results()
  metrics/
    quality.py           ← QR, VAF, R², cosine_similarity
  utils/
    validation.py        ← check_signal_1d, check_emg_matrix, check_n_synergies
    numba_support.py     ← conditional Numba import + fallback decorators
    typing.py            ← type aliases

tests/
  conftest.py            ← rng, synthetic_emg_1d, synthetic_emg_matrix fixtures
  test_envelope/
  test_decomposition/
  test_selection/
  test_metrics/

examples/
  generate_sample_data.py   ← run once to create examples/data/sample_emg.npz
  01_basic_envelope.py
  02_synergy_extraction.py
  03_synergy_number_selection.py
  04_full_pipeline.py
  data/
    sample_emg.npz           ← synthetic 8-channel, 4000-sample, k=3 dataset
```

## Development Setup

```bash
# Clone and install in editable mode (builds Cython extension if available)
pip install -e ".[all]"

# Generate sample data for examples (run once)
python examples/generate_sample_data.py

# Run tests
pytest tests/

# Run example pipeline
python examples/04_full_pipeline.py
```

## Key Design Decisions

### Performance (three-tier, envelope hot loop)
| Backend | Speedup | Availability |
|---------|---------|--------------|
| Cython C extension (`_loop.pyx`) | ~100-500x | compiled at `pip install` |
| Numba JIT (`_numba_loop.py`) | ~100-200x | `pip install musyn[jit]` |
| NumPy vectorized (`_numpy_loop.py`) | ~5-20x | always |

The active backend is reported in the `info` dict returned by `extract_envelope`.

### Parallelism
- **Multi-channel envelope**: `joblib.Parallel(n_jobs=-1)` in `envelope/api.py`
- **NMF multi-run**: sequential in `nnmf.py` (fast enough per run)
- **AIC k-sweep**: `joblib.Parallel` in `selection/aic.py`

### MATLAB replacements
| MATLAB file | Python equivalent |
|-------------|------------------|
| `loopFunction.c` (MEX) | `_loop.pyx` (Cython) or `_numba_loop.py` |
| `posAutoCorr.c` (MEX) | `scipy.signal.correlate` |
| `whiteningSignal.c` (MEX) | `scipy.linalg.solve_toeplitz` + `scipy.signal.lfilter` |
| `chiTable.mat` (17 MB) | `scipy.stats.chi2.ppf` |
| `findNSynAIC.m` | `selection/criteria.py` |
| `DoFWaveletEvents.m` | `selection/wavelet_dof.py` |

### Caveats
- Cython extension requires a C compiler at install time (`gcc`, `clang`, or MSVC).
  If compilation fails, the package installs without it and falls back.
- `select_synergy_number` requires `PyWavelets` (`pip install musyn[wavelet]`) **only**
  for the AIC-curve-based methods (`'min'`, `'der'`, `'firstpeak'`, `'lastpeak'`) as of
  2026-09-16 (see below) -- `'vaf'`/`'r2'`/`'plateau'`/`'surrogate'` no longer touch
  `wavelet_dof.py` at all, so PyWavelets is no longer a hard dependency for those.
- `biblio/` is excluded from git (large PDFs). Papers are on the local machine only.

### `selection/aic.py` + `selection/api.py` (2026-09-16, on request from risveglio-analysis)
`select_synergy_number` used to call `aic_curve` unconditionally, which computed the
expensive wavelet-based DoF term (`compute_total_dof`, one call per k in `k_range`) even
when `method` was `'vaf'`/`'r2'`/`'plateau'`/`'surrogate'` -- methods whose
`select_synergy_count` branch never reads `aic_values` at all (only the NMF `solutions`).
Wasteful for any caller using the fast VAF path, which is the default for both
`risveglio-core`'s `compute_synergies` and `risveglio-analysis/build_metrics_database.py`.
Fixed by adding a `compute_dof: bool` flag to `compute_aic_for_k`/`aic_curve` (still runs
`run_nnmf_multi`, always needed; skips `compute_likelihood` + `compute_total_dof` when
False, returning `aic_k=None`) and having `select_synergy_number` pass
`compute_dof=True` only for the four AIC-curve methods (`_NEEDS_AIC_CURVE` in `api.py`).
`return_full=True` now comes back with `aic_values=None` for the other four methods --
document this if anything downstream inspects that key unconditionally. Verified with a
synthetic mock test (`compute_total_dof` call-count 0 under `'vaf'`, 1-per-k under
`'min'`) rather than real EMG data, since this is a pure control-flow fix with no change
to any method's actual selection math.

### `preprocessing/filtering.py` (2026-09-05, on request from risveglio-suite)
`condition_emg`'s bandpass (default 20-450 Hz) and mains-hum notch (default 50 Hz x8
harmonics) stages, previously hardcoded on with no way to change or disable either, are
now configurable via `apply_bandpass`/`bandpass_range`/`apply_notch`/
`notch_fundamental_hz`/`notch_harmonics` -- all default to the exact original fixed
behaviour, so existing callers (the sibling EMG-synergy projects) are unaffected.
Driven by risveglio-suite wanting a user-facing EMG-filtering settings tab (clinicians
deciding per-recording whether to apply notch/bandpass and at what cutoffs), not an
internal need of this package. Added `tests/test_preprocessing/test_filtering.py` --
this module had no tests at all before. One thing worth knowing if touching this again:
a single narrow (+/-0.5 Hz) `filtfilt` notch's edge transients dominate a short test
signal's overall amplitude even when the tone is genuinely well-attenuated in steady
state -- tests measure amplitude over the middle of a 12000-sample signal, not the
first/last ~1000 samples, to avoid false failures from this.

## Algorithm Parameters (defaults matching papers)

| Parameter | Value | Location |
|-----------|-------|----------|
| AR order | 12 | `envelope/api.py` → `prewhiten` |
| α (shape) | 1.0 | `extract_envelope` |
| ν (detection order) | 2 | `extract_envelope` |
| init window | 100 ms | `detection.initialize_window_lengths` |
| NMF init | `'sparse'` | `extract_synergies` |
| NMF runs | 10 | `extract_synergies` |
| AIC wavelet | `'db5'` | `select_synergy_number` |
| SDN coefficient c | 0.175 | `select_synergy_number` |
| Default AIC method | `'min'` | `select_synergy_number` |

## Public API

```python
import musyn

# 1. Envelope extraction
env = musyn.extract_envelope(signal, fs=1000.0)          # (N,) → (N,)
env = musyn.extract_envelope(emg_matrix, fs=1000.0)      # (M,N) → (M,N)

# 2. Synergy extraction
W, C, info = musyn.extract_synergies(envelope, n_synergies=4)

# 3. Synergy number selection
k = musyn.select_synergy_number(envelope)
result = musyn.select_synergy_number(envelope, return_full=True)

# 4. Quality metrics
qr  = musyn.quality_ratio(W_extracted, W_true)  # 0-1
vaf_val = musyn.vaf(D, W, C)                    # 0-1
r2  = musyn.r_squared(D, W, C)                  # 0-1
```

## Dependencies

```toml
# Core (always required)
numpy >= 1.24
scipy >= 1.10
joblib >= 1.3

# Optional
PyWavelets >= 1.4   # required for select_synergy_number
numba >= 0.57       # Numba JIT backend for envelope loop
matplotlib >= 3.7   # example plots
```
