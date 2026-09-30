# Adaptive envelope extraction

Implements Ranaldi, De Marchis & Conforto (2018), *"An automatic, adaptive,
information-based algorithm for the extraction of the sEMG envelope"*
(J Electromyogr Kinesiol) — replacing the MATLAB/MEX implementation in
[SmRanaldi/EMG_envelope](https://github.com/SmRanaldi/EMG_envelope).

## Why adaptive?

A fixed low-pass filter (as in {func}`musyn.preprocessing.linear_envelope`)
trades off temporal resolution against smoothness with one global cutoff.
The adaptive algorithm instead estimates a **different window length at
every sample**, widening the window where the signal is stationary (for
smoother, lower-variance estimates) and narrowing it during fast
transitions (to preserve onset/offset timing). This removes the need to
hand-tune a cutoff frequency per recording or per muscle.

## Pipeline

{func}`musyn.extract_envelope` runs four stages per channel (see
{doc}`../api/internals/envelope` for the function-level reference):

1. **Pre-whitening** ({func}`musyn.envelope.prewhiten.prewhiten`) — an
   AR(12) model, fit via Yule-Walker on the raw signal, is used as a
   prediction-error (whitening) filter. This replaces the MATLAB MEX
   functions `posAutoCorr.c` and `whiteningSignal.c`. The paper uses AR
   order 11; musyn defaults to 12 (`ar_order`).
2. **Nu-order detection** ({func}`musyn.envelope.detection.nu_order_detection`)
   — the whitened signal is rectified and raised to the power `nu` (paper
   fixes `nu=2`, i.e. squaring).
3. **Window initialization**
   ({func}`musyn.envelope.detection.initialize_window_lengths`) — every
   sample starts with the same window length, `init_ms` milliseconds
   (500 ms by default in {func}`musyn.extract_envelope`; the underlying
   `initialize_window_lengths` function itself defaults to 100 ms, matching
   the paper, when called directly), clipped to `[w_min, w_max]`.
4. **Adaptive iteration** ({func}`musyn.envelope.adaptive.adaptive_envelope`)
   — at each iteration, per-sample derivatives are used to re-estimate an
   optimal window length `M_k` for every sample; the loop stops early once
   `convergence_threshold` (default 95%) of samples have converged, or after
   `max_iter` iterations.

For multi-channel input (`signal.shape == (M, N)`), channels are processed
independently and in parallel via `joblib.Parallel(n_jobs=n_jobs)`.

## Backends

The per-sample iteration in step 4 is the hot loop, and the reason this
package exists as a compiled extension rather than pure Python. Three
backends implement the identical algorithm at different speeds:

| Backend | Speedup vs. naive Python | Requires |
|---------|--------------------------|----------|
| Cython (`envelope/_loop.pyx`) | ~100–500× | a C compiler at `pip install` time |
| Numba JIT (`envelope/_numba_loop.py`) | ~100–200× | `pip install musyn[jit]` |
| NumPy vectorized (`envelope/_numpy_loop.py`) | ~5–20× | always available |

`extract_envelope` picks the fastest available backend automatically at
import time (see {func}`musyn.envelope.adaptive.backend`) — Cython first,
then Numba, then NumPy. Pass `return_info=True` to see which one ran:

```python
env, info = musyn.extract_envelope(signal, fs=1000.0, return_info=True)
print(info[0]["backend"])         # 'cython', 'numba', or 'numpy'
print(info[0]["converged"])       # bool
print(info[0]["window_lengths"])  # (N,) — optimal M_k per sample
```

If the compiled Cython extension is from an older, incompatible build, musyn
falls back to the Numba/NumPy backend automatically rather than crashing —
see the `try/except` around `_cython_loop` in
{func}`musyn.envelope.adaptive.adaptive_envelope`.

## Key parameters

| Parameter | Paper value | Default | Notes |
|-----------|-------------|---------|-------|
| `ar_order` | 11 | 12 | AR prewhitening order |
| `alpha` | 1.0 | 1.0 | Signal model shape parameter, fixed in the paper |
| `nu` | 2 | 2 | Detection order, fixed in the paper |
| `init_ms` | 100 ms | 500 ms | Initial adaptive window length |
| `convergence_threshold` | — | 0.95 | Fraction of converged samples to stop early |
| `max_iter` | — | 100 | Hard cap on iterations |

`alpha` and `nu` are exposed as parameters but the paper fixes both — change
them only if you're deliberately deviating from the published algorithm.

## When to use `linear_envelope` instead

{func}`musyn.preprocessing.linear_envelope` (rectify + Butterworth low-pass)
is a reasonable, much cheaper baseline when you don't need the paper's
adaptive behavior, or want a quick sanity check before running the full
adaptive pipeline.
