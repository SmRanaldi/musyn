# AIC-based synergy number selection

Implements Ranaldi, De Marchis, Severini & Conforto (2021), *"An Objective,
Information-Based Approach for Selecting the Number of Muscle Synergies to
be Extracted via Non-Negative Matrix Factorization"* (IEEE TNSRE) —
replacing `findNSynAIC.m`, `DoFWaveletEvents.m`, `nSyn5Perc.m`, and
`nSynRand.m` from
[SmRanaldi/NSyn_Criteria](https://github.com/SmRanaldi/NSyn_Criteria).

## The problem

NMF always reduces reconstruction error as `k` increases — at `k =
n_muscles` the fit is (trivially) near-perfect. Picking `k` by eye from a
VAF or R² curve is subjective and inconsistent across studies. This
algorithm instead builds an information-theoretic criterion that penalizes
model complexity, so the curve has a genuine minimum instead of monotonically
improving.

## The AIC formula

$$
\mathrm{AIC}(k) = L(k) + 2 k N_M + 2 \sum_i \mathrm{DoF}_i(k)
$$

Computed by {func}`musyn.selection.aic.compute_aic_for_k` for each `k`, this
combines three terms:

**1. Likelihood `L(k)`** ({func}`musyn.selection.aic.compute_likelihood`) —
per-channel squared reconstruction error, normalized by signal variance plus
per-channel reconstruction noise:

$$
L(k) = \sum_i \frac{\sum_t (\hat{M}_{i,t} - M_{i,t})^2}{\operatorname{var}(M) + \operatorname{std}(\hat{M}_i)}
$$

**2. Parameter penalty `2 k N_M`** — the number of free parameters in `W`
grows linearly with `k` and the muscle count `N_M`; this term penalizes
larger `k` directly, the classic AIC complexity term.

**3. Wavelet degrees of freedom `DoF_i(k)`**
({func}`musyn.selection.wavelet_dof.compute_total_dof`) — rather than
counting `C`'s raw sample count as "free parameters" (which would
wildly overpenalize smooth activation signals), each synergy's activation
row is decomposed with a Daubechies-5 discrete wavelet transform, and an
*effective* number of degrees of freedom is estimated from how much of the
signal's energy survives at the decorrelation scale. A smoother activation
signal has fewer effective DoF than a noisy one of the same length.

Signal-dependent noise ({func}`musyn.selection.noise`) also feeds into the
NMF fit quality assessment: total variance is modeled as
`σ²_M + σ²_SDN`, where `σ²_M` is a robust (MAD-based) measurement-noise
estimate and `σ²_SDN = (c · local_mean)²` captures the fact that sEMG noise
scales with signal amplitude.

## Why `events` matters

```{important}
Always pass `events` when you have movement-cycle or trial boundaries.
Without them, the wavelet DoF is computed on the *entire* signal as one
block rather than per-cycle, which **underestimates** the DoF penalty and
can suppress the ascending phase of the AIC curve — biasing `'min'` toward
too large a `k`.
```

```python
# Four movement cycles of 1200 samples each
k = musyn.select_synergy_number(envelope, events=[0, 1200, 2400, 3600, 4800])
```

`events` are segment-boundary sample indices; DoF is computed independently
per segment and **summed**
({func}`musyn.selection.wavelet_dof.compute_wavelet_dof`).

## Choosing a `method`

Eight criteria are available, split into two families:

**AIC-curve-based** (need `musyn[wavelet]`; build the full AIC curve above):

| `method` | Rule |
|----------|------|
| `'min'` (default) | Global minimum of the AIC curve — the paper's recommended criterion |
| `'der'` | First point where the normalized AIC derivative crosses a plateau threshold |
| `'firstpeak'` | First local minimum |
| `'lastpeak'` | Last local minimum |

**Solutions-based** (only need the NMF fits, not the AIC curve — work
without `musyn[wavelet]`, and are skipped entirely when not selected, see
below):

| `method` | Rule |
|----------|------|
| `'vaf'` | First `k` with global VAF ≥ 0.97 |
| `'r2'` | First `k` with mean per-channel R² ≥ 0.95 |
| `'plateau'` | First `k` where the VAF gain from `k` to `k+1` drops to ≤ 5% |
| `'surrogate'` | First `k` where the real VAF gain falls below 75% of the gain expected from column-shuffled (surrogate) data — a data-driven null baseline instead of a fixed threshold |

If you only ever use `'vaf'`, `'r2'`, `'plateau'`, or `'surrogate'`, you
don't need `musyn[wavelet]` at all: `select_synergy_number` detects this and
skips the wavelet DoF computation entirely (see the performance note below),
so `aic_values` comes back as `None` in the `return_full=True` result.

## Inspecting the full result

```python
result = musyn.select_synergy_number(envelope, events=events, return_full=True)
result["k_opt"]       # selected k
result["aic_values"]  # AIC per k, or None for solutions-based methods
result["k_range"]     # the k values evaluated
result["solutions"]   # list of (W, C) per k — reuse these, no need to re-run NMF
```

`result["solutions"]` lets you plot VAF/R² vs. `k` or inspect intermediate
fits without re-running {func}`musyn.extract_synergies`.

## Performance

Every `method` needs an NMF fit at each `k` in `k_range`
({func}`musyn.decomposition.nnmf.run_nnmf_multi`, `n_runs` restarts each) —
that cost is unavoidable. The wavelet-based likelihood/DoF terms, on the
other hand, are only computed for the four AIC-curve methods; `k_range` is
evaluated in parallel via `joblib` (`n_jobs`).

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `method` | `'min'` | See table above |
| `k_range` | `range(1, n_muscles + 1)` | Narrow this to cut runtime if you have a prior expectation |
| `events` | `None` | Strongly recommended — see above |
| `n_runs` | 5 | NMF restarts per `k` (fewer than `extract_synergies`' default of 10, for speed across the whole sweep) |
| `wavelet` | `'db5'` | Daubechies-5, as in the paper |
