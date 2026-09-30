# NMF synergy extraction

Implements Soomro, Conforto, Giunta, Ranaldi & De Marchis (2018), *"Comparison
of Initialization Techniques for the Accurate Extraction of Muscle Synergies
from Myoelectric Signals via Nonnegative Matrix Factorization"* (Applied
Bionics and Biomechanics).

## The model

Muscle synergy analysis factors a non-negative envelope matrix `D` (shape
`(n_muscles, n_samples)`) as

$$D \approx W C$$

where `W` (shape `(n_muscles, k)`) is the **synergy weight matrix** — each
column a fixed muscle-weighting pattern — and `C` (shape `(k, n_samples)`)
holds the **activation time courses** — how strongly each synergy is
recruited at each instant. Both `W` and `C` are constrained non-negative,
which is what makes the factorization biologically interpretable (a synergy
can only contribute additively to muscle activity, never subtract from it).

{func}`musyn.extract_synergies` finds `W` and `C` via the multiplicative
update rules of Lee & Seung (1999), iterated to convergence
({func}`musyn.decomposition.updates.update_W`,
{func}`musyn.decomposition.updates.update_C`):

$$
W \leftarrow W \odot \frac{D C^\top}{W C C^\top + \varepsilon}
\qquad\qquad
C \leftarrow C \odot \frac{W^\top D}{W^\top W C + \varepsilon}
$$

These rules provably never increase the reconstruction error
`||D - WC||_F` and keep both matrices non-negative at every step.

## Initialization matters

NMF is non-convex — different starting points converge to different local
optima. Soomro et al. (2018) compare three initialization strategies (see
{doc}`../api/internals/decomposition`):

- **`'rand'`** ({func}`musyn.decomposition.init_strategies.init_rand`) — both
  `W` and `C` drawn `Uniform[0, 1]`. Simple, but no structural prior.
- **`'nsvd'`** ({func}`musyn.decomposition.init_strategies.init_nsvd`) —
  Boutsidis & Gallopoulos' non-negative double SVD: splits each singular
  vector into positive/negative parts and keeps whichever has larger norm.
  Deterministic given `D`.
- **`'sparse'`** ({func}`musyn.decomposition.init_strategies.init_sparse`,
  **default**) — `W` starts near-zero with exactly one large ("peak") entry
  per column, biasing each synergy toward a distinct dominant muscle from
  the start. Empirically the best-performing strategy across all
  muscle-correlation levels tested in the paper, which is why it's the
  package default.

Because results depend on initialization, {func}`musyn.extract_synergies`
runs `n_runs` independent restarts (default 10) and keeps the one with the
lowest final reconstruction error — see
{func}`musyn.decomposition.nnmf.run_nnmf_multi`.

## Choosing `k`

`extract_synergies` requires `n_synergies` up front. Don't guess it by eye —
use {func}`musyn.select_synergy_number` (see {doc}`selection`) to pick `k`
objectively, then pass the result straight through:

```python
k = musyn.select_synergy_number(envelope)
W, C, info = musyn.extract_synergies(envelope, n_synergies=k)
```

## Interpreting the output

- `info["converged"]` — whether the *best* run hit the `tol` threshold
  before `max_iter`.
- `info["final_error"]`, `info["error_history"]` — Frobenius reconstruction
  error trace for the best run
  ({func}`musyn.decomposition.updates.reconstruction_error`).
- `info["best_run"]` — which of the `n_runs` restarts won.

Once you have `W` and `C`, check reconstruction quality with
{func}`musyn.vaf` and {func}`musyn.r_squared`, and — if you have a
ground-truth `W` to compare against (e.g. on synthetic data) — with
{func}`musyn.quality_ratio`.

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `init` | `'sparse'` | Empirically best per Soomro et al. (2018) |
| `n_runs` | 10 | More restarts reduce the risk of a bad local optimum, at linear cost |
| `max_iter` | 1000 | Per-run cap |
| `tol` | 1e-4 | Relative reconstruction-error change to declare convergence |
| `normalize_W` | `True` | L1-normalizes each `W` column, rescaling `C` accordingly, so columns are comparable across runs/subjects |
