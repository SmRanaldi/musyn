"""
AIC computation for muscle synergy model order selection.

Matches the reference implementation (pysyn.py, UCD-UNIFE 2026).

Likelihood:
  L = Σ_i [ Σ_t (rec_it - M_it)² / (var(M) + std(rec_i)) ]

AIC:
  AIC(k) = L + 2·k·N_M + 2·Σ_i DoF_i

References
----------
Ranaldi et al. (2021) "An Objective, Information-Based Approach for Selecting
the Number of Muscle Synergies via NNMF."
IEEE Trans Neural Syst Rehabil Eng.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from musyn.decomposition.nnmf import run_nnmf_multi
from musyn.decomposition.init_strategies import InitStrategy
from musyn.selection.wavelet_dof import compute_total_dof


def compute_likelihood(M: np.ndarray, rec: np.ndarray) -> float:
    """
    Reconstruction loss normalised by signal + model noise.

    For each channel i:
      L_i = Σ_t (rec_it - M_it)² / (var(M) + std(rec_i))

    Parameters
    ----------
    M : np.ndarray, shape (n_muscles, n_samples)
        Observed envelope matrix.
    rec : np.ndarray, shape (n_muscles, n_samples)
        NMF reconstruction W @ C.

    Returns
    -------
    L : float
    """
    global_var = float(np.var(M))
    total = 0.0
    for i in range(M.shape[0]):
        channel_noise = global_var + float(np.std(rec[i]))
        total += float(np.sum((rec[i] - M[i]) ** 2) / channel_noise)
    return total


def compute_aic_for_k(
    M_matrix: np.ndarray,
    k: int,
    n_muscles: int,
    wavelet: str = "db5",
    n_runs: int = 5,
    init: InitStrategy = "sparse",
    events: np.ndarray | None = None,
    rng=None,
    compute_dof: bool = True,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """
    Compute AIC for a single synergy count k.

    AIC(k) = L + 2·k·N_M + 2·Σ_i DoF_i

    Parameters
    ----------
    M_matrix : np.ndarray, shape (n_muscles, n_samples)
    k : int
    n_muscles : int
    wavelet : str
    n_runs : int
        NMF restarts (best solution used).
    init : InitStrategy
    events : 1D array-like of int, optional
        Segment-boundary indices for per-cycle DoF estimation.
    rng : seed
    compute_dof : bool
        If False, skip the likelihood and wavelet-DoF terms (the AIC-specific,
        expensive part -- see ``compute_total_dof``) and return ``aic_k=None``.
        Set False by ``select_synergy_number`` when ``method`` doesn't need an
        AIC curve at all (e.g. 'vaf'), since the NMF fit (W, C) is still needed
        by every method but the DoF term only by AIC-curve-based ones.

    Returns
    -------
    aic_k : float or None
        None when ``compute_dof=False``.
    W : np.ndarray, shape (n_muscles, k)
    C : np.ndarray, shape (k, n_samples)
    """
    W, C, _ = run_nnmf_multi(M_matrix, k, n_runs=n_runs, init=init, rng=rng)
    if not compute_dof:
        return None, W, C
    rec = W @ C
    L = compute_likelihood(M_matrix, rec)
    total_dof = compute_total_dof(C, wavelet=wavelet, events=events)
    return L + 2.0 * k * n_muscles + 2.0 * total_dof, W, C


def aic_curve(
    M_matrix: np.ndarray,
    k_range: list[int],
    wavelet: str = "db5",
    n_runs: int = 5,
    init: InitStrategy = "sparse",
    events: np.ndarray | None = None,
    n_jobs: int = -1,
    rng=None,
    compute_dof: bool = True,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Compute AIC for each k in k_range (parallelised).

    Parameters
    ----------
    M_matrix : np.ndarray, shape (n_muscles, n_samples)
    k_range : list[int]
    wavelet : str
    n_runs : int
    init : InitStrategy
    events : 1D array-like of int, optional
    n_jobs : int
    rng : seed
    compute_dof : bool
        If False, every k skips the likelihood + wavelet-DoF terms (see
        ``compute_aic_for_k``) and ``aic_values`` comes back as None -- only
        the NMF solutions are computed. Use when the caller's selection
        method doesn't consume an AIC curve.

    Returns
    -------
    aic_values : np.ndarray, shape (len(k_range),), or None
        None when ``compute_dof=False``.
    solutions : list of (W, C) tuples
    """
    n_muscles = M_matrix.shape[0]
    master_rng = np.random.default_rng(rng)
    seeds = master_rng.integers(0, 2 ** 31, size=len(k_range))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_aic_for_k)(
            M_matrix, k, n_muscles,
            wavelet=wavelet, n_runs=n_runs, init=init,
            events=events, rng=int(seed), compute_dof=compute_dof,
        )
        for k, seed in zip(k_range, seeds)
    )

    aic_values = np.array([r[0] for r in results]) if compute_dof else None
    solutions = [(r[1], r[2]) for r in results]
    return aic_values, solutions
