"""
Modified AIC computation for muscle synergy model order selection.

Key fix over classical AIC:
  - Uses signal-dependent noise in the log-likelihood (avoids minimum at k=1)
  - Uses wavelet-estimated effective DoF (not simply k*n_muscles + k*n_samples)

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


def log_likelihood(
    M_matrix: np.ndarray,
    M_hat: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Modified log-likelihood under signal-dependent Gaussian noise.

    log L = -0.5 * Σ_ij [ (M_ij - M̂_ij)² / σ²_ij + log(2π σ²_ij) ]

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
        Observed envelope matrix.
    M_hat : np.ndarray, shape (M, N)
        NMF reconstruction W @ C.
    sigma2 : np.ndarray, shape (M, N)
        Total noise variance at each (muscle, time) point.

    Returns
    -------
    log_L : float

    Notes
    -----
    Paper: Eq. (5), Ranaldi et al. (2021).
    """
    safe_sigma2 = np.maximum(sigma2, 1e-12)
    sq_err = (M_matrix - M_hat) ** 2 / safe_sigma2
    log_term = np.log(2.0 * np.pi * safe_sigma2)
    return float(-0.5 * np.sum(sq_err + log_term))


def compute_aic_for_k(
    M_matrix: np.ndarray,
    k: int,
    sigma2: np.ndarray,
    n_muscles: int,
    wavelet: str = "db5",
    n_runs: int = 5,
    init: InitStrategy = "sparse",
    rng=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the modified AIC for a single synergy count k.

    AIC(k) = -log L(M̂) + 2k·N_M + 2·Σ_i N_DoF,i

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
    k : int
        Number of synergies.
    sigma2 : np.ndarray, shape (M, N)
    n_muscles : int
        N_M in the AIC formula.
    wavelet : str
        Wavelet for DoF estimation.
    n_runs : int
        NMF restarts (best solution used).
    init : InitStrategy
    rng : seed

    Returns
    -------
    aic_k : float
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Notes
    -----
    Paper: AIC(k) = -log L(M̂) + 2k·N_M + 2·Σ N_DoF,i (Eq. 4).
    """
    W, C, _ = run_nnmf_multi(
        M_matrix, k, n_runs=n_runs, init=init, rng=rng,
    )
    M_hat = W @ C
    log_L = log_likelihood(M_matrix, M_hat, sigma2)
    total_dof = compute_total_dof(C, wavelet=wavelet)
    aic_k = -log_L + 2.0 * k * n_muscles + 2.0 * total_dof
    return aic_k, W, C


def aic_curve(
    M_matrix: np.ndarray,
    k_range: list[int],
    sigma2: np.ndarray,
    wavelet: str = "db5",
    n_runs: int = 5,
    init: InitStrategy = "sparse",
    n_jobs: int = -1,
    rng=None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Compute modified AIC for each k in k_range (parallelized).

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
    k_range : list[int]
        Synergy counts to evaluate.
    sigma2 : np.ndarray, shape (M, N)
        Total noise variance.
    wavelet : str
        Default 'db5'.
    n_runs : int
        NMF restarts per k.
    init : InitStrategy
    n_jobs : int
        Parallel jobs. -1 = all CPUs.
    rng : seed

    Returns
    -------
    aic_values : np.ndarray, shape (len(k_range),)
    solutions : list of (W, C) tuples
    """
    n_muscles = M_matrix.shape[0]
    master_rng = np.random.default_rng(rng)
    seeds = master_rng.integers(0, 2**31, size=len(k_range))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_aic_for_k)(
            M_matrix, k, sigma2, n_muscles,
            wavelet=wavelet, n_runs=n_runs, init=init, rng=int(seed),
        )
        for k, seed in zip(k_range, seeds)
    )

    aic_values = np.array([r[0] for r in results])
    solutions = [(r[1], r[2]) for r in results]
    return aic_values, solutions
