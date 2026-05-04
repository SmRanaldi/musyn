"""
Core NMF loop for muscle synergy extraction.

References
----------
Soomro et al. (2018) "Comparison of Initialization Techniques for Accurate
Extraction of Muscle Synergies from Myoelectric Signals via NNMF."
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from musyn.decomposition.init_strategies import InitStrategy, get_init
from musyn.decomposition.updates import update_W, update_C, reconstruction_error


def run_nnmf(
    D: np.ndarray,
    n_synergies: int,
    init: InitStrategy = "sparse",
    max_iter: int = 1000,
    tol: float = 1e-4,
    rng=None,
    callback: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run NMF with multiplicative update rules.

    Factors D ≈ W @ C where W ≥ 0 (synergy weights) and C ≥ 0
    (activation coefficients).

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
        Non-negative data matrix (EMG envelope).
    n_synergies : int
        Number of synergies k.
    init : {'rand', 'nsvd', 'sparse'}
        Initialization strategy. 'sparse' is empirically best
        (Soomro et al. 2018).
    max_iter : int
        Maximum update iterations. Default 1000.
    tol : float
        Convergence threshold on relative change of reconstruction
        error. Default 1e-4.
    rng : int, np.random.Generator, or None
        Random seed / generator.
    callback : callable, optional
        Called each iteration as ``callback(iteration, W, C, error)``.

    Returns
    -------
    W : np.ndarray, shape (M, k)
        Synergy weight matrix.
    C : np.ndarray, shape (k, N)
        Synergy activation matrix.
    info : dict
        ``n_iter`` (int), ``converged`` (bool),
        ``final_error`` (float), ``error_history`` (list[float]).

    Notes
    -----
    Update rules (Soomro et al. 2018, Eq. 2-3):
      W ← W * (D C^T) / (W C C^T + ε)
      C ← C * (W^T D) / (W^T W C + ε)
    """
    W, C = get_init(init, D, n_synergies, rng)

    error_history = []
    prev_error = reconstruction_error(D, W, C)
    error_history.append(prev_error)
    converged = False

    for it in range(max_iter):
        W = update_W(D, W, C)
        C = update_C(D, W, C)
        error = reconstruction_error(D, W, C)
        error_history.append(error)

        if callback is not None:
            callback(it, W, C, error)

        if prev_error > 0 and abs(prev_error - error) / prev_error < tol:
            converged = True
            break
        prev_error = error

    return W, C, {
        "n_iter": it + 1,
        "converged": converged,
        "final_error": float(error_history[-1]),
        "error_history": error_history,
    }


def run_nnmf_multi(
    D: np.ndarray,
    n_synergies: int,
    n_runs: int = 10,
    init: InitStrategy = "sparse",
    max_iter: int = 1000,
    tol: float = 1e-4,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run NMF ``n_runs`` times and return the solution with minimum error.

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
    n_synergies : int
    n_runs : int
        Number of independent restarts. Default 10.
    init : InitStrategy
        Initialization strategy.
    max_iter : int
    tol : float
    rng : int, np.random.Generator, or None
        Master seed; each run gets an independent child generator.

    Returns
    -------
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)
    info : dict
        Same as ``run_nnmf`` plus ``n_runs`` and ``best_run`` index.
    """
    master_rng = np.random.default_rng(rng)
    seeds = master_rng.integers(0, 2**31, size=n_runs)

    best_error = np.inf
    best_W = best_C = best_info = None

    for run_idx, seed in enumerate(seeds):
        W, C, info = run_nnmf(D, n_synergies, init=init, max_iter=max_iter,
                               tol=tol, rng=int(seed))
        if info["final_error"] < best_error:
            best_error = info["final_error"]
            best_W, best_C, best_info = W, C, info
            best_info["best_run"] = run_idx

    best_info["n_runs"] = n_runs
    return best_W, best_C, best_info
