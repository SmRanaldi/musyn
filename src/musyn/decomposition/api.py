"""
High-level API for NMF-based muscle synergy extraction.

References
----------
Soomro et al. (2018) "Comparison of Initialization Techniques for Accurate
Extraction of Muscle Synergies from Myoelectric Signals via NNMF."
Applied Bionics and Biomechanics.
"""
from __future__ import annotations

import numpy as np

from musyn.decomposition.init_strategies import InitStrategy
from musyn.decomposition.nnmf import run_nnmf_multi
from musyn.utils.validation import check_emg_matrix, check_n_synergies


def extract_synergies(
    D: np.ndarray | list,
    n_synergies: int,
    init: InitStrategy = "sparse",
    n_runs: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-4,
    normalize_W: bool = True,
    seed=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract muscle synergies via Non-negative Matrix Factorization.

    Factors the envelope matrix D ≈ W @ C where W (synergy weights)
    and C (activation coefficients) are both non-negative. Uses the
    SPARSE initialization strategy by default, which outperforms RAND
    and NSVD across all muscle-correlation levels (Soomro et al. 2018).

    Parameters
    ----------
    D : array-like, shape (M x N)
        Non-negative EMG envelope matrix. M = number of muscles,
        N = number of time samples. Use ``extract_envelope`` first
        to obtain a non-negative envelope from raw sEMG.
    n_synergies : int
        Number of synergies k to extract (1 ≤ k ≤ min(M, N)).
        Use ``select_synergy_number`` to determine k objectively.
    init : {'rand', 'nsvd', 'sparse'}
        Initialization strategy. 'sparse' (default) is empirically best
        (Soomro et al. 2018).
    n_runs : int
        Number of independent NMF restarts; the best solution
        (minimum Frobenius reconstruction error) is returned.
        Default 10.
    max_iter : int
        Maximum multiplicative update iterations per run. Default 1000.
    tol : float
        Convergence threshold on relative error change. Default 1e-4.
    normalize_W : bool
        If True, normalize each column of W to unit L1-norm and rescale
        C rows accordingly. This is the standard convention and makes
        W columns comparable across experiments. Default True.
    seed : int, np.random.Generator, or None
        Random seed for reproducibility.

    Returns
    -------
    W : np.ndarray, shape (M x k)
        Synergy weight matrix. Each column is a synergy vector.
    C : np.ndarray, shape (k x N)
        Synergy activation time courses. Each row is an activation signal.
    info : dict
        ``n_iter``, ``converged``, ``final_error``, ``error_history``,
        ``n_runs``, ``best_run``.

    References
    ----------
    Soomro MH, Conforto S, Giunta G, Ranaldi S, De Marchis C (2018)
    "Comparison of Initialization Techniques for the Accurate Extraction
    of Muscle Synergies from Myoelectric Signals via Nonnegative Matrix
    Factorization."
    Applied Bionics and Biomechanics.

    Examples
    --------
    >>> import numpy as np
    >>> import musyn
    >>> rng = np.random.default_rng(0)
    >>> D = np.abs(rng.normal(size=(8, 5000)))  # 8 muscles, 5000 samples
    >>> W, C, info = musyn.extract_synergies(D, n_synergies=4)
    >>> W.shape, C.shape
    ((8, 4), (4, 5000))
    """
    D = check_emg_matrix(D)
    check_n_synergies(n_synergies, D.shape[0], D.shape[1])

    W, C, info = run_nnmf_multi(
        D, n_synergies,
        n_runs=n_runs, init=init,
        max_iter=max_iter, tol=tol, rng=seed,
    )

    if normalize_W:
        col_norms = np.sum(np.abs(W), axis=0)
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        W = W / col_norms
        C = C * col_norms[:, np.newaxis]

    return W, C, info
