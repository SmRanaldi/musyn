"""
Initialization strategies for NMF-based muscle synergy extraction.

Three strategies are implemented (Soomro et al. 2018, Section II-B):
  - RAND: uniform random initialization
  - NSVD: SVD-based nonnegative initialization (Boutsidis & Gallopoulos 2008)
  - SPARSE: sparse initialization (empirically best across all correlation levels)

References
----------
Soomro MH et al. (2018) "Comparison of Initialization Techniques for Accurate
Extraction of Muscle Synergies from Myoelectric Signals via NNMF."
Applied Bionics and Biomechanics.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse.linalg import svds

InitStrategy = Literal["rand", "nsvd", "sparse"]


def _resolve_rng(seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def init_rand(
    n_muscles: int,
    n_samples: int,
    n_synergies: int,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random initialization: W, C ~ Uniform[0, 1].

    Parameters
    ----------
    n_muscles : int
        Number of EMG channels (M).
    n_samples : int
        Number of time samples (N).
    n_synergies : int
        Number of synergies (k).
    rng : int, np.random.Generator, or None
        Random seed or generator.

    Returns
    -------
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Notes
    -----
    Paper: "RAND — W and C initialized from Uniform[0,1]."
    (Soomro et al. 2018, Section II-B).
    """
    rng = _resolve_rng(rng)
    W = rng.uniform(0.0, 1.0, size=(n_muscles, n_synergies))
    C = rng.uniform(0.0, 1.0, size=(n_synergies, n_samples))
    return W, C


def init_nsvd(
    D: np.ndarray,
    n_synergies: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD-based nonnegative initialization (Boutsidis & Gallopoulos 2008).

    Performs a truncated SVD on D, then uses the positive and negative
    parts of each singular vector pair to build nonneg starting matrices.

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
        Data matrix.
    n_synergies : int
        Number of synergies k.

    Returns
    -------
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Notes
    -----
    Paper: "NSVD — SVD-based nonneg initialization (Boutsidis & Gallopoulos)."
    (Soomro et al. 2018, Section II-B).
    Uses ``scipy.sparse.linalg.svds`` for O(M*N*k) truncated SVD.
    """
    k = min(n_synergies, min(D.shape) - 1)
    U, s, Vt = svds(D, k=k)
    # svds returns singular values in ascending order; reverse
    idx = np.argsort(s)[::-1]
    U, s, Vt = U[:, idx], s[idx], Vt[idx, :]

    W = np.zeros((D.shape[0], n_synergies), dtype=np.float64)
    C = np.zeros((n_synergies, D.shape[1]), dtype=np.float64)

    for i in range(min(k, n_synergies)):
        u = U[:, i]
        v = Vt[i, :] * s[i]
        # Positive/negative parts
        u_pos = np.maximum(u, 0.0)
        u_neg = np.maximum(-u, 0.0)
        v_pos = np.maximum(v, 0.0)
        v_neg = np.maximum(-v, 0.0)
        norm_pos = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
        norm_neg = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)
        if norm_pos >= norm_neg:
            W[:, i] = u_pos
            C[i, :] = v_pos
        else:
            W[:, i] = u_neg
            C[i, :] = v_neg

    # Clip any numerical negatives
    W = np.maximum(W, 0.0)
    C = np.maximum(C, 0.0)
    # Add small constant to avoid exact zeros
    eps = np.finfo(float).eps
    W += eps
    C += eps
    return W, C


def init_sparse(
    n_muscles: int,
    n_samples: int,
    n_synergies: int,
    w_low: float = 0.0,
    w_high: float = 0.05,
    peak_low: float = 0.7,
    peak_high: float = 0.8,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse initialization (Soomro et al. 2018) — empirically best strategy.

    W is initialized with near-zero values, with exactly one large element
    per synergy column (the "peak"). C is uniform random.

    Parameters
    ----------
    n_muscles : int
    n_samples : int
    n_synergies : int
    w_low, w_high : float
        Range for the background (near-zero) elements of W.
        Default: [0.0, 0.05].
    peak_low, peak_high : float
        Range for the single large element per column.
        Default: [0.7, 0.8].
    rng : int, np.random.Generator, or None

    Returns
    -------
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Notes
    -----
    Paper: "SPARSE — W ~ Uniform[0, 0.05] with one element per column
    set to Uniform[0.7, 0.8]; C ~ Uniform[0, 1]."
    (Soomro et al. 2018, Section II-B).
    """
    rng = _resolve_rng(rng)
    W = rng.uniform(w_low, w_high, size=(n_muscles, n_synergies))
    # Set one random element per column to the peak range
    peak_rows = rng.integers(0, n_muscles, size=n_synergies)
    peak_vals = rng.uniform(peak_low, peak_high, size=n_synergies)
    for i, (row, val) in enumerate(zip(peak_rows, peak_vals)):
        W[row, i] = val
    C = rng.uniform(0.0, 1.0, size=(n_synergies, n_samples))
    return W, C


def get_init(
    strategy: InitStrategy,
    D: np.ndarray,
    n_synergies: int,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the appropriate initialization function.

    Parameters
    ----------
    strategy : {'rand', 'nsvd', 'sparse'}
    D : np.ndarray, shape (M, N)
    n_synergies : int
    rng : int, np.random.Generator, or None

    Returns
    -------
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)
    """
    M, N = D.shape
    if strategy == "rand":
        return init_rand(M, N, n_synergies, rng)
    elif strategy == "nsvd":
        return init_nsvd(D, n_synergies)
    elif strategy == "sparse":
        return init_sparse(M, N, n_synergies, rng=rng)
    else:
        raise ValueError(
            f"Unknown initialization strategy '{strategy}'. "
            "Choose from 'rand', 'nsvd', 'sparse'."
        )
