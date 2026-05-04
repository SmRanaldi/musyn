"""
Multiplicative update rules for NMF.

Vectorized BLAS-backed operations via NumPy's ``@`` operator.

References
----------
Lee DD, Seung HS (1999) "Learning the parts of objects by non-negative
matrix factorization." Nature.

Soomro et al. (2018), Eq. (2)-(3): multiplicative update rules.
"""
from __future__ import annotations

import numpy as np

_EPS = np.finfo(float).eps


def update_W(
    D: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """
    Multiplicative update for synergy weight matrix W.

    W_ik ← W_ik * (D @ C.T)_ik / (W @ C @ C.T + ε)_ik

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
        Data (envelope) matrix.
    W : np.ndarray, shape (M, k)
        Current synergy weights.
    C : np.ndarray, shape (k, N)
        Current activation coefficients.

    Returns
    -------
    W_new : np.ndarray, shape (M, k)
        Updated synergy weights (nonneg guaranteed).

    Notes
    -----
    Paper notation: W (Eq. 2, Soomro et al. 2018).
    ε = machine epsilon prevents division by zero (Lee & Seung stability).
    """
    numerator = D @ C.T
    denominator = W @ (C @ C.T) + _EPS
    return W * (numerator / denominator)


def update_C(
    D: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """
    Multiplicative update for activation coefficient matrix C.

    C_kj ← C_kj * (W.T @ D)_kj / (W.T @ W @ C + ε)_kj

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Returns
    -------
    C_new : np.ndarray, shape (k, N)

    Notes
    -----
    Paper notation: C (Eq. 3, Soomro et al. 2018).
    """
    numerator = W.T @ D
    denominator = (W.T @ W) @ C + _EPS
    return C * (numerator / denominator)


def reconstruction_error(
    D: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
) -> float:
    """
    Frobenius norm of the reconstruction residual ||D - W @ C||_F.

    Parameters
    ----------
    D : np.ndarray, shape (M, N)
    W : np.ndarray, shape (M, k)
    C : np.ndarray, shape (k, N)

    Returns
    -------
    error : float
        Non-negative reconstruction error.

    Notes
    -----
    Paper notation: FN = ||D - WC||_F (Soomro et al. 2018, Section II-A).
    """
    return float(np.linalg.norm(D - W @ C, "fro"))
