"""
Signal-dependent noise estimation for AIC computation.

References
----------
Ranaldi et al. (2021), Section II-B:
  σ²_SDN,i,j = (c · S̄_i,j)²  where c ∈ [0.1, 0.25].
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def estimate_measurement_noise(M_matrix: np.ndarray) -> float:
    """
    Estimate additive measurement noise variance σ²_M.

    Uses the Median Absolute Deviation (MAD) of the residuals after
    a rank-1 SVD approximation — robust to outliers and signal content.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (n_muscles, n_samples)
        sEMG envelope matrix.

    Returns
    -------
    sigma2_M : float
        Estimated measurement noise variance.

    Notes
    -----
    Paper notation: σ²_M (Ranaldi et al. 2021, Section II-B).
    """
    # Rank-1 approximation
    U, s, Vt = np.linalg.svd(M_matrix, full_matrices=False)
    rank1 = s[0] * np.outer(U[:, 0], Vt[0, :])
    residuals = M_matrix - rank1
    # MAD estimator: σ ≈ MAD / 0.6745 (consistent with Gaussian noise)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = mad / 0.6745
    return float(sigma ** 2)


def estimate_sdn_noise(
    M_matrix: np.ndarray,
    c: float = 0.175,
    smoothing_samples: int = 50,
) -> np.ndarray:
    """
    Estimate signal-dependent noise (SDN) variance.

    σ²_SDN,i,j = (c · S̄_i,j)²

    where S̄_i,j is the local mean envelope computed with a rectangular
    window, using fast convolution.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
        sEMG envelope matrix.
    c : float
        SDN coefficient. Paper: random ∈ [0.1, 0.25]; default midpoint 0.175.
    smoothing_samples : int
        Window length for local mean computation (samples). Default 50.

    Returns
    -------
    sigma2_sdn : np.ndarray, shape (M, N)
        SDN variance at each (muscle, time) point.

    Notes
    -----
    Paper notation: σ²_SDN,i,j = (c · S̄_i,j)² (Eq. 6, Ranaldi et al. 2021).
    """
    win = np.ones(smoothing_samples) / smoothing_samples
    S_bar = np.apply_along_axis(
        lambda row: fftconvolve(row, win, mode="same"), axis=1, arr=M_matrix
    )
    return (c * S_bar) ** 2


def total_noise_variance(
    M_matrix: np.ndarray,
    sigma2_M: float,
    c: float = 0.175,
    smoothing_samples: int = 50,
) -> np.ndarray:
    """
    Compute total noise variance: σ²_total = σ²_M + σ²_SDN.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
    sigma2_M : float
        Measurement noise variance.
    c : float
        SDN coefficient.
    smoothing_samples : int

    Returns
    -------
    sigma2 : np.ndarray, shape (M, N)
        Total noise variance.

    Notes
    -----
    Paper: σ²_i,j = σ²_M + σ²_SDN,i,j (Ranaldi et al. 2021, Section II-B).
    """
    sdn = estimate_sdn_noise(M_matrix, c=c, smoothing_samples=smoothing_samples)
    return sigma2_M + sdn
