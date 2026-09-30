"""
Quality metrics for muscle synergy analysis.

References
----------
Soomro et al. (2018), Section II-C: QR metric.
Ranaldi et al. (2021), Section II-E: VAF, R².
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Parameters
    ----------
    a, b : np.ndarray, shape (M,)

    Returns
    -------
    similarity : float
        In [-1, 1]; 1.0 means identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def quality_ratio(
    W_extracted: np.ndarray,
    W_true: np.ndarray,
) -> float:
    """
    Quality Ratio (QR): mean best-match cosine similarity.

    For each extracted synergy, finds the best-matching true synergy
    (maximum cosine similarity), then averages over all extracted synergies.

    Parameters
    ----------
    W_extracted : np.ndarray, shape (M x k)
        Extracted synergy weight matrix.
    W_true : np.ndarray, shape (M x k_true)
        Ground-truth synergy weight matrix.

    Returns
    -------
    qr : float
        QR ∈ [0, 1]; 1.0 = perfect recovery.

    Notes
    -----
    Paper notation: QR = (1/k) Σ_n cos_sim(W_ext_n, W_true_n)
    (Soomro et al. 2018, Eq. 4). Uses ``scipy.spatial.distance.cdist``
    with ``metric='cosine'`` for the full pairwise similarity matrix.
    """
    # cdist with 'cosine' returns 1 - cosine_similarity
    dist = cdist(W_extracted.T, W_true.T, metric="cosine")
    sim = 1.0 - dist  # shape (k, k_true)
    best_match = np.max(sim, axis=1)  # best true synergy for each extracted
    return float(np.mean(best_match))


def vaf(
    D: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
    per_channel: bool = False,
) -> float | np.ndarray:
    """
    Variance Accounted For (VAF).

    Global VAF = 1 - ||D - WC||²_F / ||D||²_F

    Parameters
    ----------
    D : np.ndarray, shape (M x N)
        Original data matrix.
    W : np.ndarray, shape (M x k)
    C : np.ndarray, shape (k x N)
    per_channel : bool
        If True, compute per-channel VAF (one value per muscle row).

    Returns
    -------
    vaf_value : float or np.ndarray
        Scalar global VAF, or array of shape (M,) per channel.

    Notes
    -----
    Paper notation: VAF = 1 - Σ(M - M̂)² / ΣM² (Ranaldi et al. 2021, Eq. 8).
    """
    residual = D - W @ C
    if per_channel:
        ss_res = np.sum(residual ** 2, axis=1)
        ss_tot = np.sum(D ** 2, axis=1)
        return np.where(ss_tot == 0, 1.0, 1.0 - ss_res / ss_tot)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(D ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def r_squared(
    D: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
    per_channel: bool = False,
) -> float | np.ndarray:
    """
    Coefficient of determination R².

    R² = 1 - SS_res / SS_tot  where SS_tot uses the mean of D.

    Parameters
    ----------
    D : np.ndarray, shape (M x N)
    W : np.ndarray, shape (M x k)
    C : np.ndarray, shape (k x N)
    per_channel : bool
        If True, compute per-channel R².

    Returns
    -------
    r2 : float or np.ndarray
        Scalar (mean over channels) or per-channel array.

    Notes
    -----
    Paper notation: R² (Ranaldi et al. 2021, Section II-E).
    """
    D_hat = W @ C
    residual = D - D_hat
    if per_channel:
        ss_res = np.sum(residual ** 2, axis=1)
        D_mean = D.mean(axis=1, keepdims=True)
        ss_tot = np.sum((D - D_mean) ** 2, axis=1)
        return np.where(ss_tot == 0, 1.0, 1.0 - ss_res / ss_tot)
    ss_res = float(np.sum(residual ** 2))
    D_mean = D.mean()
    ss_tot = float(np.sum((D - D_mean) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
