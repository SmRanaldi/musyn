"""
Numba JIT implementation of the adaptive envelope inner loop.

Compiled on first call (``cache=True`` saves to ``__pycache__``).
Falls back gracefully to the NumPy implementation if Numba is unavailable —
import this module only after checking ``numba_available()``.

Performance: ~100-200x faster than pure Python, ~5-20x faster than NumPy.

References
----------
Ranaldi et al. (2018), Equations (3)–(6). Replaces MATLAB MEX loopFunction.c.
"""
from __future__ import annotations

import math

import numpy as np

from musyn.utils.numba_support import njit, prange, NUMBA_AVAILABLE


@njit(cache=True)
def _f_alpha_nu_nb(alpha: float, nu: float) -> float:
    """f(alpha, nu) from Eq. (2), Ranaldi et al. (2018)."""
    ratio = math.sqrt(math.pi) * math.exp(
        math.lgamma(nu + 0.5) - math.lgamma(nu + 1.0)
    )
    return (ratio * ratio - 1.0) / ((alpha * nu) ** 2)


@njit(cache=True)
def _local_sum_nb(
    signal: np.ndarray,
    k: int,
    half: int,
    n: int,
    nu: float,
) -> float:
    """Sum |s_{k+i}|^nu for i in [-half, half], clamped to [0, n-1]."""
    total = 0.0
    for i in range(-half, half + 1):
        idx = k + i
        if 0 <= idx < n:
            total += abs(signal[idx]) ** nu
    return total


@njit(cache=True)
def _first_deriv_nb(
    signal: np.ndarray,
    k: int,
    half: int,
    n: int,
    alpha: float,
) -> float:
    """First derivative a_k (Eq. 4, Ranaldi et al. 2018)."""
    num = 0.0
    denom = 0.0
    for j in range(-half, half + 1):
        idx = k + j
        if 0 <= idx < n:
            num += j * (abs(signal[idx]) ** alpha)
        denom += float(j * j)
    if denom == 0.0:
        return 0.0
    return 2.0 * num / denom


@njit(cache=True)
def _second_deriv_nb(
    signal: np.ndarray,
    k: int,
    half: int,
    n: int,
    alpha: float,
    w_k: float,
    a_k: float,
    p_k: float,
) -> float:
    """Second derivative b_k (Eq. 5, Ranaldi et al. 2018)."""
    sum_j2 = 0.0
    t1_num = 0.0
    for j in range(-half, half + 1):
        j2 = float(j * j)
        sum_j2 += j2
        idx = k + j
        if 0 <= idx < n:
            t1_num += j2 * (abs(signal[idx]) ** alpha)

    if sum_j2 == 0.0 or p_k == 0.0:
        return 0.0

    t1 = 2.0 * t1_num / (p_k * sum_j2)

    Lc = w_k / 2.0
    c_k = a_k / (2.0 * p_k)
    denom2 = 1.0 + 2.0 * Lc + c_k * sum_j2
    if denom2 == 0.0:
        return 2.0 * t1

    t2_num = 0.0
    for j in range(-half, half + 1):
        idx = k + j
        if 0 <= idx < n:
            weight = 1.0 - float(j * j) * c_k
            t2_num += weight * (abs(signal[idx]) ** alpha)
    t2 = c_k * t2_num / (p_k * denom2)

    return 2.0 * (t1 + t2)


@njit(cache=True)
def _adaptive_loop_core(
    signal: np.ndarray,
    w: np.ndarray,
    alpha: float,
    nu: float,
    max_iter: int,
    convergence_threshold: float,
    w_min: float,
    w_max: float,
) -> tuple[np.ndarray, int, bool]:
    """
    Core Numba JIT adaptive loop.

    Iterates the Ranaldi et al. (2018) algorithm sample-by-sample.
    Uses explicit loops (no vectorization) to handle variable window sizes
    at each sample — this is the computationally efficient path.

    Returns
    -------
    w : np.ndarray, shape (N,)
        Final window lengths / envelope.
    n_iter : int
    converged : bool
    """
    n = len(signal)
    f = _f_alpha_nu_nb(alpha, nu)
    w_new = np.empty(n, dtype=np.float64)
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_conv = 0

        for k in range(n):
            half = max(1, int(w[k]) // 2)
            p_k = max(w[k], 1.0) ** (1.0 / alpha)

            a_k = _first_deriv_nb(signal, k, half, n, alpha)
            b_k = _second_deriv_nb(signal, k, half, n, alpha, w[k], a_k, p_k)

            # Optimal smoothing constant M_k (Eq. 3)
            w2 = w[k] * w[k]
            denom = b_k + 0.5 * (alpha * nu - 1.0) * a_k * a_k / max(w2, 1e-12)
            if abs(denom) < 1e-12:
                denom = 1e-12
            M_k = abs(4.0 * w2 * f / denom) ** 0.2
            M_k = min(max(M_k, w_min), w_max)

            # Re-linearize (Eq. 6)
            half_M = max(1, int(M_k) // 2)
            local_sum = _local_sum_nb(signal, k, half_M, n, nu)
            w_k_new = (local_sum / max(M_k, 1.0)) ** (1.0 / nu)
            w_k_new = min(max(w_k_new, w_min), w_max)
            w_new[k] = w_k_new

            # Convergence check
            rel_change = abs(w_k_new - w[k]) / max(w[k], 1e-12)
            if rel_change < 0.01:  # simple per-sample convergence criterion
                n_conv += 1

        # Copy w_new into w
        for k in range(n):
            w[k] = w_new[k]

        n_iter = iteration + 1
        if float(n_conv) / float(n) >= convergence_threshold:
            converged = True
            break

    return w, n_iter, converged


def adaptive_loop_nb(
    detected: np.ndarray,
    w_init: np.ndarray,
    alpha: float = 1.0,
    nu: float = 2.0,
    max_iter: int = 100,
    convergence_threshold: float = 0.95,
    chi2_alpha: float = 0.05,
    w_min: int = 1,
    w_max: int = 10000,
) -> tuple[np.ndarray, dict]:
    """
    Adaptive envelope loop — Numba JIT implementation.

    Parameters
    ----------
    detected : np.ndarray, shape (N,)
        Nu-order detected signal.
    w_init : np.ndarray, shape (N,)
        Initial window lengths.
    alpha, nu : float
        Signal model parameters (fixed at 1.0 and 2.0 in the paper).
    max_iter : int
        Maximum iterations.
    convergence_threshold : float
        Fraction of converged samples for early stopping.
    chi2_alpha : float
        Not used in Numba path (convergence uses a simpler 1% threshold
        per sample for JIT compatibility). Kept for API consistency.
    w_min, w_max : int
        Window length bounds.

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    info : dict
    """
    if not NUMBA_AVAILABLE:
        from musyn.envelope._numpy_loop import adaptive_loop_np
        return adaptive_loop_np(
            detected, w_init, alpha, nu, max_iter,
            convergence_threshold, chi2_alpha, w_min, w_max,
        )

    w = w_init.copy()
    envelope, n_iter, converged = _adaptive_loop_core(
        detected, w,
        float(alpha), float(nu),
        int(max_iter), float(convergence_threshold),
        float(w_min), float(w_max),
    )
    return envelope, {"iterations": n_iter, "converged": converged, "backend": "numba"}
