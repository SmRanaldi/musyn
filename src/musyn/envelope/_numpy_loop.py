"""
NumPy vectorized fallback for the adaptive envelope inner loop.

Always available (no compilation needed). Uses a prefix-sum (cumsum)
groupby strategy: samples are grouped by their integer-rounded window
length, and a sliding sum is computed for each unique width.

This is ~5-20x faster than pure Python but slower than the Cython/Numba
backends. It is used automatically when neither Cython nor Numba is available.

References
----------
Ranaldi et al. (2018), Equations (3)–(6).
"""
from __future__ import annotations

import numpy as np


def _variable_window_sum(
    signal: np.ndarray,
    w_int: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Compute sum_{i=-m//2}^{m//2} |s_{k+i}|^nu for each sample k,
    where m = w_int[k] (half-window radius = m//2).

    Uses the prefix-sum groupby trick: for each unique half-window size,
    compute the prefix sum of |signal|^nu, then look up the window sum
    for all samples that share that window size.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Signal on which to compute local power.
    w_int : np.ndarray, shape (N,), dtype int
        Integer window lengths (full window, symmetric).
    nu : float
        Exponent.

    Returns
    -------
    local_sum : np.ndarray, shape (N,)
    """
    n = len(signal)
    abs_nu = np.abs(signal) ** nu
    # Pad signal for boundary handling
    pad_max = int(w_int.max()) // 2 + 1
    padded = np.pad(abs_nu, pad_max, mode="edge")
    # prefix sum on padded array
    prefix = np.zeros(len(padded) + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(padded)

    local_sum = np.empty(n, dtype=np.float64)
    unique_ws = np.unique(w_int)

    for w in unique_ws:
        half = w // 2
        mask = w_int == w
        indices = np.where(mask)[0]
        # in padded coordinates: sample k → padded index k + pad_max
        left = indices + pad_max - half
        right = indices + pad_max + half + 1  # exclusive
        local_sum[indices] = prefix[right] - prefix[left]

    return local_sum


def _compute_first_derivative(
    signal: np.ndarray,
    w_int: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Estimate first derivative of the modulating waveform a_k.

    a_k = 2 * sum_{j=-L}^{L} j * |s_{k+j}|^alpha  / sum_{j=-L}^{L} j^2

    where L = w_int[k] // 2.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    w_int : np.ndarray, shape (N,), dtype int
    alpha : float

    Returns
    -------
    a : np.ndarray, shape (N,)

    Notes
    -----
    Paper: Eq. (4), Ranaldi et al. (2018).
    """
    n = len(signal)
    abs_alpha = np.abs(signal) ** alpha
    pad_max = int(w_int.max()) // 2 + 1
    padded = np.pad(abs_alpha, pad_max, mode="edge")

    a = np.zeros(n, dtype=np.float64)
    unique_ws = np.unique(w_int)

    for w in unique_ws:
        half = int(w // 2)
        if half == 0:
            continue
        js = np.arange(-half, half + 1, dtype=np.float64)
        denom = float(np.sum(js ** 2))
        if denom == 0:
            continue
        mask = w_int == w
        for k in np.where(mask)[0]:
            pk = k + pad_max
            window = padded[pk - half : pk + half + 1]
            a[k] = 2.0 * np.dot(js, window) / denom

    return a


def _compute_second_derivative(
    signal: np.ndarray,
    w_int: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
    alpha: float,
    nu: float,
) -> np.ndarray:
    """
    Estimate second derivative of the modulating waveform b_k.

    Based on Eq. (5) of Ranaldi et al. (2018).

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    w_int : np.ndarray, shape (N,), dtype int
    w : np.ndarray, shape (N,)
        Current window lengths (float).
    a : np.ndarray, shape (N,)
        First derivative estimates.
    alpha, nu : float

    Returns
    -------
    b : np.ndarray, shape (N,)
    """
    n = len(signal)
    abs_alpha = np.abs(signal) ** alpha
    pad_max = int(w_int.max()) // 2 + 1
    padded = np.pad(abs_alpha, pad_max, mode="edge")

    b = np.zeros(n, dtype=np.float64)
    # normalisation factor p = (M^(1/alpha)) as in paper
    p = np.maximum(w, 1.0) ** (1.0 / alpha)
    unique_ws = np.unique(w_int)

    for wv in unique_ws:
        half = int(wv // 2)
        if half == 0:
            continue
        js = np.arange(-half, half + 1, dtype=np.float64)
        j2 = js ** 2
        sum_j2 = float(np.sum(j2))
        if sum_j2 == 0:
            continue
        mask = w_int == wv
        for k in np.where(mask)[0]:
            pk = k + pad_max
            window = padded[pk - half : pk + half + 1]
            # First term of Eq. (5)
            t1 = 2.0 * np.dot(j2, window) / (p[k] * sum_j2)
            # Second term — uses Lc = w[k]/2
            Lc = w[k] / 2.0
            c_k = a[k] / (2.0 * p[k]) if p[k] != 0 else 0.0
            weights = 1.0 - j2 * c_k
            denom2 = 1.0 + 2.0 * Lc + c_k * sum_j2
            if denom2 != 0:
                t2 = c_k * np.dot(weights, window) / (p[k] * denom2)
            else:
                t2 = 0.0
            b[k] = 2.0 * (t1 + t2)

    return b


def _f_alpha_nu(alpha: float, nu: float) -> float:
    """
    Compute f(alpha, nu) from Eq. (2) of Ranaldi et al. (2018).

    f(alpha, nu) = [ (sqrt(pi) * Gamma(nu+0.5) / Gamma(nu+1))^2 - 1 ]
                  / (alpha * nu)^2

    For nu=2, alpha=1: evaluates to a specific constant.
    """
    from math import gamma, sqrt, pi
    ratio = sqrt(pi) * gamma(nu + 0.5) / gamma(nu + 1.0)
    return (ratio ** 2 - 1.0) / (alpha * nu) ** 2


def _compute_optimal_smoothing(
    w: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    nu: float,
    w_min: int,
    w_max: int,
) -> np.ndarray:
    """
    Compute optimal point-by-point smoothing constant M_k.

    M_k = |4 w_k^2 f(alpha,nu) / (b_k + 0.5*(alpha*nu-1)*a_k^2/w_k^2)|^(1/5)

    Parameters
    ----------
    w, a, b : np.ndarray, shape (N,)
    alpha, nu : float
    w_min, w_max : int

    Returns
    -------
    M : np.ndarray, shape (N,)

    Notes
    -----
    Paper: Eq. (3), Ranaldi et al. (2018).
    """
    f = _f_alpha_nu(alpha, nu)
    w2 = w * w
    safe_w2 = np.where(w2 == 0, 1.0, w2)
    denom = b + 0.5 * (alpha * nu - 1.0) * a * a / safe_w2
    # Prevent division by zero and negative denominator issues
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    M = np.abs(4.0 * w2 * f / safe_denom) ** 0.2  # ^(1/5)
    return np.clip(M, w_min, w_max)


def _relinearize(
    signal: np.ndarray,
    M: np.ndarray,
    nu: float,
    w_min: int,
    w_max: int,
) -> np.ndarray:
    """
    Re-linearize to obtain envelope estimate w_k.

    w_k = (1/M_k) * [sum_{i=-M_k/2}^{M_k/2} |s_{k+i}|^nu]^(1/nu)

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    M : np.ndarray, shape (N,)
        Optimal smoothing constants (float).
    nu : float
    w_min, w_max : int

    Returns
    -------
    w_new : np.ndarray, shape (N,)

    Notes
    -----
    Paper: Eq. (6), Ranaldi et al. (2018).
    """
    M_int = np.clip(np.round(M).astype(np.int64), w_min, w_max)
    local_sum = _variable_window_sum(signal, M_int, nu)
    w_new = (local_sum / np.maximum(M, 1.0)) ** (1.0 / nu)
    return w_new


def adaptive_loop_np(
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
    Adaptive envelope loop — pure NumPy implementation.

    Iteratively refines window lengths until the entropy criterion
    reaches quasi-stationarity (Ranaldi et al. 2018, Section II-C).

    Parameters
    ----------
    detected : np.ndarray, shape (N,)
        Nu-order detected signal (output of ``nu_order_detection``).
    w_init : np.ndarray, shape (N,)
        Initial window lengths.
    alpha : float
        Shape parameter (fixed at 1.0 in the paper).
    nu : float
        Detection order (fixed at 2.0 in the paper).
    max_iter : int
        Maximum number of iterations.
    convergence_threshold : float
        Stop early when this fraction of samples have converged.
    chi2_alpha : float
        Significance level for the chi-squared convergence test.
    w_min, w_max : int
        Window length bounds.

    Returns
    -------
    envelope : np.ndarray, shape (N,)
        Final envelope estimate.
    info : dict
        Keys: ``iterations``, ``converged``, ``backend``.
    """
    from scipy.stats import chi2

    w = w_init.copy()
    n = len(detected)
    converged = False

    for iteration in range(max_iter):
        w_int = np.clip(np.round(w).astype(np.int64), w_min, w_max)

        # Compute derivatives
        a = _compute_first_derivative(detected, w_int, alpha)
        b = _compute_second_derivative(detected, w_int, w, a, alpha, nu)

        # Optimal smoothing constants
        M = _compute_optimal_smoothing(w, a, b, alpha, nu, w_min, w_max)

        # Re-linearize
        w_new = _relinearize(detected, M, nu, w_min, w_max)

        # Convergence: fraction of samples with relative change < chi2 threshold
        w_safe = np.where(w > 0, w, 1.0)
        rel_change = np.abs(w_new - w) / w_safe
        # Use chi-squared critical value as threshold for each sample
        M_int = np.clip(np.round(M).astype(np.int64), w_min, w_max)
        chi2_thresh = np.vectorize(lambda m: chi2.ppf(1.0 - chi2_alpha, df=max(m, 1)))(M_int)
        conv_mask = rel_change < (chi2_thresh / np.maximum(M, 1.0))
        conv_frac = float(conv_mask.mean())

        w = w_new

        if conv_frac >= convergence_threshold:
            converged = True
            break

    return w, {"iterations": iteration + 1, "converged": converged, "backend": "numpy"}
