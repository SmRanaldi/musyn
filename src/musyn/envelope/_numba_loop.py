"""
Numba JIT implementation of the adaptive envelope inner loop.

Direct port of loopFunction.c (SmRanaldi/EMG_envelope). Compiled on first
call (cache=True). Falls back to the NumPy implementation when Numba is
unavailable.

Constants match loopFunction.c:
    P_NORM   = sqrt(2/pi)    = 0.797884560802866
    F_FACTOR = (pi-1)/(alpha*nu)^2  (from hardcoded F_FACTOR in the C code)

References
----------
Ranaldi et al. (2018), Equations (3)–(6). MATLAB repo SmRanaldi/EMG_envelope.
"""
from __future__ import annotations

import math

import numpy as np

from musyn.utils.numba_support import NUMBA_AVAILABLE, njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_P_NORM_DEFAULT = math.sqrt(2.0 / math.pi)   # for alpha=1


# ---------------------------------------------------------------------------
# JIT helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _p_norm_nb(alpha: float) -> float:
    """p = 2^(1/(2α)) · Γ((α+1)/(2α)) / √π  (P_NORM in loopFunction.c)."""
    return (2.0 ** (1.0 / (2.0 * alpha))) * math.gamma((alpha + 1.0) / (2.0 * alpha)) / math.sqrt(math.pi)


@njit(cache=True)
def _f_alpha_nu_nb(alpha: float, nu: float) -> float:
    """F_FACTOR = (pi - 1) / (alpha * nu)^2  (hardcoded in loopFunction.c)."""
    return (math.pi - 1.0) / ((alpha * nu) ** 2)


@njit(cache=True)
def _envelope_k(
    signal: np.ndarray,
    k: int,
    m_k: float,
    n: int,
    nu: float,
    p: float,
    exp: float,
) -> float:
    """
    Envelope at sample k: (mean_{window} |s|^ν / p)^(1/(α·ν)).

    semiLen = ceil(m_k / 2); window = [k-semiLen, k+semiLen] ∩ [0, n-1].
    """
    semi = int(math.ceil(m_k * 0.5))
    lo = max(0, k - semi)
    hi = min(n - 1, k + semi)
    L = hi - lo + 1
    acc = 0.0
    for j in range(lo, hi + 1):
        acc += abs(signal[j]) ** nu
    return (acc / L / p) ** exp


@njit(cache=True)
def _derivatives_k(
    signal: np.ndarray,
    k: int,
    m_k: float,
    n: int,
    inv_alpha: float,
    p: float,
) -> tuple[float, float]:
    """
    First (d1) and second (d2) derivatives at sample k.

    Port of derivativesEstimation() in loopFunction.c.

    Centering: a[j] = j - ceil(L/2)  for j = 0 … L-1.

    d1 = sum(a·|s|^(1/α)) / (sum(a²)·p)
    d2 = 2·(t1 − t2)
        t1  = sum(a²·|s|^(1/α)) / (sum(a⁴)·p)
        c   = sum(a²)/sum(a⁴)
        t2  = (c/p)·sum((1−a²c)·|s|^(1/α)) / (L + c·sum(a²))
    """
    semi = int(math.ceil(m_k * 0.5))
    lo = max(0, k - semi)
    hi = min(n - 1, k + semi)
    L = hi - lo + 1
    center = int(math.ceil(0.5 * L))

    r = 0.0   # sum(a^2)
    r2 = 0.0  # sum(a^4)
    for j in range(L):
        a = float(j) - float(center)
        a2 = a * a
        r += a2
        r2 += a2 * a2

    if r == 0.0:
        return 0.0, 0.0

    est1 = 0.0
    est2_1 = 0.0
    est2_2 = 0.0
    c = r / r2 if r2 != 0.0 else 0.0
    for j in range(L):
        a = float(j) - float(center)
        s_pow = abs(signal[lo + j]) ** inv_alpha
        a2 = a * a
        est1 += a * s_pow
        est2_1 += a2 * s_pow
        est2_2 += (1.0 - a2 * c) * s_pow

    d1_val = est1 / (r * p)

    if r2 == 0.0:
        return d1_val, 0.0

    t1 = est2_1 / (r2 * p)
    denom2 = float(L) + c * r
    t2 = (r / (r2 * p)) * est2_2 / denom2 if denom2 != 0.0 else 0.0
    d2_val = 2.0 * (t1 - t2)

    return d1_val, d2_val


@njit(cache=True)
def _filter_length_k(
    env_k: float,
    d1_k: float,
    d2_k: float,
    alpha_nu_m1: float,   # = alpha*nu - 1
    f_val: float,
    w_min: float,
    w_max: float,
) -> float:
    """
    Optimal window length at sample k (filterLength in loopFunction.c).

    M_k = clip(round(|4f·W_k⁴ / den²|^(1/5)), w_min, w_max)
    where W_k = env_k (envelope amplitude, NOT window length).
    """
    aa = -0.5 * d1_k
    denom_bb = 4.0 * env_k if env_k > 1e-12 else 1e-12
    bb = (1.0 / 6.0) * (d2_k + alpha_nu_m1 * d1_k * d1_k / denom_bb)
    num = 4.0 * f_val * env_k ** 4
    den_raw = (bb * env_k + alpha_nu_m1 * aa * aa) / 2.0
    if abs(den_raw) < 1e-12:
        den_raw = 1e-12
    den = den_raw * den_raw
    m_k = round(abs(num / den) ** 0.2)
    if m_k < w_min:
        m_k = w_min
    if m_k > w_max:
        m_k = w_max
    return m_k


@njit(cache=True)
def _adaptive_loop_core(
    signal: np.ndarray,
    m_init: np.ndarray,
    alpha: float,
    nu: float,
    max_iter: int,
    convergence_threshold: float,
    w_min: float,
    w_max: float,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Core Numba JIT adaptive loop.

    Maintains window lengths m and envelope w_env as separate state variables,
    iterating the fixed-point map until convergence.

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    m_arr : np.ndarray, shape (N,)   — converged window lengths
    n_iter : int
    converged : bool
    """
    n = len(signal)
    p = _p_norm_nb(alpha)
    f_val = _f_alpha_nu_nb(alpha, nu)
    inv_alpha = 1.0 / alpha
    exp_env = 1.0 / (alpha * nu)
    alpha_nu_m1 = alpha * nu - 1.0

    # Initialise state
    m = m_init.copy()
    w_env = np.empty(n, dtype=np.float64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    for k in range(n):
        w_env[k] = _envelope_k(signal, k, m[k], n, nu, p, exp_env)
        d1[k], d2[k] = _derivatives_k(signal, k, m[k], n, inv_alpha, p)

    m_new = np.empty(n, dtype=np.float64)
    w_new = np.empty(n, dtype=np.float64)
    d1_new = np.empty(n, dtype=np.float64)
    d2_new = np.empty(n, dtype=np.float64)

    converged = False
    n_iter = 0

    # Entropy-based convergence (log(m) proxy): track last 3 iterations
    # conv when rate of log(M) decrease slows, i.e. second difference < 0
    log_m_t2 = np.zeros(n, dtype=np.float64)   # log M two iterations ago
    log_m_t1 = np.zeros(n, dtype=np.float64)   # log M one iteration ago
    frozen = np.zeros(n, dtype=np.uint8)        # per-sample freeze flag

    for iteration in range(max_iter):
        n_conv = int(np.sum(frozen))

        for k in range(n):
            if frozen[k]:
                continue

            m_new[k] = _filter_length_k(
                w_env[k], d1[k], d2[k], alpha_nu_m1, f_val, w_min, w_max
            )
            w_new[k] = _envelope_k(signal, k, m_new[k], n, nu, p, exp_env)
            d1_new[k], d2_new[k] = _derivatives_k(
                signal, k, m_new[k], n, inv_alpha, p
            )

            # Entropy proxy: log(M) tracks chi2 entropy monotonically
            log_m_now = math.log(max(m_new[k], 1.0))
            if iteration >= 2:
                e2 = log_m_t1[k] - log_m_t2[k]
                e1 = log_m_now - log_m_t1[k]
                if e2 - e1 < 0.0:   # rate of change slowing → converged
                    frozen[k] = 1
                    n_conv += 1

            log_m_t2[k] = log_m_t1[k]
            log_m_t1[k] = log_m_now

        for k in range(n):
            if not frozen[k]:
                m[k] = m_new[k]
                w_env[k] = w_new[k]
                d1[k] = d1_new[k]
                d2[k] = d2_new[k]

        n_iter = iteration + 1
        if float(n_conv) / float(n) >= convergence_threshold:
            converged = True
            break

    return w_env, m, n_iter, converged


def adaptive_loop_nb(
    signal: np.ndarray,
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
    signal : np.ndarray, shape (N,)
        Pre-whitened sEMG signal.
    w_init : np.ndarray, shape (N,)
        Initial window lengths M_0.
    alpha, nu : float
        Signal model parameters.
    max_iter : int
    convergence_threshold : float
    chi2_alpha : float
        Not used in Numba path. Kept for API consistency.
    w_min, w_max : int

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    info : dict
    """
    if not NUMBA_AVAILABLE:
        from musyn.envelope._numpy_loop import adaptive_loop_np
        return adaptive_loop_np(
            signal, w_init, alpha, nu, max_iter,
            convergence_threshold, chi2_alpha, w_min, w_max,
        )

    envelope, m_arr, n_iter, converged = _adaptive_loop_core(
        signal.astype(np.float64),
        w_init.copy().astype(np.float64),
        float(alpha), float(nu),
        int(max_iter), float(convergence_threshold),
        float(w_min), float(w_max),
    )
    return envelope, {
        "iterations": n_iter,
        "converged": converged,
        "backend": "numba",
        "window_lengths": m_arr,
    }
