# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython C extension for the adaptive envelope hot loop.

Direct port of loopFunction.c (SmRanaldi/EMG_envelope mex/ subfolder).
Compiled at install time via setup.py.

Constants match loopFunction.c exactly:
    P_NORM   = 0.797884560802866   (= sqrt(2/pi))
    F_FACTOR = 0.535398163397448   (= (pi-1) / (alpha*nu)^2 for alpha=1,nu=2)

If this extension is not compiled, the package falls back to Numba JIT or
NumPy — see adaptive.py.

References
----------
Ranaldi et al. (2018), Equations (3)–(6). MATLAB repo SmRanaldi/EMG_envelope.
"""

import numpy as np
cimport numpy as np
from libc.math cimport (
    fabs as c_abs, pow as c_pow, sqrt, log, exp, lgamma, ceil as c_ceil,
    round as c_round
)

np.import_array()

# ---------------------------------------------------------------------------
# Constants (from loopFunction.c)
# ---------------------------------------------------------------------------

cdef double P_NORM = 0.797884560802866   # sqrt(2/pi), alpha=1
cdef double F_FACTOR = 0.535398163397448 # (pi-1)/4, alpha=1,nu=2

cdef double PI = 3.141592653589793238


# ---------------------------------------------------------------------------
# Compile-time helpers for generic alpha/nu
# ---------------------------------------------------------------------------

cdef inline double _p_norm_c(double alpha) noexcept nogil:
    """p = 2^(1/(2α)) · exp(lgamma((α+1)/(2α)) − 0.5·log(π))."""
    return c_pow(2.0, 1.0 / (2.0 * alpha)) * exp(lgamma((alpha + 1.0) / (2.0 * alpha)) - 0.5 * log(PI))


cdef inline double _f_factor_c(double alpha, double nu) noexcept nogil:
    """F = (π − 1) / (α·ν)²  (matches loopFunction.c F_FACTOR)."""
    return (PI - 1.0) / ((alpha * nu) * (alpha * nu))


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------

cdef double _envelope_k(
    double[::1] signal, int k, double m_k, int n,
    double nu, double p, double exp_env
) noexcept nogil:
    """
    W_k = (mean_{[k-semi, k+semi]} |s|^ν / p)^(1/(α·ν)).

    semiLen = ceil(m_k / 2); clamped to [0, n-1].
    """
    cdef int semi, lo, hi, L, j
    cdef double acc

    semi = <int>c_ceil(m_k * 0.5)
    lo = k - semi
    if lo < 0:
        lo = 0
    hi = k + semi
    if hi >= n:
        hi = n - 1
    L = hi - lo + 1

    acc = 0.0
    for j in range(lo, hi + 1):
        acc += c_pow(c_abs(signal[j]), nu)

    return c_pow(acc / <double>L / p, exp_env)


cdef void _derivatives_k(
    double[::1] signal, int k, double m_k, int n,
    double inv_alpha, double p,
    double *d1_out, double *d2_out
) noexcept nogil:
    """
    d1, d2 at sample k — port of derivativesEstimation() in loopFunction.c.

    Centering: a[j] = j − ceil(L/2)  for j = 0 … L-1.

    d1 = sum(a·|s|^(1/α)) / (sum(a²)·p)
    d2 = 2·(t1 − t2)
        t1  = sum(a²·|s|^(1/α)) / (sum(a⁴)·p)
        c   = sum(a²)/sum(a⁴)
        t2  = (c/p)·sum((1−a²c)·|s|^(1/α)) / (L + c·sum(a²))
    """
    cdef int semi, lo, hi, L, center, j
    cdef double a, a2, s_pow
    cdef double r, r2, c
    cdef double est1, est2_1, est2_2
    cdef double t1, t2, denom2

    semi = <int>c_ceil(m_k * 0.5)
    lo = k - semi
    if lo < 0:
        lo = 0
    hi = k + semi
    if hi >= n:
        hi = n - 1
    L = hi - lo + 1
    center = <int>c_ceil(0.5 * <double>L)

    # First pass: accumulate r = sum(a^2), r2 = sum(a^4)
    r = 0.0
    r2 = 0.0
    for j in range(L):
        a = <double>j - <double>center
        a2 = a * a
        r += a2
        r2 += a2 * a2

    if r == 0.0:
        d1_out[0] = 0.0
        d2_out[0] = 0.0
        return

    c = r / r2 if r2 != 0.0 else 0.0

    # Second pass: accumulate est1, est2_1, est2_2
    est1 = 0.0
    est2_1 = 0.0
    est2_2 = 0.0
    for j in range(L):
        a = <double>j - <double>center
        s_pow = c_pow(c_abs(signal[lo + j]), inv_alpha)
        a2 = a * a
        est1 += a * s_pow
        est2_1 += a2 * s_pow
        est2_2 += (1.0 - a2 * c) * s_pow

    d1_out[0] = est1 / (r * p)

    if r2 == 0.0:
        d2_out[0] = 0.0
        return

    t1 = est2_1 / (r2 * p)
    denom2 = <double>L + c * r
    t2 = (r / (r2 * p)) * est2_2 / denom2 if denom2 != 0.0 else 0.0
    d2_out[0] = 2.0 * (t1 - t2)


cdef double _filter_length_k(
    double env_k, double d1_k, double d2_k,
    double alpha_nu_m1, double f_val,
    int w_min, int w_max
) noexcept nogil:
    """
    M_k = clip(round(|4f·W_k⁴/den²|^(1/5)), w_min, w_max).

    Uses envelope W_k (env_k), NOT window length m.
    den = (bb·W_k + (α·ν−1)·aa²) / 2.
    """
    cdef double aa, bb, num, den_raw, den, m_k, denom_bb

    aa = -0.5 * d1_k
    denom_bb = 4.0 * env_k if env_k > 1e-12 else 1e-12
    bb = (1.0 / 6.0) * (d2_k + alpha_nu_m1 * d1_k * d1_k / denom_bb)
    num = 4.0 * f_val * c_pow(env_k, 4.0)
    den_raw = (bb * env_k + alpha_nu_m1 * aa * aa) / 2.0
    if c_abs(den_raw) < 1e-12:
        den_raw = 1e-12
    den = den_raw * den_raw
    m_k = c_round(c_pow(c_abs(num / den), 0.2))
    if m_k < <double>w_min:
        m_k = <double>w_min
    if m_k > <double>w_max:
        m_k = <double>w_max
    return m_k


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def adaptive_loop_c(
    double[::1] signal not None,
    double[::1] w_init not None,
    double alpha = 1.0,
    double nu = 2.0,
    int max_iter = 100,
    double convergence_threshold = 0.95,
    int w_min = 1,
    int w_max = 10000,
):
    """
    Adaptive envelope loop — Cython C implementation.

    Direct port of loopFunction.c. Maintains window lengths m and
    envelope w_env as separate state variables.

    Parameters
    ----------
    signal : double[::1]
        Pre-whitened signal, contiguous float64 memoryview.
    w_init : double[::1]
        Initial window lengths M_0 (samples).
    alpha, nu : double
    max_iter : int
    convergence_threshold : double
    w_min, w_max : int

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    n_iter : int
    converged : bool
    window_lengths : np.ndarray, shape (N,)
    """
    cdef int n = len(signal)
    cdef double p = _p_norm_c(alpha)
    cdef double f_val = _f_factor_c(alpha, nu)
    cdef double inv_alpha = 1.0 / alpha
    cdef double exp_env = 1.0 / (alpha * nu)
    cdef double alpha_nu_m1 = alpha * nu - 1.0
    cdef double chi_scale = 1.0 + 2.0 * sqrt(2.0)

    cdef double[:] m = np.array(w_init, dtype=np.float64)
    cdef double[:] w_env = np.empty(n, dtype=np.float64)
    cdef double[:] d1 = np.empty(n, dtype=np.float64)
    cdef double[:] d2 = np.empty(n, dtype=np.float64)
    cdef double[:] m_new = np.empty(n, dtype=np.float64)
    cdef double[:] w_new = np.empty(n, dtype=np.float64)
    cdef double[:] d1_new = np.empty(n, dtype=np.float64)
    cdef double[:] d2_new = np.empty(n, dtype=np.float64)

    cdef int k, iteration, n_conv
    cdef double log_m_now, e1, e2
    cdef bint converged = False

    # Entropy proxy arrays (log(M) tracks chi2 entropy monotonically)
    cdef double[:] log_m_t2 = np.zeros(n, dtype=np.float64)
    cdef double[:] log_m_t1 = np.zeros(n, dtype=np.float64)
    cdef unsigned char[:] frozen = np.zeros(n, dtype=np.uint8)

    # Initialise state
    for k in range(n):
        w_env[k] = _envelope_k(signal, k, m[k], n, nu, p, exp_env)
        _derivatives_k(signal, k, m[k], n, inv_alpha, p, &d1[k], &d2[k])

    for iteration in range(max_iter):
        n_conv = 0
        for k in range(n):
            if frozen[k]:
                n_conv += 1
                continue

        for k in range(n):
            if frozen[k]:
                continue

            m_new[k] = _filter_length_k(
                w_env[k], d1[k], d2[k], alpha_nu_m1, f_val, w_min, w_max
            )
            w_new[k] = _envelope_k(signal, k, m_new[k], n, nu, p, exp_env)
            _derivatives_k(signal, k, m_new[k], n, inv_alpha, p, &d1_new[k], &d2_new[k])

            # Entropy proxy: log(M) — converge when rate of change slows
            log_m_now = log(m_new[k]) if m_new[k] > 1.0 else 0.0
            if iteration >= 2:
                e2 = log_m_t1[k] - log_m_t2[k]
                e1 = log_m_now - log_m_t1[k]
                if e2 - e1 < 0.0:
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

        if (<double>n_conv / <double>n) >= convergence_threshold:
            converged = True
            break

    return np.asarray(w_env), iteration + 1, converged, np.asarray(m)
