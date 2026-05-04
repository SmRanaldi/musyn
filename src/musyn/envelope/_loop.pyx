# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython C extension for the adaptive envelope hot loop.

Direct port of the MATLAB MEX ``loopFunction.c`` from SmRanaldi/EMG_envelope.
Compiled at install time via ``setup.py``. ~100-500x faster than pure Python.

If this extension is not compiled (Cython unavailable or build failed),
the package falls back to Numba JIT or NumPy — see ``adaptive.py``.

References
----------
Ranaldi et al. (2018), Equations (3)–(6).
"""

import numpy as np
cimport numpy as np
from libc.math cimport abs as c_abs, pow as c_pow, sqrt, log, exp, lgamma

np.import_array()

# ─── internal helpers ────────────────────────────────────────────────────────

cdef double _f_alpha_nu(double alpha, double nu) noexcept nogil:
    """f(alpha, nu) from Eq. (2), Ranaldi et al. (2018)."""
    cdef double ratio = sqrt(3.14159265358979323846) * exp(
        lgamma(nu + 0.5) - lgamma(nu + 1.0)
    )
    return (ratio * ratio - 1.0) / ((alpha * nu) * (alpha * nu))


cdef double _local_sum(
    double[::1] signal, int k, int half, int n, double nu
) noexcept nogil:
    """Sum |s_{k+i}|^nu for i in [-half, half], clamped to [0, n-1]."""
    cdef double total = 0.0
    cdef int i, idx
    for i in range(-half, half + 1):
        idx = k + i
        if 0 <= idx < n:
            total += c_pow(c_abs(signal[idx]), nu)
    return total


cdef double _first_deriv(
    double[::1] signal, int k, int half, int n, double alpha
) noexcept nogil:
    """First derivative a_k (Eq. 4)."""
    cdef double num = 0.0
    cdef double denom = 0.0
    cdef int j, idx
    cdef double jf
    for j in range(-half, half + 1):
        jf = <double>j
        idx = k + j
        if 0 <= idx < n:
            num += jf * c_pow(c_abs(signal[idx]), alpha)
        denom += jf * jf
    if denom == 0.0:
        return 0.0
    return 2.0 * num / denom


cdef double _second_deriv(
    double[::1] signal, int k, int half, int n,
    double alpha, double w_k, double a_k, double p_k
) noexcept nogil:
    """Second derivative b_k (Eq. 5)."""
    cdef double sum_j2 = 0.0
    cdef double t1_num = 0.0
    cdef int j, idx
    cdef double jf, j2
    for j in range(-half, half + 1):
        jf = <double>j
        j2 = jf * jf
        sum_j2 += j2
        idx = k + j
        if 0 <= idx < n:
            t1_num += j2 * c_pow(c_abs(signal[idx]), alpha)

    if sum_j2 == 0.0 or p_k == 0.0:
        return 0.0

    cdef double t1 = 2.0 * t1_num / (p_k * sum_j2)
    cdef double Lc = w_k / 2.0
    cdef double c_k = a_k / (2.0 * p_k)
    cdef double denom2 = 1.0 + 2.0 * Lc + c_k * sum_j2
    if denom2 == 0.0:
        return 2.0 * t1

    cdef double t2_num = 0.0
    for j in range(-half, half + 1):
        idx = k + j
        if 0 <= idx < n:
            jf = <double>j
            t2_num += (1.0 - jf * jf * c_k) * c_pow(c_abs(signal[idx]), alpha)

    cdef double t2 = c_k * t2_num / (p_k * denom2)
    return 2.0 * (t1 + t2)


# ─── public function ─────────────────────────────────────────────────────────

def adaptive_loop_c(
    double[::1] signal not None,
    double[::1] w not None,
    double alpha = 1.0,
    double nu = 2.0,
    int max_iter = 100,
    double convergence_threshold = 0.95,
    int w_min = 1,
    int w_max = 10000,
):
    """
    Adaptive envelope loop — Cython C implementation.

    Primary backend for ``extract_envelope``. Compiled at install.

    Parameters
    ----------
    signal : double[::1]
        Nu-order detected signal, contiguous float64 memoryview.
    w : double[::1]
        Initial window lengths (modified in-place).
    alpha, nu : double
        Signal model parameters (fixed at 1.0, 2.0 in the paper).
    max_iter : int
    convergence_threshold : double
    w_min, w_max : int

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    n_iter : int
    converged : bool
    """
    cdef int n = len(signal)
    cdef double f = _f_alpha_nu(alpha, nu)
    cdef double[:] w_new = np.empty(n, dtype=np.float64)

    cdef int k, half, half_M, iteration, n_conv
    cdef double a_k, b_k, p_k, w2, denom, M_k, local_sum_val, w_k_new, rel_change
    cdef bint converged = False

    for iteration in range(max_iter):
        n_conv = 0
        for k in range(n):
            half = max(1, <int>(w[k]) // 2)
            p_k = c_pow(max(w[k], 1.0), 1.0 / alpha)

            a_k = _first_deriv(signal, k, half, n, alpha)
            b_k = _second_deriv(signal, k, half, n, alpha, w[k], a_k, p_k)

            # Optimal smoothing constant M_k (Eq. 3)
            w2 = w[k] * w[k]
            denom = b_k + 0.5 * (alpha * nu - 1.0) * a_k * a_k / max(w2, 1e-12)
            if c_abs(denom) < 1e-12:
                denom = 1e-12
            M_k = c_pow(c_abs(4.0 * w2 * f / denom), 0.2)
            if M_k < w_min:
                M_k = w_min
            if M_k > w_max:
                M_k = w_max

            # Re-linearize (Eq. 6)
            half_M = max(1, <int>M_k // 2)
            local_sum_val = _local_sum(signal, k, half_M, n, nu)
            w_k_new = c_pow(local_sum_val / max(M_k, 1.0), 1.0 / nu)
            if w_k_new < w_min:
                w_k_new = w_min
            if w_k_new > w_max:
                w_k_new = w_max
            w_new[k] = w_k_new

            rel_change = c_abs(w_k_new - w[k]) / max(w[k], 1e-12)
            if rel_change < 0.01:
                n_conv += 1

        for k in range(n):
            w[k] = w_new[k]

        if (<double>n_conv / <double>n) >= convergence_threshold:
            converged = True
            break

    return np.asarray(w), iteration + 1, converged
