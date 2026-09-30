"""
NumPy vectorized fallback for the adaptive envelope inner loop.

Direct port of the MATLAB adaptiveEnvelope algorithm (SmRanaldi/EMG_envelope).
The C MEX implementation (loopFunction.c) is the authoritative reference.

Algorithm state per iteration: window lengths m AND envelope w_env are
maintained separately. filterLength uses the envelope w_env (not m) in
its formula. m is the smoothing window; w_env is the amplitude estimate.

Constants from loopFunction.c:
    P_NORM   = sqrt(2/pi)   = 0.797884560802866
    F_FACTOR = (pi-1)/(alpha*nu)^2  (derived from the buggy f.m which
               cancels gamma(nu+0.5)/gamma(nu+0.5) = 1, giving sqrt(pi)^2-1 = pi-1)

References
----------
Ranaldi et al. (2018), Equations (3)–(6). MATLAB repo SmRanaldi/EMG_envelope.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

def _p_norm(alpha: float) -> float:
    """Normalization constant p = 2^(1/(2α)) · Γ((α+1)/(2α)) / √π.

    For alpha=1: p = sqrt(2/pi) ≈ 0.7979  (P_NORM in loopFunction.c).
    """
    return (2.0 ** (1.0 / (2.0 * alpha))) * math.gamma((alpha + 1.0) / (2.0 * alpha)) / math.sqrt(math.pi)


def _f_alpha_nu(alpha: float, nu: float) -> float:
    """F_FACTOR = (pi - 1) / (alpha * nu)^2.

    Matches the hardcoded F_FACTOR in loopFunction.c (= 0.5354 for alpha=1,nu=2),
    which comes from the MATLAB f.m where gamma(nu+0.5) cancels itself,
    reducing to (sqrt(pi)^2 - 1) = (pi - 1).
    """
    return (math.pi - 1.0) / ((alpha * nu) ** 2)


# ---------------------------------------------------------------------------
# Core estimation functions (port of MATLAB/C loopFunction.c)
# ---------------------------------------------------------------------------

def _envelope_estimation(
    signal: np.ndarray,
    m: np.ndarray,
    alpha: float,
    nu: float,
    p: float,
) -> np.ndarray:
    """
    Envelope W_k = (mean_{window} |s|^ν / p)^(1/(α·ν)).

    Port of envelopeEstimation() in loopFunction.c / envelopeEstimationMat.m.
    Window half-length: semiLen = ceil(m[k] / 2).
    Divides by actual window length (not m[k]).
    """
    n = len(signal)
    w = np.empty(n, dtype=np.float64)
    exp = 1.0 / (alpha * nu)
    abs_sig = np.abs(signal)

    for k in range(n):
        semi = math.ceil(m[k] * 0.5)
        lo = max(0, k - semi)
        hi = min(n - 1, k + semi)
        s = abs_sig[lo : hi + 1]
        L = hi - lo + 1
        est = float(np.sum(s ** nu)) / L
        w[k] = (est / p) ** exp

    return w


def _derivatives_estimation(
    signal: np.ndarray,
    m: np.ndarray,
    alpha: float,
    nu: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First (d1) and second (d2) derivatives of the log-envelope.

    Port of derivativesEstimation() in loopFunction.c / derivativesEstimationMat.m.

    Centering: a[j] = j - ceil(winLen / 2)  for j = 0 … winLen-1
    (matches C code: a = j - lowerLimit - ceil(0.5*winLen))

    d1[k] = sum(a · |s|^(1/α)) / (sum(a²) · p)
    d2[k] = 2 · (t1 − t2)
        t1  = sum(a² · |s|^(1/α)) / (sum(a⁴) · p)
        c   = sum(a²) / sum(a⁴)
        t2  = (c / p) · sum((1 − a²·c) · |s|^(1/α)) / (winLen + c·sum(a²))
    """
    n = len(signal)
    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)
    inv_alpha = 1.0 / alpha
    abs_sig = np.abs(signal)

    for k in range(n):
        semi = math.ceil(m[k] * 0.5)
        lo = max(0, k - semi)
        hi = min(n - 1, k + semi)
        s = abs_sig[lo : hi + 1]
        L = hi - lo + 1

        # Centered position indices (C convention)
        center = math.ceil(0.5 * L)
        a = np.arange(L, dtype=np.float64) - center   # a[j] = j - ceil(L/2)

        s_pow = s ** inv_alpha                          # |s|^(1/α)

        r = float(np.dot(a, a))         # sum(a²)
        if r == 0.0:
            continue

        r2 = float(np.sum(a ** 4))      # sum(a⁴)

        d1[k] = float(np.dot(a, s_pow)) / (r * p)

        if r2 == 0.0:
            continue

        c = r / r2
        t1 = float(np.dot(a * a, s_pow)) / (r2 * p)
        est2_2 = float(np.dot(1.0 - a * a * c, s_pow))
        denom2 = L + c * r
        t2 = (r / (r2 * p)) * est2_2 / denom2 if denom2 != 0.0 else 0.0
        d2[k] = 2.0 * (t1 - t2)

    return d1, d2


def _filter_length(
    w_env: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    alpha: float,
    nu: float,
    w_min: int,
    w_max: int,
    f_val: float,
) -> np.ndarray:
    """
    Optimal window lengths from the filterLengthMat / filterLength formula.

    M_k = clip(round(|4f·W_k⁴ / den²|^(1/5)), w_min, w_max)
    where W_k  = envelope amplitude (w_env),
          aa   = −d1/2,
          bb   = (1/6)·(d2 + (α·ν−1)·d1²/(4·W_k)),
          den  = (bb·W_k + (α·ν−1)·aa²) / 2.

    NOTE: uses the ENVELOPE w_env (not window length m) throughout.
    """
    aa = -d1 / 2.0
    bb = (1.0 / 6.0) * (
        d2 + (alpha * nu - 1.0) * d1 ** 2 / np.maximum(4.0 * w_env, 1e-12)
    )
    num = 4.0 * f_val * w_env ** 4
    den_raw = (bb * w_env + (alpha * nu - 1.0) * aa ** 2) / 2.0
    safe_den = np.where(np.abs(den_raw) < 1e-12, 1e-12, den_raw)
    den = safe_den ** 2
    M_new = np.round(np.abs(num / den) ** 0.2)
    return np.clip(M_new, w_min, w_max)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def adaptive_loop_np(
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
    Adaptive envelope loop — NumPy implementation.

    Maintains window lengths m and envelope w_env as separate state variables
    and iterates the fixed-point map until convergence, matching the MATLAB
    adaptiveEnvelope algorithm (loopFunction.c).

    Convergence follows the MATLAB C code: per-sample entropy of chi2(df=m[k])
    is tracked over 3 consecutive iterations; a sample is marked converged when
    the rate of entropy change slows (second difference of entropy < 0). Once
    converged, a sample's state is frozen. The loop stops when
    ``convergence_threshold`` fraction of samples have converged.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Pre-whitened sEMG signal (NOT the nu-order detected signal).
    w_init : np.ndarray, shape (N,)
        Initial window lengths M_0 (samples).
    alpha, nu : float
        Signal model parameters (paper defaults: 1.0 and 2.0).
    max_iter : int
        Maximum iterations.
    convergence_threshold : float
        Stop when this fraction of samples has converged.
    chi2_alpha : float
        Not used directly; kept for API consistency.
    w_min, w_max : int
        Window length bounds.

    Returns
    -------
    envelope : np.ndarray, shape (N,)
    info : dict
        Keys: ``iterations``, ``converged``, ``backend``, ``window_lengths``.
    """
    from scipy.stats import chi2 as _chi2

    p = _p_norm(alpha)
    f_val = _f_alpha_nu(alpha, nu)
    n = len(signal)

    # Initialise state: window lengths m, envelope w_env, derivatives d1/d2
    m = w_init.copy()
    w_env = _envelope_estimation(signal, m, alpha, nu, p)
    d1, d2 = _derivatives_estimation(signal, m, alpha, nu, p)

    # Entropy history for convergence (3 rolling values per sample, like C code)
    ent_t2 = np.zeros(n, dtype=np.float64)   # two iterations ago
    ent_t1 = np.zeros(n, dtype=np.float64)   # one iteration ago
    frozen = np.zeros(n, dtype=bool)          # per-sample freeze flag

    # Precompute chi2 entropy for all integer window lengths
    _ent_cache: dict[int, float] = {}

    def _chi2_entropy(mv: int) -> float:
        if mv not in _ent_cache:
            _ent_cache[mv] = float(_chi2.entropy(df=max(mv, 1)))
        return _ent_cache[mv]

    converged = False
    iteration = 0
    n_conv = 0

    for iteration in range(max_iter):
        # Compute new state only for non-frozen samples
        m_new = m.copy()
        w_new = w_env.copy()
        d1_new = d1.copy()
        d2_new = d2.copy()

        active = ~frozen
        if not np.any(active):
            converged = True
            break

        m_new[active] = _filter_length(
            w_env[active], d1[active], d2[active],
            alpha, nu, w_min, w_max, f_val,
        )
        w_new[active] = _envelope_estimation(signal, m_new, alpha, nu, p)[active]
        d1_full, d2_full = _derivatives_estimation(signal, m_new, alpha, nu, p)
        d1_new[active] = d1_full[active]
        d2_new[active] = d2_full[active]

        # Entropy-based convergence check (matches MATLAB C loopFunction.c)
        # convCtrl = (ent_t1 - ent_t2) - (ent_now - ent_t1) < 0 → converged
        if iteration >= 2:
            m_int = np.clip(np.round(m_new).astype(np.int64), w_min, w_max)
            ent_now = np.array([_chi2_entropy(int(mv)) for mv in m_int])
            conv_ctrl = (ent_t1 - ent_t2) - (ent_now - ent_t1)
            newly_converged = active & (conv_ctrl < 0.0)
            frozen |= newly_converged
            n_conv = int(np.sum(frozen))
        else:
            m_int = np.clip(np.round(m_new).astype(np.int64), w_min, w_max)
            ent_now = np.array([_chi2_entropy(int(mv)) for mv in m_int])

        # Roll entropy history
        ent_t2 = ent_t1.copy()
        ent_t1 = ent_now.copy()

        # Advance state
        m, w_env, d1, d2 = m_new, w_new, d1_new, d2_new

        if n_conv >= convergence_threshold * n:
            converged = True
            break

    return w_env, {
        "iterations": iteration + 1,
        "converged": converged,
        "backend": "numpy",
        "window_lengths": m,
    }
