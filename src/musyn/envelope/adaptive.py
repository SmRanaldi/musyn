"""
Backend dispatcher and top-level adaptive loop for envelope extraction.

Selects the fastest available backend:
  1. Cython C extension (``_loop.so``) — compiled at install
  2. Numba JIT (``_numba_loop``) — compiled on first call
  3. NumPy vectorized (``_numpy_loop``) — always available

References
----------
Ranaldi et al. (2018), Section II-C, Equations (3)–(7).
"""
from __future__ import annotations

import numpy as np

# ── Backend selection ────────────────────────────────────────────────────────
from musyn.utils.numba_support import get_adaptive_loop as _get_loop

_fallback_loop, _FALLBACK_BACKEND = _get_loop()

try:
    from musyn.envelope._loop import adaptive_loop_c as _cython_loop
    _BACKEND = "cython"
except ImportError:
    _cython_loop = None
    _BACKEND = _FALLBACK_BACKEND


def backend() -> str:
    """Return the name of the active adaptive-loop backend."""
    return _BACKEND


def adaptive_envelope(
    whitened: np.ndarray,
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
    Run the adaptive envelope algorithm on a pre-processed signal.

    Dispatches to the fastest compiled backend available. The result is
    the point-by-point envelope estimate w_k.

    Parameters
    ----------
    whitened : np.ndarray, shape (N,)
        Pre-whitened signal (output of ``prewhiten``). The nu-order
        detection (|·|^ν) is applied internally by each backend.
    w_init : np.ndarray, shape (N,)
        Initial window lengths (output of ``initialize_window_lengths``).
    alpha : float
        Shape parameter. Paper fixes alpha=1. Default 1.0.
    nu : float
        Detection order. Paper fixes nu=2. Default 2.0.
    max_iter : int
        Maximum number of iterations. Default 100.
    convergence_threshold : float
        Stop when this fraction of samples has converged. Default 0.95.
    chi2_alpha : float
        Significance level for chi-squared convergence test (NumPy backend).
    w_min : int
        Minimum window length in samples. Default 1.
    w_max : int
        Maximum window length in samples. Default 10000.

    Returns
    -------
    envelope : np.ndarray, shape (N,)
        Estimated amplitude envelope.
    info : dict
        ``iterations`` (int), ``converged`` (bool), ``backend`` (str).

    Notes
    -----
    Paper notation: w_k = (1/M_k) [Σ |s_{k+i}|^ν]^{1/ν} (Eq. 6).
    """
    whitened = np.asarray(whitened, dtype=np.float64)
    w_init_arr = np.asarray(w_init, dtype=np.float64)

    if _cython_loop is not None:
        try:
            w = w_init_arr.copy()
            envelope_arr, n_iter, converged, window_lengths = _cython_loop(
                whitened, w, alpha, nu, max_iter, convergence_threshold, w_min, w_max
            )
            return envelope_arr, {
                "iterations": n_iter, "converged": converged,
                "backend": "cython", "window_lengths": window_lengths,
            }
        except (ValueError, TypeError):
            pass  # old compiled extension — fall through to Python backend

    return _fallback_loop(
        whitened, w_init_arr.copy(), alpha, nu, max_iter,
        convergence_threshold, chi2_alpha, w_min, w_max,
    )
