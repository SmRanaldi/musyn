"""
Nu-order detection and window initialization for the adaptive envelope.

References
----------
Ranaldi et al. (2018), Section II-B: "nu-order detection of the
pre-whitened signal" with nu=2 (i.e., squaring the rectified signal).
"""
from __future__ import annotations

import numpy as np


def nu_order_detection(signal: np.ndarray, nu: int = 2) -> np.ndarray:
    """
    Apply nu-th order detection (rectify then raise to the power nu).

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Pre-whitened sEMG signal.
    nu : int
        Detection order. The paper fixes nu=2, which is equivalent to
        squaring the absolute value.

    Returns
    -------
    detected : np.ndarray, shape (N,)
        Non-negative detected signal ``|signal|^nu``.

    Notes
    -----
    Paper notation: s_k = |α w_k| n_k (Eq. 1, Ranaldi et al. 2018).
    For nu=2: detected[k] = signal[k]^2, implemented as element-wise
    squaring to avoid the overhead of ``np.abs`` + ``np.power``.
    """
    if nu == 2:
        return signal * signal
    return np.abs(signal) ** nu


def initialize_window_lengths(
    signal_length: int,
    fs: float = 1000.0,
    w_min: int = 1,
    w_max: int = 10000,
    init_ms: float = 100.0,
) -> np.ndarray:
    """
    Initialize the adaptive filter window lengths array.

    All windows start at the same value (``init_ms`` milliseconds),
    clipped to [w_min, w_max].

    Parameters
    ----------
    signal_length : int
        Number of samples N.
    fs : float
        Sampling frequency in Hz. Used to convert ``init_ms`` to samples.
    w_min : int
        Minimum allowed window length (samples).
    w_max : int
        Maximum allowed window length (samples).
    init_ms : float
        Initial window length in milliseconds (default 100 ms).

    Returns
    -------
    w : np.ndarray, shape (N,), dtype float64
        Initial window length at each sample, all equal to
        ``clip(init_ms * fs / 1000, w_min, w_max)``.

    Notes
    -----
    Paper: "adaptive window initialized to typical literature values
    (e.g., 40–200 samples at 1 kHz)." (Section II-C, Ranaldi et al. 2018).
    """
    init_samples = float(np.clip(init_ms * fs / 1000.0, w_min, w_max))
    return np.full(signal_length, init_samples, dtype=np.float64)
