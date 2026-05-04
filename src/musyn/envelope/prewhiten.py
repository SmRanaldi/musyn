"""
AR-based prewhitening for sEMG signals.

Replaces the MATLAB MEX functions ``posAutoCorr.c`` and ``whiteningSignal.c``
from the EMG_envelope repository (SmRanaldi/EMG_envelope).

References
----------
Ranaldi et al. (2018), Section II-A: "The signal is pre-whitened by means of
an AR model of order 11 estimated via the Yule-Walker method."
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter


def compute_autocorrelation(signal: np.ndarray, max_lag: int, biased: bool = True) -> np.ndarray:
    """
    Compute the autocorrelation sequence R(0), R(1), ..., R(max_lag).

    Uses FFT-based cross-correlation via ``numpy.correlate`` for efficiency.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Zero-mean input signal.
    max_lag : int
        Maximum lag index (inclusive).
    biased : bool
        If True, divide by N (biased estimator, matches MATLAB ``aryule``).
        If False, divide by (N - lag) (unbiased).

    Returns
    -------
    r : np.ndarray, shape (max_lag + 1,)
        r[k] = R(k), the autocorrelation at lag k.

    Notes
    -----
    Paper notation: R(k) — autocorrelation at lag k (Eq. implicit in Section II-A).
    """
    n = len(signal)
    # Full correlation, then take positive lags [N-1 : N-1+max_lag+1]
    full = np.correlate(signal, signal, mode="full")
    r_full = full[n - 1 : n + max_lag]  # shape (max_lag+1,)
    if biased:
        return r_full / n
    else:
        lags = np.arange(max_lag + 1)
        return r_full / (n - lags)


def estimate_ar_coefficients(signal: np.ndarray, order: int = 12) -> np.ndarray:
    """
    Estimate AR(p) coefficients via the Yule-Walker method.

    Solves the Toeplitz system R * a = -r using
    ``scipy.linalg.solve_toeplitz``, which matches the behaviour of
    MATLAB's ``aryule``.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input signal (need not be zero-mean; mean is removed internally).
    order : int
        AR model order p. The paper uses order 11; default here is 12.

    Returns
    -------
    a : np.ndarray, shape (order,)
        AR coefficients [a1, ..., ap] such that the prediction-error
        filter is ``1 + a1*z^{-1} + ... + ap*z^{-p}``.

    Notes
    -----
    Paper notation: AR order p=11 (Section II-A of Ranaldi et al. 2018).
    """
    x = signal - signal.mean()
    r = compute_autocorrelation(x, order, biased=True)
    # Yule-Walker: Toeplitz(r[0..p-1]) * a = -r[1..p]
    a = solve_toeplitz(r[:order], -r[1 : order + 1])
    return a


def whiten_signal(signal: np.ndarray, ar_coeffs: np.ndarray) -> np.ndarray:
    """
    Filter the signal through the AR prediction-error filter.

    The whitening filter is an all-pole IIR: ``H(z) = 1 / A(z)`` is the
    AR model; the prediction-error filter is ``A(z) = 1 + sum_k a_k z^{-k}``.
    Applying ``A(z)`` to the signal yields approximately white residuals.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input sEMG signal.
    ar_coeffs : np.ndarray, shape (order,)
        AR coefficients as returned by ``estimate_ar_coefficients``.

    Returns
    -------
    residuals : np.ndarray, shape (N,)
        Pre-whitened signal (prediction residuals).

    Notes
    -----
    Replaces MATLAB MEX ``whiteningSignal.c``.
    Uses ``scipy.signal.lfilter`` with ``b=[1]``, ``a=[1, *ar_coeffs]``.
    Initial conditions are set to reduce edge effects.
    """
    # Prediction-error (whitening) filter: A(z) applied as FIR
    # e[t] = x[t] + a1*x[t-1] + ... + ap*x[t-p]
    # scipy convention: lfilter(b, a) → b is the FIR numerator
    b = np.concatenate([[1.0], ar_coeffs])
    a = np.array([1.0])
    residuals = lfilter(b, a, signal)
    return residuals


def prewhiten(signal: np.ndarray, order: int = 12) -> np.ndarray:
    """
    Pre-whiten an sEMG signal using an AR model.

    Convenience wrapper: estimates AR coefficients then applies the
    prediction-error filter.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Raw sEMG signal (single channel).
    order : int
        AR model order. Paper uses 11; default 12.

    Returns
    -------
    whitened : np.ndarray, shape (N,)
        Pre-whitened residual signal.
    """
    coeffs = estimate_ar_coefficients(signal, order)
    return whiten_signal(signal, coeffs)
