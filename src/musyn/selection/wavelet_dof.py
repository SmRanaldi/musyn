"""
Daubechies-5 wavelet-based degrees-of-freedom (DoF) estimation.

Replaces the MATLAB DoFWaveletEvents.m from SmRanaldi/NSyn_Criteria.

References
----------
Ranaldi et al. (2021), Section II-C:
  N_DoF = L_{h+1} * (1 + P_D,h+1 / P_A,h+1)
where h is the level where the detail spectrum decorrelates from the signal.
"""
from __future__ import annotations

import numpy as np

try:
    import pywt
    _PWT_AVAILABLE = True
except ImportError:
    _PWT_AVAILABLE = False
    pywt = None


def _check_pywt() -> None:
    if not _PWT_AVAILABLE:
        raise ImportError(
            "PyWavelets (pywt) is required for AIC-based synergy selection. "
            "Install it with: pip install musyn[wavelet]"
        )


def wavelet_decompose(
    signal: np.ndarray,
    wavelet: str = "db5",
    max_level: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Perform multi-level DWT using PyWavelets.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input time series (one row of C matrix).
    wavelet : str
        Wavelet name. Paper uses 'db5' (Daubechies-5). Default 'db5'.
    max_level : int or None
        Maximum decomposition level. If None, uses
        ``pywt.dwt_max_level(len(signal), wavelet)``.

    Returns
    -------
    approx : list[np.ndarray]
        Approximation coefficients at each level (index 0 = coarsest).
    detail : list[np.ndarray]
        Detail coefficients at each level (index 0 = coarsest).

    Notes
    -----
    Uses ``pywt.wavedec`` with ``mode='periodization'`` so that
    coefficient lengths are exactly ``ceil(N / 2^level)``.
    Paper notation: Daubechies-5 (db5) wavelet, Eq. 7.
    """
    _check_pywt()
    if max_level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet, mode="periodization", level=max_level)
    # pywt returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # Re-organise: approx[0]=coarsest, detail[0]=coarsest
    approx = [coeffs[0]]
    detail = list(reversed(coeffs[1:]))  # detail[0]=coarsest level
    return approx, detail


def _power_spectrum_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between power spectra of two signals.

    Both signals are zero-padded to the same length before FFT.
    """
    n = max(len(a), len(b))
    Pa = np.abs(np.fft.rfft(a, n=n)) ** 2
    Pb = np.abs(np.fft.rfft(b, n=n)) ** 2
    if Pa.std() == 0 or Pb.std() == 0:
        return 0.0
    return float(np.corrcoef(Pa, Pb)[0, 1])


def find_decorrelation_level(
    signal: np.ndarray,
    detail: list[np.ndarray],
    corr_threshold: float = -0.05,
) -> int:
    """
    Find the wavelet decomposition level h at which the detail coefficients
    decorrelate from the original signal.

    Computes correlation between the power spectrum of each level's detail
    coefficients and the original signal power spectrum. Returns the first
    level where the correlation drops below ``corr_threshold``.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    detail : list[np.ndarray]
        Detail coefficients from ``wavelet_decompose``, coarsest first.
    corr_threshold : float
        Decorrelation threshold. Paper uses -0.05. Default -0.05.

    Returns
    -------
    h : int
        Decorrelation level index (0-based, into detail list).
        If no level falls below threshold, returns the last level.

    Notes
    -----
    Paper: "find level h where correlation between detail spectrum
    and signal drops below -0.05" (Ranaldi et al. 2021, Section II-C).
    """
    for h, d in enumerate(detail):
        corr = _power_spectrum_corr(d, signal)
        if corr < corr_threshold:
            return h
    return len(detail) - 1


def compute_wavelet_dof(
    C_row: np.ndarray,
    wavelet: str = "db5",
    corr_threshold: float = -0.05,
) -> float:
    """
    Compute effective degrees of freedom for one synergy activation row.

    N_DoF = L_{h+1} * (1 + P_D,h+1 / P_A,h+1)

    where h is the decorrelation level, L_{h+1} is the length of
    detail coefficients at level h+1, P_D,h+1 and P_A,h+1 are the
    total power of detail and approximation coefficients at h+1.

    Parameters
    ----------
    C_row : np.ndarray, shape (N,)
        One row of the activation matrix C.
    wavelet : str
        Default 'db5'.
    corr_threshold : float
        Default -0.05.

    Returns
    -------
    n_dof : float
        Effective degrees of freedom (≥ 1).

    Notes
    -----
    Paper: N_DoF = L_{h+1} * (1 + P_D,h+1 / P_A,h+1)
    (Ranaldi et al. 2021, Eq. 7).
    """
    _check_pywt()
    approx, detail = wavelet_decompose(C_row, wavelet=wavelet)
    h = find_decorrelation_level(C_row, detail, corr_threshold)

    # Level h+1 (clamped to last available)
    h1 = min(h + 1, len(detail) - 1)
    d_h1 = detail[h1]
    # Approximation at h+1 is reconstructed from coarser levels;
    # use the coarsest available approximation resampled to same length
    a_h1 = approx[0]

    L_h1 = float(len(d_h1))
    P_D = float(np.sum(d_h1 ** 2))
    P_A_arr = np.abs(np.fft.rfft(a_h1, n=len(d_h1))) ** 2
    P_A = float(np.sum(P_A_arr))

    if P_A == 0:
        ratio = 0.0
    else:
        ratio = P_D / P_A

    return max(1.0, L_h1 * (1.0 + ratio))


def compute_total_dof(
    C: np.ndarray,
    wavelet: str = "db5",
    corr_threshold: float = -0.05,
) -> float:
    """
    Sum effective DoF over all rows of the activation matrix C.

    Parameters
    ----------
    C : np.ndarray, shape (k, N)
    wavelet : str
    corr_threshold : float

    Returns
    -------
    total_dof : float
    """
    return sum(
        compute_wavelet_dof(C[i], wavelet=wavelet, corr_threshold=corr_threshold)
        for i in range(C.shape[0])
    )
