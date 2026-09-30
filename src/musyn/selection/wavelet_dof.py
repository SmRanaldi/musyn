"""
Daubechies-5 wavelet-based degrees-of-freedom (DoF) estimation.

Direct Python port of DoFWaveletEvents from the reference implementation.

The algorithm operates on **event-segmented** signals: the activation row is
split into cycles (gait cycles, movement repetitions) by providing
event-boundary indices, and DoF is computed independently for each cycle then
**summed**. Without segmentation the DoF scale is wrong and the AIC curve
loses its ascending phase.

Key algorithm per segment:
  1. Iterative single-level DWT, n-1 levels total (n = floor(log2(N))).
  2. Resample each level's CA and CD back to segment length; compute periodogram.
  3. Correlate the detail periodogram (first L = round(L_pS/2^i) bins) with
     the signal periodogram at each level → cpCD vector.
  4. Decorrelation level h: first index where diff(cpCD) exceeds
     ``-0.05 * mean(|diff(cpCD)|)``.
  5. DoF_cycle = len(CA[h+1]) * (1 + min(1, P_D[h+1] / P_A[h+1])).
  6. Total DoF for the row = sum over all cycles.

References
----------
Ranaldi et al. (2021) "An Objective, Information-Based Approach for Selecting
the Number of Muscle Synergies via NNMF."
IEEE Trans Neural Syst Rehabil Eng.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram as scipy_periodogram
from scipy.signal import resample as scipy_resample

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


def _compute_dof_segment(segment: np.ndarray, wavelet: str) -> float:
    """
    DoF for a single mean-removed signal segment.

    Matches the inner loop body of DoFWaveletEvents from the reference
    implementation. Returns max(1, DoF) so degenerate segments contribute
    at least 1.
    """
    signal = segment - np.mean(segment)
    N = len(signal)
    n = int(np.floor(np.log2(N)))
    if n < 2:
        return 1.0

    # --- Level 1 DWT (outside the loop, as in the reference)
    ca, cd = pywt.dwt(signal, wavelet)
    lca = [len(ca)]
    rca = scipy_resample(ca, N)
    rca -= np.mean(rca)
    rcd = scipy_resample(cd, N)
    rcd -= np.mean(rcd)
    pca_list = [scipy_periodogram(rca, nfft=N)[1]]
    pcd_list = [scipy_periodogram(rcd, nfft=N)[1]]

    # --- Levels 2..n-1  (n-2 additional iterations, for n-1 total)
    for _ in range(n - 2):
        ca, cd = pywt.dwt(ca, wavelet)
        lca.append(len(ca))
        rca = scipy_resample(ca, N)
        rca -= np.mean(rca)
        rcd = scipy_resample(cd, N)
        rcd -= np.mean(rcd)
        pca_list.append(scipy_periodogram(rca, nfft=N)[1])
        pcd_list.append(scipy_periodogram(rcd, nfft=N)[1])

    ps = scipy_periodogram(signal, nfft=N)[1]
    L_ps = len(ps)
    n_levels = len(pca_list)  # = n - 1

    # --- Correlation between each detail spectrum and the signal spectrum.
    # L2-normalize each slice before corrcoef (matches reference).
    cpCD = np.empty(n_levels)
    for i in range(n_levels):
        L = max(1, min(int(np.round(L_ps / (2 ** i))), L_ps))
        pcd_i = pcd_list[i][:L]
        ps_i = ps[:L]
        cpCD[i] = float(np.corrcoef(
            pcd_i / (np.linalg.norm(pcd_i) + 1e-7),
            ps_i / (np.linalg.norm(ps_i) + 1e-7),
        )[0, 1])

    # --- Decorrelation index: threshold is -0.05 * mean(|diff|)
    diff_cpCD = np.diff(cpCD)
    if len(diff_cpCD) == 0:
        idx = len(lca) - 2  # single level: -1, Python negative indexing gives [0]
    else:
        mean_abs = float(np.mean(np.abs(diff_cpCD)))
        threshold = -0.05 * mean_abs
        candidates = np.where(diff_cpCD > threshold)[0]
        idx = int(candidates[0]) if len(candidates) > 0 else len(lca) - 2
        idx = min(idx, len(lca) - 2)  # clamp (reference: if idxCD > len(lca)-2)

    # --- DoF = CA_length * (1 + min(1, P_detail / P_approx))
    p_d = float(np.sum(pcd_list[idx + 1]))
    p_a = float(np.sum(pca_list[idx + 1]))
    ratio = min(1.0, p_d / p_a) if p_a > 0.0 else 0.0
    dof = float(lca[idx + 1]) * (1.0 + ratio)

    return 1.0 if np.isnan(dof) else max(1.0, dof)


def compute_wavelet_dof(
    C_row: np.ndarray,
    wavelet: str = "db5",
    events: np.ndarray | None = None,
) -> float:
    """
    Compute effective DoF for one synergy activation row.

    When ``events`` is provided the signal is split into cycles and DoF is
    computed independently for each cycle, then **summed** — exactly as in
    the reference implementation. Without ``events`` the whole row is treated
    as a single segment, which gives a different (incorrect) DoF scale.

    Parameters
    ----------
    C_row : np.ndarray, shape (N,)
        One row of the activation matrix C.
    wavelet : str
        Default 'db5' (Daubechies-5), matching the paper.
    events : 1D array-like of int, optional
        Segment-boundary sample indices (0-based). The i-th cycle is
        ``C_row[events[i] : events[i+1]]``. Provide at least 2 values.
        When omitted the whole row is treated as a single cycle.

    Returns
    -------
    n_dof : float
        Sum of per-cycle effective DoF (>= 1).
    """
    _check_pywt()
    arr = np.asarray(C_row, dtype=float)

    if events is None or len(events) < 2:
        return _compute_dof_segment(arr, wavelet)

    total = 0.0
    ev = np.asarray(events, dtype=int)
    for i in range(len(ev) - 1):
        seg = arr[ev[i]: ev[i + 1]]
        if len(seg) >= 4:
            total += _compute_dof_segment(seg, wavelet)
    return max(1.0, total)


def compute_total_dof(
    C: np.ndarray,
    wavelet: str = "db5",
    events: np.ndarray | None = None,
) -> float:
    """
    Sum effective DoF over all rows of the activation matrix C.

    Parameters
    ----------
    C : np.ndarray, shape (k x N)
    wavelet : str
    events : 1D array-like of int, optional

    Returns
    -------
    total_dof : float
    """
    return sum(
        compute_wavelet_dof(C[i], wavelet=wavelet, events=events)
        for i in range(C.shape[0])
    )
