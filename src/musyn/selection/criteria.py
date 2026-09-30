"""
Synergy number selection criteria applied to the AIC curve.

Implements all criteria compared in Ranaldi et al. (2021), Table I.

References
----------
Ranaldi et al. (2021), Section II-D and Table I.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import find_peaks, savgol_filter

SelectionMethod = Literal[
    "min", "der", "firstpeak", "lastpeak",
    "vaf", "r2", "plateau", "surrogate",
]


def select_min(aic_values: np.ndarray, k_range: list[int]) -> int:
    """
    Select k at the global minimum of the AIC curve.

    Parameters
    ----------
    aic_values : np.ndarray, shape (K,)
    k_range : list[int], length K

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: N_AIC selection criterion 'min' (Ranaldi et al. 2021).
    """
    return int(k_range[int(np.argmin(aic_values))])


def select_der(
    aic_values: np.ndarray,
    k_range: list[int],
    smooth_window: int = 3,
) -> int:
    """
    Select k at the first stationary point of the AIC curve derivative.

    Normalises the first derivative by the maximum previous absolute
    change, then returns the first k where it crosses the derivative
    threshold (-0.005 as in the MATLAB findNSynAIC.m).

    Parameters
    ----------
    aic_values : np.ndarray, shape (K,)
    k_range : list[int]
    smooth_window : int
        Savitzky-Golay filter window for smoothing. Odd, ≥ 3.

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: findNSynAIC.m criterion 'der', threshold -0.005
    (Ranaldi et al. 2021 / NSyn_Criteria MATLAB code).
    """
    aic = np.asarray(aic_values, dtype=float)
    if smooth_window >= 3 and len(aic) >= smooth_window:
        aic_s = savgol_filter(aic, window_length=smooth_window, polyorder=2)
    else:
        aic_s = aic
    diff = np.diff(aic_s)
    # Normalise each diff by the max abs previous change
    norm_diff = np.empty_like(diff)
    for i, d in enumerate(diff):
        max_prev = np.max(np.abs(diff[:i + 1])) if i >= 0 else 1.0
        norm_diff[i] = d / max_prev if max_prev != 0 else d
    # First index where normalised derivative >= -0.005 (first plateau)
    candidates = np.where(norm_diff >= -0.005)[0]
    idx = int(candidates[0]) if len(candidates) > 0 else len(k_range) - 1
    return int(k_range[idx])


def select_firstpeak(aic_values: np.ndarray, k_range: list[int]) -> int:
    """
    Select k at the first local minimum (negative peak) of the AIC curve.

    Parameters
    ----------
    aic_values : np.ndarray
    k_range : list[int]

    Returns
    -------
    k_opt : int
    """
    peaks, _ = find_peaks(-np.asarray(aic_values))
    if len(peaks) == 0:
        return select_min(aic_values, k_range)
    return int(k_range[int(peaks[0])])


def select_lastpeak(aic_values: np.ndarray, k_range: list[int]) -> int:
    """
    Select k at the last local minimum (negative peak) of the AIC curve.

    Parameters
    ----------
    aic_values : np.ndarray
    k_range : list[int]

    Returns
    -------
    k_opt : int
    """
    peaks, _ = find_peaks(-np.asarray(aic_values))
    if len(peaks) == 0:
        return select_min(aic_values, k_range)
    return int(k_range[int(peaks[-1])])


def select_vaf_threshold(
    M_matrix: np.ndarray,
    solutions: list[tuple[np.ndarray, np.ndarray]],
    k_range: list[int],
    vaf_threshold: float = 0.97,
) -> int:
    """
    Select first k where global VAF ≥ vaf_threshold.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M x N)
    solutions : list of (W; C) tuples
    k_range : list[int]
    vaf_threshold : float
        Default 0.97 (optimised in Ranaldi et al. 2021).

    Returns
    -------
    k_opt : int
    """
    from musyn.metrics.quality import vaf as compute_vaf
    for k, (W, C) in zip(k_range, solutions):
        if compute_vaf(M_matrix, W, C) >= vaf_threshold:
            return int(k)
    return int(k_range[-1])


def select_r2_threshold(
    M_matrix: np.ndarray,
    solutions: list[tuple[np.ndarray, np.ndarray]],
    k_range: list[int],
    r2_threshold: float = 0.95,
) -> int:
    """
    Select first k where mean per-channel R² ≥ r2_threshold.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M x N)
    solutions : list of (W; C)
    k_range : list[int]
    r2_threshold : float
        Default 0.95 (optimised in Ranaldi et al. 2021).

    Returns
    -------
    k_opt : int
    """
    from musyn.metrics.quality import r_squared as compute_r2
    for k, (W, C) in zip(k_range, solutions):
        if compute_r2(M_matrix, W, C) >= r2_threshold:
            return int(k)
    return int(k_range[-1])


def select_plateau(
    M_matrix: np.ndarray,
    solutions: list[tuple[np.ndarray, np.ndarray]],
    k_range: list[int],
    delta_vaf: float = 0.05,
) -> int:
    """
    Select first k where the VAF increment from k to k+1 drops below 5%.

    Matches the N_5% criterion of Ranaldi et al. (2021) / ``nSyn5Perc.m``
    in the NSyn_Criteria MATLAB repository:
    ``first k where VAF(k+1) - VAF(k) <= delta_vaf``

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M x N)
        Envelope matrix (needed to compute VAF from solutions).
    solutions : list of (W; C) tuples
        NMF solutions for each k in k_range.
    k_range : list[int]
    delta_vaf : float
        VAF increment threshold. Default 0.05 (5%).

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: N_5% — "first k such that VAF(k+1) - VAF(k) ≤ 0.05"
    (Ranaldi et al. 2021, Table I; ``nSyn5Perc.m``).
    """
    from musyn.metrics.quality import vaf as compute_vaf
    vaf_curve = np.array([compute_vaf(M_matrix, W, C) for W, C in solutions])
    diffs = np.diff(vaf_curve)
    candidates = np.where(diffs <= delta_vaf)[0]
    if len(candidates) == 0:
        return int(k_range[-1])
    # Return the k at which the increment first falls below threshold
    return int(k_range[int(candidates[0])])


def select_surrogate(
    M_matrix: np.ndarray,
    solutions: list[tuple[np.ndarray, np.ndarray]],
    k_range: list[int],
    n_surrogates: int = 50,
    surrogate_fraction: float = 0.75,
    n_runs: int = 3,
    rng=None,
) -> int:
    """
    Surrogate-baseline VAF criterion (N_SURR).

    Matches ``nSynRand.m`` from the NSyn_Criteria MATLAB repository:
      1. Compute real VAF(k) from the provided solutions.
      2. For each of ``n_surrogates`` column-shuffled versions of M,
         run NMF for every k and compute surrogate VAF(k).
      3. ``ths = mean( diff( mean_surrogate_VAF ) )``
         (mean per-step VAF increment expected by chance).
      4. Return first k where ``diff(VAF_real)[k] <= surrogate_fraction * ths``.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M x N)
    solutions : list of (W; C) tuples
        Real-data NMF solutions for each k in k_range.
    k_range : list[int]
    n_surrogates : int
        Number of column-shuffled surrogates. Default 50.
    surrogate_fraction : float
        Fraction of the surrogate threshold to use (paper: 0.75). Default 0.75.
    n_runs : int
        NMF restarts per k per surrogate. Default 3 (kept low for speed).
    rng : int, np.random.Generator, or None

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: N_SURR criterion (Ranaldi et al. 2021, Table I).
    MATLAB: ``nSynRand.m`` — ``min(find(diff(VAFCurve) <= 0.75*ths))``
    where ``ths = mean(diff(VAFRand))``.
    """
    from musyn.decomposition.nnmf import run_nnmf_multi
    from musyn.metrics.quality import vaf as compute_vaf

    rng_obj = np.random.default_rng(rng)

    # Real VAF curve
    vaf_real = np.array([compute_vaf(M_matrix, W, C) for W, C in solutions])

    # Surrogate VAF curves: shape (n_surrogates, len(k_range))
    surr_vafs = np.empty((n_surrogates, len(k_range)))
    for s in range(n_surrogates):
        surr = M_matrix[:, rng_obj.permutation(M_matrix.shape[1])]
        for i, k in enumerate(k_range):
            W_s, C_s, _ = run_nnmf_multi(surr, k, n_runs=n_runs, rng=rng_obj)
            surr_vafs[s, i] = compute_vaf(surr, W_s, C_s)

    # Mean surrogate VAF curve, then mean increment
    mean_surr_vaf = surr_vafs.mean(axis=0)
    ths = float(np.mean(np.diff(mean_surr_vaf)))

    # Find first k where real increment <= 75% of chance increment
    diffs_real = np.diff(vaf_real)
    candidates = np.where(diffs_real <= surrogate_fraction * ths)[0]
    if len(candidates) == 0:
        return int(k_range[-1])
    return int(k_range[int(candidates[0])])


def select_synergy_count(
    aic_values: np.ndarray,
    k_range: list[int],
    method: SelectionMethod,
    M_matrix: np.ndarray | None = None,
    solutions: list[tuple[np.ndarray, np.ndarray]] | None = None,
    **kwargs,
) -> int:
    """
    Select the optimal synergy count using the specified criterion.

    Parameters
    ----------
    aic_values : np.ndarray
        AIC values for each k. Required for 'min', 'der', 'firstpeak', 'lastpeak'.
    k_range : list[int]
    method : SelectionMethod
    M_matrix : np.ndarray, optional
        Envelope matrix. Required for 'vaf', 'r2', 'plateau', 'surrogate'.
    solutions : list of (W; C), optional
        NMF solutions per k. Required for 'vaf', 'r2', 'plateau', 'surrogate'.
    **kwargs
        Forwarded to the individual criterion function.

    Returns
    -------
    k_opt : int
    """
    _needs_solutions = {"vaf", "r2", "plateau", "surrogate"}
    if method in _needs_solutions:
        if M_matrix is None or solutions is None:
            raise ValueError(
                f"method='{method}' requires both M_matrix and solutions."
            )

    if method == "min":
        return select_min(aic_values, k_range)
    elif method == "der":
        return select_der(aic_values, k_range, **kwargs)
    elif method == "firstpeak":
        return select_firstpeak(aic_values, k_range)
    elif method == "lastpeak":
        return select_lastpeak(aic_values, k_range)
    elif method == "vaf":
        return select_vaf_threshold(M_matrix, solutions, k_range, **kwargs)
    elif method == "r2":
        return select_r2_threshold(M_matrix, solutions, k_range, **kwargs)
    elif method == "plateau":
        return select_plateau(M_matrix, solutions, k_range, **kwargs)
    elif method == "surrogate":
        return select_surrogate(M_matrix, solutions, k_range, **kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'min', 'der', 'firstpeak', 'lastpeak', "
            "'vaf', 'r2', 'plateau', 'surrogate'."
        )
