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
    M_matrix : np.ndarray, shape (M, N)
    solutions : list of (W, C) tuples
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
    M_matrix : np.ndarray, shape (M, N)
    solutions : list of (W, C)
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
    aic_values: np.ndarray,
    k_range: list[int],
    plateau_fraction: float = 0.05,
) -> int:
    """
    Select first k where the AIC improvement is < 5% of the total range.

    Parameters
    ----------
    aic_values : np.ndarray
    k_range : list[int]
    plateau_fraction : float
        Default 0.05 (5%).

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: N_5% criterion — first k where ΔVAF ≤ 5% (Ranaldi et al. 2021,
    Table I). Here adapted to AIC: ΔAIC ≤ 5% of total AIC range.
    """
    aic = np.asarray(aic_values, dtype=float)
    total_range = float(aic.max() - aic.min())
    if total_range == 0:
        return int(k_range[0])
    diffs = np.abs(np.diff(aic))
    candidates = np.where(diffs <= plateau_fraction * total_range)[0]
    if len(candidates) == 0:
        return int(k_range[-1])
    return int(k_range[int(candidates[0])])


def select_surrogate(
    M_matrix: np.ndarray,
    aic_values: np.ndarray,
    k_range: list[int],
    n_surrogates: int = 100,
    alpha: float = 0.05,
    rng=None,
) -> int:
    """
    Surrogate baseline: select first k where real AIC is significantly
    lower than the surrogate (column-shuffled) distribution.

    Parameters
    ----------
    M_matrix : np.ndarray, shape (M, N)
    aic_values : np.ndarray
    k_range : list[int]
    n_surrogates : int
        Number of surrogate shuffles. Default 100.
    alpha : float
        Significance level. Default 0.05.
    rng : seed

    Returns
    -------
    k_opt : int

    Notes
    -----
    Paper: N_SURR criterion (Ranaldi et al. 2021, Table I).
    """
    from musyn.selection.noise import estimate_measurement_noise, total_noise_variance
    from musyn.selection.aic import compute_aic_for_k
    from scipy.stats import ttest_1samp

    rng_obj = np.random.default_rng(rng)
    n_muscles = M_matrix.shape[0]
    sigma2_M = estimate_measurement_noise(M_matrix)
    sigma2 = total_noise_variance(M_matrix, sigma2_M)

    for i, k in enumerate(k_range):
        # Compute surrogate AIC distribution
        surr_aics = []
        for _ in range(n_surrogates):
            surr = M_matrix[:, rng_obj.permutation(M_matrix.shape[1])]
            surr_aic, _, _ = compute_aic_for_k(
                surr, k, sigma2, n_muscles,
                n_runs=3, rng=rng_obj,
            )
            surr_aics.append(surr_aic)
        # One-sided test: real AIC < surrogate distribution?
        stat, p_val = ttest_1samp(surr_aics, aic_values[i], alternative="greater")
        if p_val < alpha:
            return int(k)

    return int(k_range[-1])


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
    k_range : list[int]
    method : SelectionMethod
    M_matrix : np.ndarray, optional
        Required for 'vaf', 'r2', 'surrogate'.
    solutions : list of (W, C), optional
        Required for 'vaf', 'r2'.
    **kwargs
        Passed to the individual selection function.

    Returns
    -------
    k_opt : int
    """
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
        return select_plateau(aic_values, k_range, **kwargs)
    elif method == "surrogate":
        return select_surrogate(M_matrix, aic_values, k_range, **kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'min', 'der', 'firstpeak', 'lastpeak', "
            "'vaf', 'r2', 'plateau', 'surrogate'."
        )
