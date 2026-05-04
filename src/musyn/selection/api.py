"""
High-level API for AIC-based synergy number selection.

References
----------
Ranaldi et al. (2021) "An Objective, Information-Based Approach for Selecting
the Number of Muscle Synergies via NNMF."
IEEE Trans Neural Syst Rehabil Eng.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from musyn.decomposition.init_strategies import InitStrategy
from musyn.selection.noise import estimate_measurement_noise, total_noise_variance
from musyn.selection.aic import aic_curve
from musyn.selection.criteria import SelectionMethod, select_synergy_count
from musyn.utils.validation import check_emg_matrix


def select_synergy_number(
    M_matrix: Union[np.ndarray, list],
    k_range: Union[range, list[int], None] = None,
    method: SelectionMethod = "min",
    sdn_c: float = 0.175,
    n_runs: int = 5,
    wavelet: str = "db5",
    init: InitStrategy = "sparse",
    n_jobs: int = -1,
    return_full: bool = False,
    seed=None,
    **method_kwargs,
) -> Union[int, dict]:
    """
    Select the optimal number of muscle synergies via modified AIC.

    Implements the information-based criterion of Ranaldi et al. (2021).
    Fixes two critical problems of classical AIC applied to NMF:
      1. Uses signal-dependent noise in the log-likelihood.
      2. Uses wavelet-based effective DoF instead of parameter count.

    Parameters
    ----------
    M_matrix : array-like, shape (n_muscles, n_samples)
        Non-negative sEMG envelope matrix. Obtain from ``extract_envelope``.
    k_range : range, list[int], or None
        Synergy counts to evaluate. Default: ``range(1, n_muscles + 1)``.
    method : SelectionMethod
        Criterion for selecting from the AIC curve:
        - ``'min'``: global minimum (default, paper recommendation).
        - ``'der'``: first stationary point of the normalised derivative.
        - ``'firstpeak'``: first local minimum.
        - ``'lastpeak'``: last local minimum.
        - ``'vaf'``: first k where VAF ≥ 0.97.
        - ``'r2'``: first k where R² ≥ 0.95.
        - ``'plateau'``: first k where ΔAIC ≤ 5%.
        - ``'surrogate'``: surrogate-data baseline comparison.
    sdn_c : float
        Signal-dependent noise coefficient c ∈ [0.1, 0.25].
        Default 0.175 (midpoint of paper's range).
    n_runs : int
        NMF restarts per k. Default 5.
    wavelet : str
        Wavelet for DoF estimation. Default 'db5' (Daubechies-5).
    init : InitStrategy
        NMF initialization. Default 'sparse'.
    n_jobs : int
        Parallel jobs for AIC k-sweep. Default -1 (all CPUs).
    return_full : bool
        If True, return a dict with the full AIC curve and solutions.
    seed : int or None
        Random seed.
    **method_kwargs
        Extra keyword arguments forwarded to the selection criterion
        (e.g., ``vaf_threshold=0.95`` for 'vaf' method).

    Returns
    -------
    k_opt : int
        Optimal number of synergies.
    result : dict, optional
        Only if ``return_full=True``. Keys:
        ``k_opt``, ``aic_values``, ``k_range``, ``solutions``,
        ``sigma2``, ``method``.

    References
    ----------
    Ranaldi S, Severini G, Bibbo D, Conforto S, De Marchis C (2021)
    "An Objective, Information-Based Approach for Selecting the Number
    of Muscle Synergies via NNMF."
    IEEE Trans Neural Syst Rehabil Eng.

    Examples
    --------
    >>> import numpy as np
    >>> import musyn
    >>> rng = np.random.default_rng(42)
    >>> D = np.abs(rng.normal(size=(8, 2000)))
    >>> k = musyn.select_synergy_number(D, k_range=range(1, 6))
    >>> isinstance(k, int)
    True
    """
    M_matrix = check_emg_matrix(M_matrix)
    n_muscles = M_matrix.shape[0]

    if k_range is None:
        k_range = list(range(1, n_muscles + 1))
    else:
        k_range = list(k_range)

    # Estimate noise
    sigma2_M = estimate_measurement_noise(M_matrix)
    sigma2 = total_noise_variance(M_matrix, sigma2_M, c=sdn_c)

    # Compute AIC curve (parallelized over k)
    aic_values, solutions = aic_curve(
        M_matrix, k_range, sigma2,
        wavelet=wavelet, n_runs=n_runs, init=init,
        n_jobs=n_jobs, rng=seed,
    )

    # Select k
    k_opt = select_synergy_count(
        aic_values, k_range, method,
        M_matrix=M_matrix,
        solutions=solutions,
        **method_kwargs,
    )

    if return_full:
        return {
            "k_opt": k_opt,
            "aic_values": aic_values,
            "k_range": k_range,
            "solutions": solutions,
            "sigma2": sigma2,
            "method": method,
        }
    return k_opt
