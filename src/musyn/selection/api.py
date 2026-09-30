"""
High-level API for AIC-based synergy number selection.

References
----------
Ranaldi et al. (2021) "An Objective, Information-Based Approach for Selecting
the Number of Muscle Synergies via NNMF."
IEEE Trans Neural Syst Rehabil Eng.
"""
from __future__ import annotations

import numpy as np

from musyn.decomposition.init_strategies import InitStrategy
from musyn.selection.aic import aic_curve
from musyn.selection.criteria import SelectionMethod, select_synergy_count
from musyn.utils.validation import check_emg_matrix

# Methods that select from an AIC curve (and therefore need the expensive,
# wavelet-based DoF term computed by aic_curve/compute_total_dof). The other
# methods ('vaf', 'r2', 'plateau', 'surrogate') only consume the NMF
# solutions (W, C) -- see criteria.select_synergy_count's _needs_solutions.
_NEEDS_AIC_CURVE: frozenset[str] = frozenset({"min", "der", "firstpeak", "lastpeak"})


def select_synergy_number(
    M_matrix: np.ndarray | list,
    k_range: range | list[int] | None = None,
    method: SelectionMethod = "min",
    n_runs: int = 5,
    wavelet: str = "db5",
    init: InitStrategy = "sparse",
    events: np.ndarray | list | None = None,
    n_jobs: int = -1,
    return_full: bool = False,
    seed=None,
    **method_kwargs,
) -> int | dict:
    """
    Select the optimal number of muscle synergies via modified AIC.

    Implements the information-based criterion of Ranaldi et al. (2021).
    The AIC formula is:

      AIC(k) = L(k) + 2·k·N_M + 2·Σ_i DoF_i(k)

    where L is the per-channel squared reconstruction error normalised by
    signal variance + per-channel reconstruction noise, and DoF_i is the
    wavelet-based effective degrees of freedom for the i-th synergy activation.

    Parameters
    ----------
    M_matrix : array-like, shape (n_muscles x n_samples)
        Non-negative sEMG envelope matrix. Obtain from ``extract_envelope``.
    k_range : range, list[int], or None
        Synergy counts to evaluate. Default: ``range(1, n_muscles + 1)``.
    method : SelectionMethod
        Criterion for selecting from the AIC curve:
        - ``'min'``: global minimum (default, paper recommendation).
        - ``'der'``: first stationary point of the normalised derivative.
        - ``'firstpeak'``: first local minimum.
        - ``'lastpeak'``: last local minimum.
        - ``'vaf'``: first k where VAF >= 0.97.
        - ``'r2'``: first k where R² >= 0.95.
        - ``'plateau'``: first k where DVAF <= 5%.
        - ``'surrogate'``: surrogate-data baseline comparison.
    n_runs : int
        NMF restarts per k. Default 5.
    wavelet : str
        Wavelet for DoF estimation. Default 'db5' (Daubechies-5).
    init : InitStrategy
        NMF initialization. Default 'sparse'.
    events : 1D array-like of int, optional
        Segment-boundary sample indices (0-based) marking the start/end of
        each movement cycle or repetition, e.g. ``[0, 1200, 2400, 3600]``
        for three cycles of 1200 samples each.
        **Strongly recommended**: without events the wavelet DoF is computed
        on the full signal rather than per-cycle, which underestimates the
        penalty term and may suppress the ascending phase of the AIC curve.
        Unused (and never even computed) for 'vaf', 'r2', 'plateau', and
        'surrogate', which don't build an AIC curve at all -- see ``method``.
    n_jobs : int
        Parallel jobs for AIC k-sweep. Default -1 (all CPUs).
    return_full : bool
        If True, return a dict with the full AIC curve and solutions. Note:
        for 'vaf', 'r2', 'plateau', and 'surrogate' the AIC curve itself is
        never computed (see below), so ``aic_values`` comes back as None.
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
        ``k_opt``, ``aic_values``, ``k_range``, ``solutions``, ``method``.

    References
    ----------
    Ranaldi S, De Marchis C, Severini G, Conforto S (2021)
    "An Objective, Information-Based Approach for Selecting the Number
    of Muscle Synergies to be Extracted via Non-Negative Matrix
    Factorization."
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

    aic_values, solutions = aic_curve(
        M_matrix, k_range,
        wavelet=wavelet, n_runs=n_runs, init=init,
        events=events, n_jobs=n_jobs, rng=seed,
        compute_dof=method in _NEEDS_AIC_CURVE,
    )

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
            "method": method,
        }
    return k_opt
