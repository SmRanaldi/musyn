"""
High-level API for sEMG envelope extraction.

References
----------
Ranaldi et al. (2018) "An automatic, adaptive, information-based algorithm
for the extraction of the sEMG envelope."
Journal of Electromyography and Kinesiology.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from musyn.envelope.adaptive import adaptive_envelope
from musyn.envelope.detection import initialize_window_lengths
from musyn.envelope.prewhiten import prewhiten
from musyn.utils.validation import check_signal_1d


def _process_single_channel(
    signal: np.ndarray,
    fs: float,
    ar_order: int,
    alpha: float,
    nu: int,
    max_iter: int,
    convergence_threshold: float,
    chi2_alpha: float,
    w_min: int,
    w_max: int,
    init_ms: float,
) -> tuple[np.ndarray, dict]:
    """Process one channel through the full envelope pipeline."""
    whitened = prewhiten(signal, order=ar_order)
    w_init = initialize_window_lengths(len(signal), fs, w_min, w_max, init_ms)
    envelope, info = adaptive_envelope(
        whitened, w_init, alpha=alpha, nu=float(nu),
        max_iter=max_iter, convergence_threshold=convergence_threshold,
        chi2_alpha=chi2_alpha, w_min=w_min, w_max=w_max,
    )
    return envelope, info


def extract_envelope(
    signal: np.ndarray | list,
    fs: float = 1000.0,
    ar_order: int = 12,
    alpha: float = 1.0,
    nu: int = 2,
    max_iter: int = 100,
    convergence_threshold: float = 0.95,
    chi2_alpha: float = 0.05,
    w_min: int = 1,
    w_max: int = 10000,
    init_ms: float = 500.0,
    n_jobs: int = -1,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """
    Extract the sEMG amplitude envelope using the adaptive algorithm.

    Implements the information-based iterative algorithm of Ranaldi et al.
    (2018). For each sample, an optimal window length M_k is estimated from
    the local signal statistics, and the envelope is computed as a running
    RMS with that adaptive window.

    Steps (per channel):
      1. Pre-whiten via AR(``ar_order``) Yule-Walker (``prewhiten``).
      2. Nu-order detection: ``|signal|^nu`` (``nu_order_detection``).
      3. Initialise window lengths to ``init_ms`` ms.
      4. Iterate: compute derivatives → M_k → re-linearise → check entropy
         convergence.

    Parameters
    ----------
    signal : array-like, shape (N,) or (M x N)
        Single-channel (1-D) or M-channel (2-D) sEMG signal.
        Values need not be rectified; the algorithm handles this internally.
    fs : float
        Sampling frequency in Hz. Default 1000.
    ar_order : int
        AR model order for pre-whitening. Paper uses 11; default 12.
    alpha : float
        Signal model shape parameter. Paper fixes alpha=1. Default 1.0.
    nu : int
        Detection order. Paper fixes nu=2. Default 2.
    max_iter : int
        Maximum number of adaptive iterations per channel. Default 100.
    convergence_threshold : float
        Stop early when this fraction of samples has converged. Default 0.95.
    chi2_alpha : float
        Significance level for chi-squared convergence test (NumPy backend
        only). Default 0.05.
    w_min : int
        Minimum adaptive window length in samples. Default 1.
    w_max : int
        Maximum adaptive window length in samples. Default 10000.
    init_ms : float
        Initial window length in milliseconds. Default 500 ms. The paper
        and ``initialize_window_lengths`` (this parameter's default when
        called directly) use 100 ms; this wrapper's default is 500 ms.
    n_jobs : int
        Number of parallel jobs for multi-channel processing.
        ``-1`` = use all CPUs. Ignored for 1-D input. Default -1.
    return_info : bool
        If True, also return a list of info dicts (one per channel).

    Returns
    -------
    envelope : np.ndarray, shape (N,) or (M x N)
        Estimated amplitude envelope. Same shape as input.
    info : list[dict], optional
        Only returned when ``return_info=True``. Each dict contains:
        ``iterations``, ``converged``, ``backend``,
        ``window_lengths`` (ndarray, shape (N,) — optimal M_k per sample).

    References
    ----------
    Ranaldi S, De Marchis C, Conforto S (2018) "An automatic, adaptive,
    information-based algorithm for the extraction of the sEMG envelope."
    J Electromyogr Kinesiol.

    Examples
    --------
    >>> import numpy as np
    >>> import musyn
    >>> rng = np.random.default_rng(0)
    >>> signal = rng.normal(size=4000)
    >>> env = musyn.extract_envelope(signal, fs=1000.0)
    >>> env.shape
    (4000,)
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim == 1:
        signal = check_signal_1d(signal)
        envelope, info = _process_single_channel(
            signal, fs, ar_order, alpha, nu, max_iter,
            convergence_threshold, chi2_alpha, w_min, w_max, init_ms,
        )
        if return_info:
            return envelope, [info]
        return envelope

    elif signal.ndim == 2:
        m_channels, n_samples = signal.shape
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_channel)(
                signal[ch], fs, ar_order, alpha, nu, max_iter,
                convergence_threshold, chi2_alpha, w_min, w_max, init_ms,
            )
            for ch in range(m_channels)
        )
        envelopes = np.stack([r[0] for r in results], axis=0)
        infos = [r[1] for r in results]
        if return_info:
            return envelopes, infos
        return envelopes

    else:
        raise ValueError(
            f"signal must be 1-D (N,) or 2-D (M, N), got shape {signal.shape}"
        )
