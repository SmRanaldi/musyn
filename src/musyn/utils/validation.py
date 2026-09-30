"""Input validation helpers."""
from __future__ import annotations

import numpy as np


def check_signal_1d(x: object, name: str = "signal") -> np.ndarray:
    """
    Convert and validate a 1-D signal array.

    Parameters
    ----------
    x : array-like
        Input signal.
    name : str
        Variable name for error messages.

    Returns
    -------
    x : np.ndarray, shape (N,), dtype float64
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if x.size < 2:
        raise ValueError(f"{name} must have at least 2 samples")
    return x


def check_emg_matrix(
    D: object,
    name: str = "D",
    min_muscles: int = 2,
) -> np.ndarray:
    """
    Convert and validate a 2-D EMG matrix.

    Parameters
    ----------
    D : array-like, shape (M x N)
        EMG matrix with M channels and N time samples.
    name : str
        Variable name for error messages.
    min_muscles : int
        Minimum required number of channels.

    Returns
    -------
    D : np.ndarray, shape (M x N), dtype float64
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"{name} must be 2-D (M, N), got shape {D.shape}")
    if D.shape[0] < min_muscles:
        raise ValueError(
            f"{name} must have at least {min_muscles} channels (rows), "
            f"got {D.shape[0]}"
        )
    if D.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 time samples")
    if not np.isfinite(D).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if (D < 0).any():
        raise ValueError(
            f"{name} contains negative values; NMF requires a non-negative matrix. "
            "Pass the envelope (absolute value) of the EMG signal."
        )
    return D


def check_n_synergies(k: int, n_muscles: int, n_samples: int) -> None:
    """
    Validate the requested number of synergies.

    Parameters
    ----------
    k : int
        Requested number of synergies.
    n_muscles : int
        Number of EMG channels.
    n_samples : int
        Number of time samples.
    """
    max_k = min(n_muscles, n_samples)
    if not (1 <= k <= max_k):
        raise ValueError(
            f"n_synergies must be in [1, min(n_muscles, n_samples)] = [1, {max_k}], "
            f"got {k}"
        )
