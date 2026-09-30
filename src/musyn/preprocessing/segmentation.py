"""
Envelope segmentation and normalisation utilities.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def segment_envelope(
    data: np.ndarray,
    events: list | np.ndarray,
    delay_on: int = 500,
    delay_off: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract envelope segments around movement events.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Envelope signal (rows = time samples, columns = channels).
    events : sequence of (start, stop) pairs
        Sample indices marking the onset and offset of each trial.
    delay_on : int
        Pre-event padding in samples. Default 500.
    delay_off : int
        Post-event padding in samples. Default 500.

    Returns
    -------
    segments : np.ndarray, shape (total_samples, n_channels)
        Concatenated trial segments.
    maxima : np.ndarray, shape (n_trials, n_channels)
        Per-trial, per-channel signal maximum.
    """
    segs, maxima = [], []
    for e in events:
        s = int(e[0]) - delay_on
        p = int(e[1]) + delay_off
        chunk = data[s:p]
        segs.append(chunk)
        maxima.append(np.max(chunk, axis=0).reshape(1, -1))
    return np.concatenate(segs, axis=0), np.concatenate(maxima, axis=0)


def time_normalize_envelope(
    data: np.ndarray,
    events: list | np.ndarray,
    n_samples: int = 100,
    delay_on: int = 100,
    delay_off: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-normalise envelope trials to a fixed number of samples.

    Each trial is interpolated to exactly ``n_samples`` points, enabling
    cross-trial averaging regardless of variable movement duration.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples_total, n_channels)
    events : sequence of (start, stop) pairs
    n_samples : int
        Target trial length in samples. Default 100.
    delay_on, delay_off : int
        Pre/post padding in samples. Default 500 each.

    Returns
    -------
    data_norm : np.ndarray, shape (n_trials * n_samples, n_channels)
    maxima : np.ndarray, shape (n_trials, n_channels)
    boundaries : np.ndarray, shape (n_trials + 1,)
        Sample indices separating trials in ``data_norm``.
    """
    segs, maxima, boundaries = [], [], [0]
    for e in events:
        s = int(e[0]) - delay_on
        p = int(e[1]) + delay_off
        trial = data[s:p]
        orig_x = np.linspace(0, trial.shape[0], trial.shape[0])
        new_x = np.linspace(0, trial.shape[0], n_samples)
        normalized = interp1d(orig_x, trial, axis=0)(new_x)
        segs.append(normalized)
        maxima.append(np.max(normalized, axis=0).reshape(1, -1))
        boundaries.append(boundaries[-1] + n_samples)
    return (
        np.concatenate(segs, axis=0),
        np.concatenate(maxima, axis=0),
        np.array(boundaries),
    )


def normalize_envelope(
    env: np.ndarray,
    maxima: np.ndarray,
    percentile: float = 80.0,
    floor: float = 1e-4,
) -> np.ndarray:
    """
    Amplitude-normalise envelope by a percentile of per-trial maxima.

    Parameters
    ----------
    env : np.ndarray, shape (n_samples, n_channels)
    maxima : np.ndarray, shape (n_trials, n_channels)
        Per-trial channel maxima (from ``segment_envelope``).
    percentile : float
        Normalisation percentile. Default 80.
    floor : float
        Minimum value after normalisation (avoids log(0)). Default 1e-4.

    Returns
    -------
    env_norm : np.ndarray, shape (n_samples, n_channels)
    """
    norm_factor = np.percentile(maxima, percentile, axis=0)
    env_norm = env / norm_factor
    env_norm[env_norm <= floor] = floor
    return env_norm
