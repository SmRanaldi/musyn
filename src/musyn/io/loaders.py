"""I/O loaders for common EMG data formats."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_csv(
    path: str | Path,
    fs: float = 1000.0,
    channel_axis: int = 0,
    **read_csv_kwargs,
) -> tuple[np.ndarray, float]:
    """
    Load EMG data from a CSV file.

    Parameters
    ----------
    path : str or Path
    fs : float
        Sampling frequency (not stored in CSV; must be provided).
    channel_axis : int
        0 = rows are channels (M, N); 1 = columns are channels (N, M).
    **read_csv_kwargs
        Forwarded to ``numpy.loadtxt``.

    Returns
    -------
    data : np.ndarray, shape (M x N)
    fs : float
    """
    data = np.loadtxt(path, delimiter=",", **read_csv_kwargs)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if channel_axis == 1:
        data = data.T
    return data.astype(np.float64), fs


def load_mat(
    path: str | Path,
    data_key: str = "emg",
    fs_key: str = "fs",
) -> tuple[np.ndarray, float]:
    """
    Load EMG data from a MATLAB .mat file.

    Parameters
    ----------
    path : str or Path
    data_key : str
        Key in the .mat file for the EMG data array.
    fs_key : str
        Key in the .mat file for the sampling frequency.

    Returns
    -------
    data : np.ndarray, shape (M x N)
    fs : float
    """
    from scipy.io import loadmat
    mat = loadmat(str(path))
    data = np.asarray(mat[data_key], dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    fs = float(mat[fs_key].flat[0]) if fs_key in mat else 1000.0
    return data, fs


def load_npz(path: str | Path) -> dict:
    """
    Load data from a NumPy .npz archive.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    data_dict : dict
        All arrays stored in the archive.
    """
    archive = np.load(str(path), allow_pickle=False)
    return dict(archive)
