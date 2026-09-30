"""I/O writers for saving analysis results."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """
    Save arrays to a compressed NumPy .npz archive.

    Parameters
    ----------
    path : str or Path
    **arrays
        Keyword arguments become array names in the archive.

    Examples
    --------
    >>> save_npz("results.npz", W=W, C=C, envelope=env)
    """
    np.savez_compressed(str(path), **arrays)


def save_results(
    path: str | Path,
    W: np.ndarray,
    C: np.ndarray,
    envelope: np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Save synergy extraction results to .npz.

    Parameters
    ----------
    path : str or Path
    W : np.ndarray, shape (M x k)
    C : np.ndarray, shape (k x N)
    envelope : np.ndarray, optional
    metadata : dict, optional
        Scalar metadata (n_synergies, fs, etc.) stored as 0-D arrays.
    """
    arrays = {"W": W, "C": C}
    if envelope is not None:
        arrays["envelope"] = envelope
    if metadata:
        for key, val in metadata.items():
            arrays[f"meta_{key}"] = np.asarray(val)
    save_npz(path, **arrays)
