"""
Single-channel ICA ECG removal for surface EMG.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


def _hankel_embedding(signal: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Time-delay (Hankel) embedding: (n_samples,) → (n_rows, embedding_dim)."""
    n = len(signal)
    n_rows = n - embedding_dim + 1
    X = np.zeros((n_rows, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = signal[i: n_rows + i]
    return X


def _hankel_inverse(X: np.ndarray, original_length: int) -> np.ndarray:
    """Collapse embedded matrix back to 1D via diagonal averaging."""
    n_rows, n_cols = X.shape
    out = np.zeros(original_length)
    count = np.zeros(original_length)
    for i in range(n_cols):
        out[i: i + n_rows] += X[:, i]
        count[i: i + n_rows] += 1
    return out / np.maximum(count, 1)


def remove_ecg_scica(
    signal: np.ndarray,
    fs: float = 1000.0,
    embedding_dim: int = 40,
) -> np.ndarray:
    """
    Remove ECG artefact from a single-channel sEMG signal via SC-ICA.

    Algorithm:
      1. Embed the 1-D signal into a 2-D Hankel matrix using time-delay copies.
      2. Decompose with FastICA (up to 10 components).
      3. Identify the ECG component as the one with maximum excess kurtosis.
      4. Zero out the ECG component and reconstruct via diagonal averaging.

    Requires ``scikit-learn`` (``pip install musyn[ecg]``).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Single-channel sEMG signal (should be filtered before ECG removal).
    fs : float
        Sampling frequency in Hz.
    embedding_dim : int
        Number of lagged copies (Hankel columns).
        Rule of thumb: ``embedding_dim / fs`` should cover one QRS complex
        (≈ 4–20 ms). Default 40 samples (40 ms at 1 kHz).

    Returns
    -------
    signal_clean : np.ndarray, shape (n_samples,)

    Examples
    --------
    >>> import numpy as np
    >>> from musyn.preprocessing import remove_ecg_scica
    >>> sig = np.random.randn(5000)
    >>> clean = remove_ecg_scica(sig, fs=1000.0, embedding_dim=int(0.005*1000))
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ECG removal. "
            "Install it with: pip install musyn[ecg]"
        ) from exc

    X = _hankel_embedding(signal, embedding_dim)
    n_components = min(embedding_dim, 10)
    ica = FastICA(n_components=n_components, random_state=42, whiten="unit-variance")
    S = ica.fit_transform(X)
    A = ica.mixing_

    ecg_idx = int(np.argmax(kurtosis(S, axis=0, fisher=True)))
    S_clean = S.copy()
    S_clean[:, ecg_idx] = 0.0

    X_clean = np.dot(S_clean, A.T) + ica.mean_
    return _hankel_inverse(X_clean, len(signal))
