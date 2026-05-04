"""
Shared fixtures for the musyn test suite.

Synthetic EMG data is generated to mirror the MATLAB simulation scripts
in SmRanaldi/NSyn_Criteria (generateEMGSynergies.m, generateSyn.m).
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_hann_activation(n_samples: int, width: int, offset: int, rng) -> np.ndarray:
    """Hann-windowed activation pattern with temporal offset."""
    h = np.hanning(width)
    c = np.zeros(n_samples)
    start = offset % n_samples
    end = min(start + width, n_samples)
    c[start:end] = h[: end - start]
    return c


@pytest.fixture
def synthetic_emg_1d(rng):
    """
    Single-channel amplitude-modulated Gaussian noise.

    Ground-truth envelope = smooth Gaussian bump.
    Returns (signal, envelope_true), shape (4000,).
    """
    n = 4000
    t = np.linspace(0, 4.0, n)
    envelope_true = 0.5 * np.exp(-((t - 2.0) ** 2) / 0.5)
    noise = rng.standard_normal(n)
    signal = envelope_true * noise
    return signal, envelope_true


@pytest.fixture
def synthetic_emg_matrix(rng):
    """
    Synthetic D = W_true @ C_true + noise.

    Parameters mirror MATLAB generateSyn.m:
    - M=8 muscles, k=3 synergies, N=3000 samples
    - W_true: random nonneg, max cross-column cosine < 0.6
    - C_true: Hann-windowed activations

    Returns (D, W_true, C_true).
    """
    M, k, N = 8, 3, 3000
    snr_db = 15.0

    # Generate W_true with low cross-column correlation
    while True:
        W = rng.uniform(0.0, 1.0, size=(M, k))
        W = W / W.sum(axis=0)  # normalise columns
        cosines = np.abs(
            (W.T @ W) / (np.linalg.norm(W, axis=0)[:, None] * np.linalg.norm(W, axis=0)[None, :])
        )
        np.fill_diagonal(cosines, 0.0)
        if cosines.max() < 0.6:
            break

    # Generate C_true: Hann activations per synergy
    C = np.zeros((k, N))
    width = 200
    offsets = [0, N // 3, 2 * N // 3]
    for i, off in enumerate(offsets):
        C[i] = _make_hann_activation(N, width, off, rng)

    # Compose clean signal and add noise
    D_clean = W @ C
    signal_power = np.mean(D_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.uniform(0.0, 1.0, size=(M, N)) * np.sqrt(noise_power * 3.0)
    D = D_clean + noise
    D = np.maximum(D, 0.0)  # ensure non-negative (envelope)

    return D, W, C
