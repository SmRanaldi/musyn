"""Tests for NMF multiplicative update rules."""
import numpy as np

from musyn.decomposition.updates import reconstruction_error, update_C, update_W


def test_update_W_nonneg(rng):
    D = np.abs(rng.normal(size=(6, 200)))
    W = np.abs(rng.normal(size=(6, 3))) + 1e-6
    C = np.abs(rng.normal(size=(3, 200))) + 1e-6
    W_new = update_W(D, W, C)
    assert (W_new >= 0).all()


def test_update_C_nonneg(rng):
    D = np.abs(rng.normal(size=(6, 200)))
    W = np.abs(rng.normal(size=(6, 3))) + 1e-6
    C = np.abs(rng.normal(size=(3, 200))) + 1e-6
    C_new = update_C(D, W, C)
    assert (C_new >= 0).all()


def test_error_decreases_after_update(rng):
    """Reconstruction error should not increase after one update step."""
    D = np.abs(rng.normal(size=(6, 200)))
    W = np.abs(rng.normal(size=(6, 3))) + 0.1
    C = np.abs(rng.normal(size=(3, 200))) + 0.1
    err_before = reconstruction_error(D, W, C)
    W = update_W(D, W, C)
    C = update_C(D, W, C)
    err_after = reconstruction_error(D, W, C)
    assert err_after <= err_before + 1e-6, (
        f"Error increased: {err_before:.4f} -> {err_after:.4f}"
    )


def test_reconstruction_error_zero_for_perfect_fit():
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    D = W @ C
    assert reconstruction_error(D, W, C) < 1e-10
