"""Tests for quality metrics."""
import numpy as np
import pytest
from musyn.metrics.quality import cosine_similarity, quality_ratio, vaf, r_squared


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-10


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-10


def test_cosine_similarity_zero_vector():
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_vaf_perfect_reconstruction():
    W = np.eye(3)
    C = np.ones((3, 100))
    D = W @ C
    assert abs(vaf(D, W, C) - 1.0) < 1e-10


def test_vaf_is_finite(rng):
    """VAF with random W, C can be arbitrarily negative; just check finiteness."""
    D = np.abs(rng.normal(size=(6, 200)))
    W = np.abs(rng.normal(size=(6, 3)))
    C = np.abs(rng.normal(size=(3, 200)))
    v = vaf(D, W, C)
    assert np.isfinite(v)


def test_vaf_near_one_for_good_fit(rng):
    """VAF should be close to 1 when WC closely approximates D."""
    W = np.abs(rng.normal(size=(6, 3)))
    C = np.abs(rng.normal(size=(3, 200)))
    D = W @ C + rng.normal(size=(6, 200)) * 0.01
    D = np.abs(D)
    v = vaf(D, W, C)
    assert v > 0.5, f"VAF should be high for near-perfect fit, got {v:.3f}"


def test_r_squared_perfect_reconstruction():
    W = np.eye(3)
    C = np.ones((3, 100))
    D = W @ C
    assert abs(r_squared(D, W, C) - 1.0) < 1e-10


def test_quality_ratio_perfect(rng):
    """QR should be 1 when extracted synergies match true synergies."""
    W_true = np.abs(rng.normal(size=(6, 3)))
    W_true = W_true / np.linalg.norm(W_true, axis=0)
    qr = quality_ratio(W_true, W_true)
    assert abs(qr - 1.0) < 1e-6


def test_quality_ratio_range(rng):
    W_ext = np.abs(rng.normal(size=(6, 3)))
    W_true = np.abs(rng.normal(size=(6, 3)))
    qr = quality_ratio(W_ext, W_true)
    assert 0.0 <= qr <= 1.0
