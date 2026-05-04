"""Tests for NMF initialization strategies."""
import numpy as np
import pytest
from musyn.decomposition.init_strategies import init_rand, init_nsvd, init_sparse, get_init


def test_init_rand_shapes(rng):
    W, C = init_rand(8, 3000, 4, rng=rng)
    assert W.shape == (8, 4)
    assert C.shape == (4, 3000)


def test_init_rand_nonneg(rng):
    W, C = init_rand(8, 3000, 4, rng=rng)
    assert (W >= 0).all()
    assert (C >= 0).all()


def test_init_nsvd_shapes():
    D = np.abs(np.random.default_rng(0).normal(size=(8, 3000)))
    W, C = init_nsvd(D, 4)
    assert W.shape == (8, 4)
    assert C.shape == (4, 3000)


def test_init_nsvd_nonneg():
    D = np.abs(np.random.default_rng(0).normal(size=(8, 3000)))
    W, C = init_nsvd(D, 4)
    assert (W >= 0).all()
    assert (C >= 0).all()


def test_init_sparse_peak_structure(rng):
    """Each column of W should have at least one value > 0.5."""
    W, C = init_sparse(8, 3000, 4, rng=rng)
    for col in range(4):
        assert W[:, col].max() > 0.5, f"Column {col} has no peak element"


def test_init_sparse_background_small(rng):
    """Most elements of W should be small (< 0.1)."""
    W, C = init_sparse(8, 3000, 4, rng=rng)
    small_fraction = (W < 0.1).mean()
    assert small_fraction > 0.8


def test_get_init_dispatcher(rng):
    D = np.abs(rng.normal(size=(6, 500)))
    for strategy in ("rand", "nsvd", "sparse"):
        W, C = get_init(strategy, D, 3, rng=rng)
        assert W.shape == (6, 3)
        assert C.shape == (3, 500)
        assert (W >= 0).all()
        assert (C >= 0).all()
