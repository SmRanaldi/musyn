"""Tests for AR-based prewhitening."""
import numpy as np

from musyn.envelope.prewhiten import (
    estimate_ar_coefficients,
    whiten_signal,
)


def compute_autocorrelation(x, max_lag=20, biased=True):
    from musyn.envelope.prewhiten import compute_autocorrelation as _c
    return _c(x, max_lag, biased)


def test_autocorrelation_shape(rng):
    x = rng.standard_normal(1000)
    r = compute_autocorrelation(x, max_lag=20)
    assert r.shape == (21,)


def test_autocorrelation_r0_positive(rng):
    x = rng.standard_normal(1000)
    r = compute_autocorrelation(x)
    assert r[0] > 0


def test_ar_coefficients_shape(rng):
    x = rng.standard_normal(2000)
    a = estimate_ar_coefficients(x, order=12)
    assert a.shape == (12,)


def test_ar_coefficients_known_process(rng):
    """AR(2) process: x[t] = 0.8*x[t-1] - 0.2*x[t-2] + noise."""
    n = 10000
    x = np.zeros(n)
    noise = rng.standard_normal(n) * 0.1
    for t in range(2, n):
        x[t] = 0.8 * x[t - 1] - 0.2 * x[t - 2] + noise[t]
    a = estimate_ar_coefficients(x, order=2)
    assert abs(a[0] + 0.8) < 0.1, f"a1 should be ~-0.8, got {a[0]}"
    assert abs(a[1] - 0.2) < 0.1, f"a2 should be ~0.2, got {a[1]}"


def test_whiten_preserves_length(rng):
    x = rng.standard_normal(1000)
    coeffs = estimate_ar_coefficients(x, order=10)
    whitened = whiten_signal(x, coeffs)
    assert whitened.shape == x.shape


def test_prewhiten_reduces_autocorrelation(rng):
    """Whitened signal should have lag-1 autocorrelation closer to zero than original."""
    n = 5000
    x = np.zeros(n)
    noise = rng.standard_normal(n) * 0.1
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + noise[t]
    # Estimate AR on second half to avoid ramp-up transient
    from musyn.envelope.prewhiten import estimate_ar_coefficients, whiten_signal
    coeffs = estimate_ar_coefficients(x[2500:], order=12)
    whitened = whiten_signal(x, coeffs)
    r_orig = abs(np.corrcoef(x[:-1], x[1:])[0, 1])
    # Discard initial transient (first 100 samples) before measuring
    r_white = abs(np.corrcoef(whitened[100:-1], whitened[101:])[0, 1])
    assert r_white < r_orig, (
        f"Whitening should reduce lag-1 correlation; got {r_white:.3f} vs {r_orig:.3f}"
    )
