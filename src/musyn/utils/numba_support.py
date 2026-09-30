"""
Conditional Numba import with graceful fallback.

The package uses a three-tier strategy for the adaptive envelope hot loop:
  1. Cython C extension (_loop.so) — fastest, compiled at install
  2. Numba JIT (@njit, cache=True) — fast, compiled on first call
  3. NumPy vectorized — always available, moderate speedup

This module handles tier 2 and exposes the identity-decorator fallback
so callers can unconditionally write ``@njit(cache=True)`` without
checking availability at the definition site.
"""
from __future__ import annotations

from collections.abc import Callable

try:
    import numba  # noqa: F401
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide no-op decorator so @njit-decorated functions work without Numba
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn: Callable) -> Callable:
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

    prange = range  # type: ignore[assignment]


def numba_available() -> bool:
    """Return True if numba is importable."""
    return NUMBA_AVAILABLE


def get_adaptive_loop() -> tuple[Callable, str]:
    """
    Return the best available non-Cython adaptive loop implementation.

    Returns
    -------
    fn : callable
        The adaptive loop function.
    backend : str
        One of ``'numba'`` or ``'numpy'``.
    """
    if NUMBA_AVAILABLE:
        from musyn.envelope._numba_loop import adaptive_loop_nb
        return adaptive_loop_nb, "numba"
    else:
        from musyn.envelope._numpy_loop import adaptive_loop_np
        return adaptive_loop_np, "numpy"
