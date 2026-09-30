# `musyn.utils`

Input validation and backend-selection helpers shared across the package.
Mostly useful if you're extending musyn or debugging backend selection
rather than for everyday analysis.

## Validation

```{eval-rst}
.. autofunction:: musyn.utils.check_signal_1d
```

```{eval-rst}
.. autofunction:: musyn.utils.check_emg_matrix
```

```{eval-rst}
.. autofunction:: musyn.utils.check_n_synergies
```

## Numba backend support

```{eval-rst}
.. autofunction:: musyn.utils.numba_available
```

```{eval-rst}
.. autofunction:: musyn.utils.get_adaptive_loop
```
