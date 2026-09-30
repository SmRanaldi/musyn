# `musyn.preprocessing`

General-purpose sEMG conditioning utilities, independent of the three
papers behind the main algorithms. All functions use `(n_samples,
n_channels)` layout (rows = time) — this is the *transpose* of the
`(n_muscles, n_samples)` convention used by
{func}`musyn.extract_envelope`, {func}`musyn.extract_synergies`, and
{func}`musyn.select_synergy_number`.

## Filtering

```{eval-rst}
.. autofunction:: musyn.preprocessing.condition_emg
```

```{eval-rst}
.. autofunction:: musyn.preprocessing.linear_envelope
```

## ECG artifact removal

```{eval-rst}
.. autofunction:: musyn.preprocessing.remove_ecg_scica
```

## Segmentation and normalization

```{eval-rst}
.. autofunction:: musyn.preprocessing.segment_envelope
```

```{eval-rst}
.. autofunction:: musyn.preprocessing.time_normalize_envelope
```

```{eval-rst}
.. autofunction:: musyn.preprocessing.normalize_envelope
```
