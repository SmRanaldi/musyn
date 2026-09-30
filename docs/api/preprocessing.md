# `musyn.preprocessing`

General-purpose sEMG conditioning utilities, independent of the three
papers behind the main algorithms. All functions use `(n_samples,
n_channels)` layout (rows = time) — this is the *transpose* of the
`(n_muscles, n_samples)` convention used by
{func}`musyn.extract_envelope`, {func}`musyn.extract_synergies`, and
{func}`musyn.select_synergy_number`.

## Filtering

```{autofunction} musyn.preprocessing.condition_emg
```

```{autofunction} musyn.preprocessing.linear_envelope
```

## ECG artifact removal

```{autofunction} musyn.preprocessing.remove_ecg_scica
```

## Segmentation and normalization

```{autofunction} musyn.preprocessing.segment_envelope
```

```{autofunction} musyn.preprocessing.time_normalize_envelope
```

```{autofunction} musyn.preprocessing.normalize_envelope
```
