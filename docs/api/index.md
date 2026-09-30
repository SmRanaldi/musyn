# API reference

All pages below are generated directly from the package's NumPy-style
docstrings.

```{toctree}
:maxdepth: 1

public
preprocessing
metrics
io
utils
internals/index
```

## Public API at a glance

| Function | Purpose |
|----------|---------|
| {func}`musyn.extract_envelope` | Algorithm 1 — adaptive sEMG envelope |
| {func}`musyn.extract_synergies` | Algorithm 2 — NMF synergy extraction |
| {func}`musyn.select_synergy_number` | Algorithm 3 — AIC-based synergy count selection |
| {func}`musyn.quality_ratio` | Compare extracted vs. ground-truth synergies |
| {func}`musyn.vaf` | Variance Accounted For |
| {func}`musyn.r_squared` | Coefficient of determination |
| {mod}`musyn.preprocessing` | Filtering, ECG removal, segmentation |
| {mod}`musyn.io` | Load/save CSV, MAT, NPZ |
