# Installation

## From PyPI

```bash
pip install musyn                  # core: envelope + NMF + AIC (min/der/firstpeak/lastpeak need `[wavelet]`)
pip install musyn[wavelet]         # + PyWavelets, for AIC-curve-based selection methods
pip install musyn[jit]             # + Numba JIT backend for the envelope hot loop
pip install musyn[ecg]             # + scikit-learn, for ECG artifact removal
pip install musyn[plot]            # + matplotlib, for the example scripts
pip install musyn[all]             # everything above, plus dev tools
```

Only `numpy`, `scipy`, and `joblib` are hard dependencies. Everything else
is an optional extra, so a minimal install stays lightweight.

### Do I need `[wavelet]`?

`select_synergy_number`'s `method` argument picks one of two families of
criteria:

- `'min'`, `'der'`, `'firstpeak'`, `'lastpeak'` build a wavelet-based AIC
  curve and **require** `musyn[wavelet]` (PyWavelets).
- `'vaf'`, `'r2'`, `'plateau'`, `'surrogate'` only need the NMF solutions
  and work with the core install.

If you only use the default `method='min'`, install `musyn[wavelet]`.

## Building from source

A C compiler (`gcc`, `clang`, or MSVC) lets `pip` compile the Cython
extension that backs the adaptive envelope loop
(`src/musyn/envelope/_loop.pyx`), which is 100–500× faster than the pure
NumPy fallback. If no compiler is found, installation still succeeds and
falls back automatically — first to Numba (if `musyn[jit]` is installed),
then to NumPy. See {doc}`user_guide/envelope` for details on the backend
selection, and check `info['backend']` in `extract_envelope`'s return value
to confirm which one is active.

```bash
git clone https://github.com/SmRanaldi/musyn.git
cd musyn
pip install -e ".[all]"
```

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24, scipy ≥ 1.10, joblib ≥ 1.3 (core)
- PyWavelets ≥ 1.4 (`[wavelet]`), numba ≥ 0.57 (`[jit]`),
  scikit-learn ≥ 1.3 (`[ecg]`), matplotlib ≥ 3.7 (`[plot]`)

## Verifying the install

```python
import musyn
print(musyn.__version__)

import numpy as np
rng = np.random.default_rng(0)
env = musyn.extract_envelope(rng.normal(size=4000), fs=1000.0)
print(env.shape)  # (4000,)
```
