# musyn — Muscle Synergy Analysis

Python package for muscle synergy analysis from surface EMG (sEMG).
Implements three peer-reviewed algorithms with high-performance backends.

## Algorithms

| Algorithm | Paper | Performance |
|-----------|-------|-------------|
| Adaptive envelope extraction | Ranaldi et al. 2018 | Cython C / Numba JIT / NumPy |
| NMF synergy extraction | Soomro et al. 2018 | BLAS-backed NumPy |
| AIC synergy number selection | Ranaldi et al. 2021 | Parallel k-sweep |

## Installation

```bash
pip install musyn                      # core (envelope + NMF)
pip install musyn[wavelet]             # + AIC selection (requires PyWavelets)
pip install musyn[jit]                 # + Numba JIT backend
pip install musyn[all]                 # everything + dev tools
```

## Quick Start

```python
import musyn
import numpy as np

# 1. Extract adaptive envelope from raw sEMG
envelope = musyn.extract_envelope(raw_emg, fs=1000.0)   # (M, N) → (M, N)

# 2. Automatically select number of synergies
k = musyn.select_synergy_number(envelope)

# 3. Extract synergies
W, C, info = musyn.extract_synergies(envelope, n_synergies=k)

# 4. Evaluate quality
print(f"VAF = {musyn.vaf(envelope, W, C):.3f}")
```

## Examples

```bash
python examples/generate_sample_data.py   # create sample data (run once)
python examples/01_basic_envelope.py
python examples/02_synergy_extraction.py
python examples/03_synergy_number_selection.py
python examples/04_full_pipeline.py
```

## References

- Ranaldi S, De Marchis C, Conforto S (2018) *An automatic, adaptive,
  information-based algorithm for the extraction of the sEMG envelope.*
  J Electromyogr Kinesiol.
- Soomro MH et al. (2018) *Comparison of Initialization Techniques for
  Accurate Extraction of Muscle Synergies from Myoelectric Signals via NNMF.*
  Applied Bionics and Biomechanics.
- Ranaldi S et al. (2021) *An Objective, Information-Based Approach for
  Selecting the Number of Muscle Synergies via NNMF.*
  IEEE Trans Neural Syst Rehabil Eng.

## License

MIT © Simone Ranaldi
