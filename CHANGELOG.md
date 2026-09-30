# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-30

### Added
- Initial package structure (src-layout)
- Algorithm 1: adaptive sEMG envelope extraction (Ranaldi et al. 2018)
  - AR(12) Yule-Walker prewhitening
  - Nu-order detection
  - Adaptive iterative loop with entropy convergence
  - Cython C extension for hot loop (primary backend)
  - Numba JIT fallback
  - NumPy vectorized fallback
- Algorithm 2: NMF muscle synergy extraction (Soomro et al. 2018)
  - RAND, NSVD, SPARSE initialization strategies
  - Multiplicative update rules
  - Multi-run with joblib parallelism
- Algorithm 3: AIC-based synergy number selection (Ranaldi et al. 2021)
  - Signal-dependent noise estimation
  - Daubechies-5 wavelet DoF calculation
  - Modified log-likelihood with SDN correction
  - Eight selection criteria (min, der, firstpeak, lastpeak, VAF, R², plateau, surrogate)
- `preprocessing` subpackage for general sEMG conditioning, independent of
  the three papers above: `condition_emg` (bandpass + mains-hum notch),
  `linear_envelope`, `remove_ecg_scica` (single-channel SC-ICA ECG removal),
  `segment_envelope`, `time_normalize_envelope`, `normalize_envelope`
- I/O: load/save CSV, MAT, NPZ formats
- Metrics: QR, VAF, R², cosine similarity
- Example scripts (01–04) and synthetic sample-data generator
- pytest test suite
- Sphinx documentation site (`docs/`), published to Read the Docs
- `CONTRIBUTING.md`, `CITATION.cff`, `CODE_OF_CONDUCT.md`
- GitHub Actions CI (test matrix) and PyPI publish workflow
- PEP 561 `py.typed` marker

### Changed
- `condition_emg`'s bandpass (default 20–450 Hz) and mains-hum notch
  (default 50 Hz × 8 harmonics) stages are now individually configurable via
  `apply_bandpass` / `bandpass_range` / `apply_notch` /
  `notch_fundamental_hz` / `notch_harmonics`, instead of being hardcoded on.
  Defaults reproduce the original fixed behavior exactly.
- `PyWavelets` moved from a hard runtime dependency to the optional
  `wavelet` extra (`pip install musyn[wavelet]`), matching the fact that
  only the AIC-curve-based selection methods (`'min'`, `'der'`,
  `'firstpeak'`, `'lastpeak'`) touch `wavelet_dof.py` — `'vaf'`, `'r2'`,
  `'plateau'`, and `'surrogate'` never import it.

### Performance
- `select_synergy_number` now skips the wavelet-based DoF computation
  (`compute_total_dof`, one call per `k` in `k_range`) entirely when
  `method` is `'vaf'`, `'r2'`, `'plateau'`, or `'surrogate'`, since those
  criteria only read the NMF `solutions`, never `aic_values`. Previously
  this was computed unconditionally regardless of `method`.
  `return_full=True` now returns `aic_values=None` for these four methods.

### Fixed
- Corrected the author lists for the Soomro et al. (2018) and Ranaldi et
  al. (2021) paper citations (`CITATION.cff`, `README.md`, docs, and the
  `extract_synergies`/`select_synergy_number` docstrings), which had
  fabricated/incorrect co-authors and a wrong author order. Verified
  against CrossRef, PubMed, and the authors' institutional page.
