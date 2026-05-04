# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- I/O: load/save CSV, MAT, NPZ formats
- Metrics: QR, VAF, R², cosine similarity
- Example scripts (01–04)
- pytest test suite
