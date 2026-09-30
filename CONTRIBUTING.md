# Contributing to musyn

Thanks for your interest in improving musyn. This project implements three
published sEMG analysis algorithms (adaptive envelope extraction, NMF synergy
decomposition, AIC-based synergy number selection); the priority for any
change is that the algorithms keep matching their reference papers and the
prior MATLAB/C implementations they replace.

## Development setup

```bash
git clone https://github.com/SmRanaldi/musyn.git
cd musyn
pip install -e ".[all]"          # editable install, builds the Cython extension if a C compiler is available
python examples/generate_sample_data.py   # one-time: creates examples/data/sample_emg.npz
pytest tests/
```

A C compiler (`gcc`, `clang`, or MSVC) is only needed to build the optional
Cython backend for the envelope hot loop. If it's missing, installation
still succeeds and falls back to Numba (if installed) or pure NumPy — see
`src/musyn/envelope/adaptive.py`.

## Project layout

See `README.md` → "Package structure" for the module map, and the
docstrings in each `api.py` for the public entry points. The short version:

- `envelope/` — Algorithm 1 (Ranaldi et al. 2018)
- `decomposition/` — Algorithm 2 (Soomro et al. 2018)
- `selection/` — Algorithm 3 (Ranaldi et al. 2021)
- `preprocessing/` — general sEMG conditioning, independent of the three
  papers above
- `metrics/`, `io/`, `utils/` — shared support code

## Running tests

```bash
pytest tests/                          # full suite
pytest tests/ -m "not slow"            # skip slow tests
pytest tests/test_envelope/            # one module
pytest tests/ -k "nsvd"                # by keyword
```

**Never use real subject EMG recordings in tests.** All fixtures are
synthetic, generated in `tests/conftest.py` (see the `rng`,
`synthetic_emg_1d`, and `synthetic_emg_matrix` fixtures) or built inline with
`numpy.random.default_rng`. This keeps the test suite runnable by anyone
without access to (or IRB approval for) real recordings, and keeps the repo
free of clinical data. `examples/generate_sample_data.py` follows the same
rule for example data.

If you're validating a control-flow change (e.g. "does this skip an
expensive computation under condition X") rather than a change to the
underlying math, prefer a mock-based test that asserts on call counts over a
numerical test on synthetic data — see the AIC `compute_dof` tests for the
pattern.

## Code style

- Formatting/linting: [ruff](https://docs.astral.sh/ruff/), configured in
  `pyproject.toml`. Run `ruff check src/ tests/` before committing.
- Docstrings: NumPy style (`Parameters` / `Returns` / `Notes` / `References`
  / `Examples` sections), matching the existing modules. Every public
  function should have a `References` section pointing at the paper/MATLAB
  file it implements, if any, and a runnable `Examples` block — the docs
  site (`docs/`) is built directly from these docstrings via Sphinx autodoc.
- Type hints on public function signatures; `from __future__ import
  annotations` is used throughout for forward-compatible syntax on Python
  3.10.
- No comments explaining *what* code does; only *why*, when it's genuinely
  non-obvious (a paper-specific constant, a numerical edge case, a
  deliberate deviation from the reference implementation).

## Changing algorithm behavior

If a change alters the numerical output of `extract_envelope`,
`extract_synergies`, or `select_synergy_number` for existing inputs:

1. Say so explicitly in the PR description — these are published,
   peer-reviewed algorithms, and silent behavior drift is the worst possible
   outcome for a package like this.
2. Add or update a test in the matching `tests/test_*/` directory using
   synthetic data.
3. Add an entry under `## [Unreleased]` in `CHANGELOG.md`.
4. If the change affects a public function's parameters or return value,
   update both its docstring and the corresponding section of `README.md`.

Non-behavioral changes (performance, refactoring, new preprocessing
utilities, docs) don't need to touch the algorithm's numerical guarantees,
but should still get a `CHANGELOG.md` entry and, for anything user-facing,
a docs update.

## Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html   # macOS; xdg-open on Linux
```

The docs are also built automatically by
[Read the Docs](https://readthedocs.org) from `.readthedocs.yaml` on every
push to `main` and on tagged releases.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your change, following the guidelines above.
3. Ensure `pytest tests/` and `ruff check src/ tests/` both pass.
4. Open a pull request describing *what* changed and *why* — link the paper
   section or MATLAB file if the change touches algorithm internals.

## Reporting bugs / requesting features

Please use the GitHub issue templates
([bug report](https://github.com/SmRanaldi/musyn/issues/new?template=bug_report.yml) /
[feature request](https://github.com/SmRanaldi/musyn/issues/new?template=feature_request.yml)).
For numerical/algorithmic bugs, a minimal synthetic reproduction (no real
EMG data) is the single most useful thing you can include.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
