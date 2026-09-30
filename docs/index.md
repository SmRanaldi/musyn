# musyn

**Muscle synergy analysis from surface EMG.**

musyn implements three peer-reviewed algorithms for sEMG amplitude and
synergy analysis, each backed by a published paper and a prior MATLAB/C
reference implementation:

| # | Algorithm | Paper |
|---|-----------|-------|
| 1 | Adaptive sEMG envelope extraction | Ranaldi et al. (2018), *J Electromyogr Kinesiol* |
| 2 | NMF muscle synergy extraction | Soomro et al. (2018), *Applied Bionics and Biomechanics* |
| 3 | AIC-based synergy number selection | Ranaldi et al. (2021), *IEEE TNSRE* |

It also ships standard sEMG preprocessing utilities (bandpass/notch
filtering, ECG artifact removal, trial segmentation) that don't depend on
any of the three papers above.

```python
import musyn
from musyn.preprocessing import condition_emg

emg_clean = condition_emg(raw_emg, fs=2000.0)
envelope = musyn.extract_envelope(emg_clean.T, fs=2000.0)
k = musyn.select_synergy_number(envelope)
W, C, info = musyn.extract_synergies(envelope, n_synergies=k)
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting started
{doc}`installation` and {doc}`quickstart` — install musyn and run your
first envelope/synergy pipeline in a few lines.
:::

:::{grid-item-card} User guide
{doc}`user_guide/index` — how each algorithm works, its parameters, and
when to reach for it.
:::

:::{grid-item-card} API reference
{doc}`api/index` — every public function, generated from its docstring.
:::

:::{grid-item-card} Examples
{doc}`examples` — the runnable scripts in `examples/`, with their output.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
user_guide/index
examples
api/index
references
changelog
contributing
CODE_OF_CONDUCT
```
