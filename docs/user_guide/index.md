# User guide

How each algorithm works, what its parameters mean, and when to reach for
it. Each page maps directly onto the paper section or MATLAB file it
replaces — see {doc}`../references` for the full citations and
{doc}`../api/index` for the exhaustive parameter/return reference.

```{toctree}
:maxdepth: 1

envelope
synergies
selection
preprocessing
```

## Which function do I need?

```text
raw sEMG (n_samples, n_channels)
  │
  ▼
preprocessing.condition_emg   — bandpass + mains-hum notch
  │
  ├─ paper-accurate, adaptive ──▶ extract_envelope           ─┐
  │                                (n_muscles, n_samples)     │
  └─ simple, fast              ──▶ preprocessing.linear_envelope
                                                               │
                                                               ▼
                                        select_synergy_number  — pick k
                                                               │
                                                               ▼
                                          extract_synergies    — W, C
                                                               │
                                                               ▼
                                       vaf / r_squared         — quality check
```

If you already know how many synergies you're looking for (e.g. replicating
a prior study), skip straight to {func}`musyn.extract_synergies` with an
explicit `n_synergies`.
