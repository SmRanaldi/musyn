# References

## Papers implemented by musyn

- Ranaldi S, De Marchis C, Conforto S (2018). **An automatic, adaptive,
  information-based algorithm for the extraction of the sEMG envelope.**
  *Journal of Electromyography and Kinesiology*, 42, 1–9.
  → {doc}`user_guide/envelope`

- Soomro MH, Conforto S, Giunta G, Ranaldi S, De Marchis C (2018).
  **Comparison of Initialization Techniques for the Accurate Extraction of
  Muscle Synergies from Myoelectric Signals via Nonnegative Matrix
  Factorization.** *Applied Bionics and Biomechanics*, 2018.
  → {doc}`user_guide/synergies`

- Ranaldi S, De Marchis C, Severini G, Conforto S (2021). **An Objective,
  Information-Based Approach for Selecting the Number of Muscle Synergies
  to be Extracted via Non-Negative Matrix Factorization.** *IEEE
  Transactions on Neural Systems and Rehabilitation Engineering*, 29,
  2676–2683.
  → {doc}`user_guide/selection`

If musyn contributes to your research, please cite the relevant paper(s)
above alongside the software itself — see `CITATION.cff` in the repository
root (GitHub's "Cite this repository" button uses it directly).

## Prior implementations

musyn is a Python/Cython reimplementation of two MATLAB/C codebases by the
same author:

- [SmRanaldi/EMG_envelope](https://github.com/SmRanaldi/EMG_envelope) —
  adaptive envelope extraction (Algorithm 1)
- [SmRanaldi/NSyn_Criteria](https://github.com/SmRanaldi/NSyn_Criteria) —
  NMF synergy extraction and AIC-based selection (Algorithms 2 and 3)

See the README's "MATLAB equivalences" table for a file-by-file mapping.

## Other algorithms used internally

- Lee DD, Seung HS (1999). **Learning the parts of objects by non-negative
  matrix factorization.** *Nature*, 401, 788–791. — the multiplicative
  update rules in {func}`musyn.decomposition.updates.update_W` /
  {func}`musyn.decomposition.updates.update_C`.
- Boutsidis C, Gallopoulos E (2008). **SVD based initialization: A head
  start for nonnegative matrix factorization.** *Pattern Recognition*, 41(4),
  1350–1362. — {func}`musyn.decomposition.init_strategies.init_nsvd`.
