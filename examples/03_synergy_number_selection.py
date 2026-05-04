"""
Example 03 — Automatic Synergy Number Selection

Uses the modified AIC (Ranaldi et al. 2021) to objectively select
the optimal number of muscle synergies.

Usage:
    python examples/03_synergy_number_selection.py
"""
from pathlib import Path
import numpy as np
import musyn

DATA = Path(__file__).parent / "data" / "sample_emg.npz"


def main():
    data = np.load(str(DATA))
    envelope_true = data["envelope_true"]
    k_true = int(data["k_true"])

    print(f"Ground-truth k: {k_true}")
    print("Evaluating k = 1..6 ...\n")

    result = musyn.select_synergy_number(
        envelope_true,
        k_range=range(1, 7),
        method="min",
        n_runs=5,
        seed=0,
        return_full=True,
    )

    k_opt = result["k_opt"]
    aic_values = result["aic_values"]
    k_range = result["k_range"]

    print(f"AIC values:")
    for k, aic in zip(k_range, aic_values):
        marker = " <-- optimal" if k == k_opt else ""
        print(f"  k={k}: AIC={aic:.2f}{marker}")

    print(f"\nSelected k = {k_opt}  (true k = {k_true})")

    # Compare criteria
    print("\nComparing all selection criteria:")
    for method in ("min", "der", "firstpeak", "vaf", "plateau"):
        from musyn.selection.criteria import select_synergy_count
        k_m = select_synergy_count(
            aic_values, k_range, method,
            M_matrix=envelope_true,
            solutions=result["solutions"],
        )
        print(f"  {method:12s}: k = {k_m}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_range, aic_values, "o-", color="steelblue", lw=2)
        ax.axvline(k_opt, color="red", ls="--", label=f"Selected k={k_opt}")
        ax.axvline(k_true, color="green", ls=":", label=f"True k={k_true}")
        ax.set_xlabel("Number of synergies k")
        ax.set_ylabel("Modified AIC")
        ax.set_title("AIC-based synergy number selection")
        ax.legend()
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "03_aic_curve.png", dpi=150)
        print("\nPlot saved to examples/03_aic_curve.png")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
