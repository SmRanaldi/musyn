"""
Example 02 — Muscle Synergy Extraction

Extracts 3 muscle synergies from the envelope matrix using
NMF with SPARSE initialization (Soomro et al. 2018).

Usage:
    python examples/02_synergy_extraction.py
"""
from pathlib import Path
import numpy as np
import musyn
from musyn.metrics.quality import quality_ratio, vaf

DATA = Path(__file__).parent / "data" / "sample_emg.npz"


def main():
    data = np.load(str(DATA))
    envelope_true = data["envelope_true"]   # shape (8, 4000)
    W_true = data["W_true"]
    k_true = int(data["k_true"])
    fs = float(data["fs"])

    print(f"Envelope shape: {envelope_true.shape}")
    print(f"True synergies: k={k_true}")

    W, C, info = musyn.extract_synergies(
        envelope_true,
        n_synergies=k_true,
        init="sparse",
        n_runs=10,
        seed=42,
    )

    print(f"\nExtracted W: {W.shape}, C: {C.shape}")
    print(f"NMF converged: {info['converged']}, iterations: {info['n_iter']}")
    print(f"Best run: {info['best_run']}/{info['n_runs']}")

    qr = quality_ratio(W, W_true)
    vaf_val = vaf(envelope_true, W, C)
    print(f"\nQuality Ratio (QR):  {qr:.4f}  (1.0 = perfect)")
    print(f"VAF:                 {vaf_val:.4f}  (1.0 = perfect)")

    try:
        import matplotlib.pyplot as plt
        t = np.arange(C.shape[1]) / fs
        fig, axes = plt.subplots(k_true, 2, figsize=(12, 8))
        for i in range(k_true):
            axes[i, 0].bar(range(W.shape[0]), W[:, i])
            axes[i, 0].set_title(f"Synergy {i+1} weights")
            axes[i, 0].set_xlabel("Muscle")
            axes[i, 1].plot(t, C[i])
            axes[i, 1].set_title(f"Synergy {i+1} activation")
            axes[i, 1].set_xlabel("Time (s)")
        plt.suptitle(f"Muscle Synergies  QR={qr:.3f}  VAF={vaf_val:.3f}")
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "02_synergies.png", dpi=150)
        print("\nPlot saved to examples/02_synergies.png")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
