"""
Example 04 — Full Pipeline

Demonstrates the complete musyn workflow on synthetic data:
  1. Load raw sEMG
  2. Extract adaptive envelope (Algorithm 1)
  3. Automatically select number of synergies (Algorithm 3)
  4. Extract synergies with optimal k (Algorithm 2)
  5. Evaluate quality

Usage:
    python examples/04_full_pipeline.py
"""
from pathlib import Path
import numpy as np
import musyn
from musyn.metrics.quality import quality_ratio, vaf

DATA = Path(__file__).parent / "data" / "sample_emg.npz"


def main():
    # ── 1. Load ────────────────────────────────────────────────────────────
    data = np.load(str(DATA))
    raw_emg = data["raw_emg"]       # (8, 4000)
    W_true = data["W_true"]
    k_true = int(data["k_true"])
    fs = float(data["fs"])
    print("=== musyn Full Pipeline ===\n")
    print(f"Input: {raw_emg.shape[0]} muscles × {raw_emg.shape[1]} samples, fs={fs} Hz")

    # ── 2. Envelope extraction ─────────────────────────────────────────────
    print("\n[1/3] Extracting adaptive envelope ...")
    envelope = musyn.extract_envelope(raw_emg, fs=fs, n_jobs=-1)
    print(f"  Envelope: {envelope.shape}, range [{envelope.min():.4f}, {envelope.max():.4f}]")

    # ── 3. Synergy number selection ────────────────────────────────────────
    print("\n[2/3] Selecting number of synergies (AIC, k=1..6) ...")
    k_opt = musyn.select_synergy_number(
        envelope, k_range=range(1, 7), method="min", n_runs=5, seed=1,
    )
    print(f"  Selected k = {k_opt}  (true k = {k_true})")

    # ── 4. Synergy extraction ──────────────────────────────────────────────
    print(f"\n[3/3] Extracting {k_opt} synergies ...")
    W, C, info = musyn.extract_synergies(envelope, n_synergies=k_opt, seed=42)
    print(f"  W: {W.shape}, C: {C.shape}")
    print(f"  Converged: {info['converged']}, iterations: {info['n_iter']}")

    # ── 5. Quality ─────────────────────────────────────────────────────────
    qr = quality_ratio(W, W_true)
    vaf_val = vaf(envelope, W, C)
    print(f"\nResults:")
    print(f"  Quality Ratio (QR): {qr:.4f}")
    print(f"  VAF:                {vaf_val:.4f}")
    print(f"  Synergy error:      Δk = {k_opt - k_true}")

    # ── 6. Save ────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "data" / "pipeline_results.npz"
    from musyn.io.writers import save_results
    save_results(
        out_path, W, C, envelope=envelope,
        metadata={"k": k_opt, "fs": fs, "qr": qr, "vaf": vaf_val},
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
