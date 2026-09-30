"""
Generate sample_emg.npz for use in examples.

Simulates a 15-second cyclical motor task with three synergies:
  - Synergy 0 (agonist):   four bursts, slight fatigue progression
  - Synergy 1 (antagonist): four bursts, alternating with synergy 0
  - Synergy 2 (stabilizer): active during transitions, overlapping with both

Run once to create examples/data/sample_emg.npz:
    python examples/generate_sample_data.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def hann_burst(n_samples: int, t_start: float, t_end: float,
               peak: float, fs: float) -> np.ndarray:
    """Hann-windowed burst between t_start and t_end (seconds)."""
    c = np.zeros(n_samples)
    i0 = int(t_start * fs)
    i1 = int(t_end * fs)
    width = i1 - i0
    if width > 0:
        c[i0:i1] = np.hanning(width) * peak
    c[c<1e-4]=1e-4
    return c


def main():
    rng = np.random.default_rng(2024)
    M, k_true = 8, 3
    fs = 1000.0
    N = int(15.0 * fs)   # 15 seconds

    # ── Ground-truth synergy weights ────────────────────────────────────────
    W_true = rng.uniform(0.1, 1.0, size=(M, k_true))
    W_true /= W_true.sum(axis=0)

    # ── Activation time courses ─────────────────────────────────────────────
    C_true = np.zeros((k_true, N))

    # Synergy 0 — agonist: four bursts, peak amplitude decreases slightly
    # (simulates mild fatigue) with a ramp-up on the first burst
    for t0, t1, peak in [
        (0.5,  2.0,  0.80),
        (4.0,  5.5,  1.00),
        (8.0,  9.5,  0.90),
        (12.0, 13.5, 0.75),
    ]:
        C_true[0] += hann_burst(N, t0, t1, peak, fs)

    # Synergy 1 — antagonist: four bursts interleaved with synergy 0
    for t0, t1, peak in [
        (2.5,  4.0,  0.85),
        (6.0,  7.5,  0.95),
        (10.0, 11.5, 0.85),
        (13.5, 14.5, 0.70),
    ]:
        C_true[1] += hann_burst(N, t0, t1, peak, fs)

    # Synergy 2 — stabilizer: active during transitions, overlaps both
    for t0, t1, peak in [
        (1.7,  3.0,  0.60),   # S0→S1 transition
        (5.2,  6.5,  0.55),   # S0→S1 transition
        (9.2,  10.5, 0.65),   # S0→S1 transition
        (11.0, 12.5, 0.50),   # S1→S0 transition
    ]:
        C_true[2] += hann_burst(N, t0, t1, peak, fs)

    C_true = np.clip(C_true, 0.001, 1.0)

    # ── Simulate amplitude-modulated Gaussian noise ──────────────────────────
    raw_emg = np.zeros((M, N))
    for m in range(M):
        envelope_m = (W_true @ C_true)[m]
        raw_emg[m] = envelope_m * rng.standard_normal(N)

    envelope_true = W_true @ C_true   # (M, N), non-negative by construction

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = HERE / "data" / "sample_emg.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        raw_emg=raw_emg,
        envelope_true=envelope_true,
        W_true=W_true,
        C_true=C_true,
        fs=np.float64(fs),
        k_true=np.int64(k_true),
    )
    print(f"Saved to {out_path}")
    print(f"  raw_emg:      {raw_emg.shape}  ({N/fs:.0f} s, {M} channels)")
    print(f"  k_true={k_true}  W_true: {W_true.shape}  C_true: {C_true.shape}")
    for i in range(k_true):
        active = C_true[i] > 0.01
        print(f"  Syn {i}: {active.sum()/fs:.1f} s active  "
              f"peak C = {C_true[i].max():.2f}")


if __name__ == "__main__":
    main()
