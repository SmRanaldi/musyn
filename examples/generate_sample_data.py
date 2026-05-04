"""
Generate sample_emg.npz for use in examples.

Run once to create examples/data/sample_emg.npz:
    python examples/generate_sample_data.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def hann_activation(n_samples, width, offset):
    h = np.hanning(width)
    c = np.zeros(n_samples)
    start = offset % n_samples
    end = min(start + width, n_samples)
    c[start:end] = h[: end - start]
    return c


def main():
    rng = np.random.default_rng(2024)
    M, k_true, N = 8, 3, 4000
    fs = 1000.0

    # Ground-truth synergies
    W_true = rng.uniform(0.0, 1.0, size=(M, k_true))
    W_true /= W_true.sum(axis=0)

    # Hann-windowed activations
    C_true = np.zeros((k_true, N))
    widths = [200, 180, 160]
    offsets = [200, N // 3 + 100, 2 * N // 3 + 50]
    for i in range(k_true):
        C_true[i] = hann_activation(N, widths[i], offsets[i])

    # Simulate amplitude-modulated Gaussian noise (each channel)
    raw_emg = np.zeros((M, N))
    for m in range(M):
        envelope_m = (W_true @ C_true)[m]  # true envelope for channel m
        noise = rng.standard_normal(N)
        raw_emg[m] = envelope_m * noise

    # Also store a clean envelope matrix
    envelope_true = np.maximum(W_true @ C_true, 0.0)

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
    print(f"Saved sample data to {out_path}")
    print(f"  raw_emg: {raw_emg.shape}, fs={fs} Hz")
    print(f"  W_true: {W_true.shape}, C_true: {C_true.shape}, k_true={k_true}")


if __name__ == "__main__":
    main()
