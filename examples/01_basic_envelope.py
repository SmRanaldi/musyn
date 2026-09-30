"""
Example 01 — Basic EMG Envelope Extraction

Loads synthetic EMG data and extracts the adaptive envelope using
the Ranaldi et al. (2018) algorithm.

Usage:
    python examples/01_basic_envelope.py
"""
from pathlib import Path
import numpy as np
import musyn

DATA = Path(__file__).parent / "data" / "sample_emg.npz"


def main():
    data = np.load(str(DATA))
    raw_emg = data["raw_emg"]       # shape (8, 4000)
    envelope_true = data["envelope_true"]
    fs = float(data["fs"])

    print(f"Signal shape: {raw_emg.shape}, fs={fs} Hz")
    print(f"Active backend: {musyn.envelope.adaptive.backend()}")

    # --- Single channel ---
    env_single, info = musyn.extract_envelope(
        raw_emg[0], fs=fs, return_info=True
    )
    M = info[0]["window_lengths"]
    print(f"\nChannel 0: {info[0]['iterations']} iterations, "
          f"converged={info[0]['converged']}, backend={info[0]['backend']}")
    print(f"  M_k: min={M.min():.0f}  max={M.max():.0f}  mean={M.mean():.0f} samples")

    # --- All channels (parallelised) ---
    env_all, infos = musyn.extract_envelope(raw_emg, fs=fs, return_info=True)
    print(f"\nAll channels envelope shape: {env_all.shape}")
    conv = [i["converged"] for i in infos]
    print(f"Converged: {sum(conv)}/{len(conv)} channels")

    # --- Optional plot ---
    try:
        import matplotlib.pyplot as plt
        t = np.arange(raw_emg.shape[1]) / fs
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, raw_emg[0], lw=0.5, alpha=0.6, label="raw EMG")
        axes[0].plot(t, envelope_true[0], "k--", lw=1.5, label="true envelope")
        axes[0].plot(t, env_all[0], "r-", lw=1.5, label="estimated envelope")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].set_title("Channel 0 — adaptive envelope")

        M_lengths = infos[0]["window_lengths"]  # optimal window per sample, channel 0
        axes[1].plot(t, M_lengths, lw=1.0, color="steelblue")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Window length (samples)")
        axes[1].set_title("Channel 0 — optimal window length $M_k$")

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "01_envelope.png", dpi=150)
        print("\nPlot saved to examples/01_envelope.png")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
