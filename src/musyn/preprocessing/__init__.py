"""
musyn.preprocessing — sEMG signal conditioning utilities.

Provides standard preprocessing steps used before envelope extraction and
synergy analysis: bandpass/notch filtering, ECG removal via single-channel
ICA, envelope computation, and trial segmentation/normalisation.

All functions follow the convention that raw EMG arrays have shape
(n_samples, n_channels) (rows = time, columns = channels). This differs from
the M_matrix convention used in the analysis functions (n_muscles, n_samples).

Quick start
-----------
>>> from musyn.preprocessing import condition_emg, linear_envelope
>>> emg_clean = condition_emg(raw_emg, fs=2000)
>>> envelope   = linear_envelope(emg_clean, fs=2000)
"""

from musyn.preprocessing.ecg import remove_ecg_scica
from musyn.preprocessing.filtering import condition_emg, linear_envelope
from musyn.preprocessing.segmentation import (
    normalize_envelope,
    segment_envelope,
    time_normalize_envelope,
)

__all__ = [
    "condition_emg",
    "linear_envelope",
    "remove_ecg_scica",
    "segment_envelope",
    "time_normalize_envelope",
    "normalize_envelope",
]
