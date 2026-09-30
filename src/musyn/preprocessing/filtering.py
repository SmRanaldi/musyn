"""
Standard sEMG filtering: bandpass, notch, and envelope extraction.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sgn


def condition_emg(
    data: np.ndarray,
    fs: float,
    ecg_channels: list[int] | None = None,
    l_qrs: float = 0.004,
    apply_bandpass: bool = True,
    bandpass_range: tuple[float, float] = (20.0, 450.0),
    apply_notch: bool = True,
    notch_fundamental_hz: float = 50.0,
    notch_harmonics: int = 8,
) -> np.ndarray:
    """
    Standard sEMG conditioning pipeline.

    Applies in order (each step optional, see parameters below):
      1. 3rd-order Butterworth bandpass filter (default [20-450 Hz]).
      2. 3rd-order Butterworth bandstop notch filters at the mains-hum
         fundamental and its harmonics (default 50, 100, ..., 400 Hz, i.e.
         8 harmonics of 50 Hz), each +/-0.5 Hz wide.
      3. Optionally, single-channel ICA ECG removal on specified channels.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples x n_channels) or (n_samples,)
        Raw sEMG signal. Rows are time samples, columns are channels.
    fs : float
        Sampling frequency in Hz.
    ecg_channels : list of int, optional
        Column indices to apply SC-ICA ECG removal. Pass ``None`` (default)
        to skip ECG removal entirely. Requires ``scikit-learn``
        (``pip install musyn[ecg]``).
    l_qrs : float
        QRS complex duration in seconds used to set the ICA embedding
        window. Default 0.004 s (4 ms). Rule of thumb: 4-10 ms.
    apply_bandpass : bool
        Whether to apply the bandpass filter at all. Default True (matches
        the pipeline's original always-on behaviour).
    bandpass_range : tuple of float
        ``(low_hz, high_hz)`` bandpass cutoffs. Default ``(20.0, 450.0)``,
        the pipeline's original fixed range. Ignored if ``apply_bandpass``
        is False.
    apply_notch : bool
        Whether to apply the mains-hum notch filters at all. Default True
        (matches the pipeline's original always-on behaviour).
    notch_fundamental_hz : float
        Mains-hum fundamental frequency in Hz. Default 50.0 (EU mains);
        pass 60.0 for a 60 Hz mains grid. Ignored if ``apply_notch`` is
        False.
    notch_harmonics : int
        Number of harmonics of ``notch_fundamental_hz`` to notch out
        (1st..Nth). Default 8, matching the pipeline's original fixed
        50/100/.../400 Hz set. Ignored if ``apply_notch`` is False.

    Returns
    -------
    data_out : np.ndarray, same shape as data

    Examples
    --------
    >>> import numpy as np
    >>> from musyn.preprocessing import condition_emg
    >>> raw = np.random.randn(10000, 8)
    >>> clean = condition_emg(raw, fs=2000.0)
    >>> clean_no_notch = condition_emg(raw, fs=2000.0, apply_notch=False)
    """
    nyq = 0.5 * fs
    # a fresh array even if both filters are disabled below -- ecg_channels removal
    # mutates data_out in place, and must never alias (and thus mutate) the caller's
    # own input array.
    data_out = np.array(data, dtype=np.float64, copy=True)

    if apply_bandpass:
        low_hz, high_hz = bandpass_range
        b, a = sgn.butter(3, [low_hz / nyq, high_hz / nyq], btype="bandpass")
        data_out = sgn.filtfilt(b, a, data_out, axis=0)

    if apply_notch:
        for harmonic in range(1, notch_harmonics + 1):
            center = harmonic * notch_fundamental_hz
            stop = np.array([center - 0.5, center + 0.5]) / nyq
            b, a = sgn.butter(3, stop, btype="bandstop")
            data_out = sgn.filtfilt(b, a, data_out, axis=0)

    if ecg_channels is not None:
        from musyn.preprocessing.ecg import remove_ecg_scica
        embedding_dim = max(1, int(l_qrs * fs))
        for ch in ecg_channels:
            data_out[:, ch] = remove_ecg_scica(
                data_out[:, ch], fs=fs, embedding_dim=embedding_dim
            )

    return data_out


def linear_envelope(
    data: np.ndarray,
    fs: float,
    fc: float = 2.0,
) -> np.ndarray:
    """
    Simple linear sEMG envelope: full-wave rectification + low-pass filter.

    This is the standard linear method. For adaptive envelope extraction
    (Ranaldi et al. 2018) use ``musyn.extract_envelope`` instead.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples x n_channels) or (n_samples,)
        Conditioned (filtered) sEMG signal.
    fs : float
        Sampling frequency in Hz.
    fc : float
        Low-pass cutoff frequency in Hz. Default 2 Hz.

    Returns
    -------
    envelope : np.ndarray, same shape as data

    Examples
    --------
    >>> import numpy as np
    >>> from musyn.preprocessing import linear_envelope
    >>> emg = np.random.randn(5000, 4)
    >>> env = linear_envelope(emg, fs=1000.0)
    """
    b, a = sgn.butter(3, fc / (0.5 * fs), btype="low")
    return sgn.filtfilt(b, a, np.abs(data), axis=0)
