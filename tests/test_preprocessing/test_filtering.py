"""condition_emg's bandpass/notch stages were hardcoded-on with a fixed range/harmonic
set until this was made configurable (downstream need: risveglio-suite's EMG-filtering
settings tab). No tests previously existed for this module at all.
"""

from __future__ import annotations

import numpy as np

from musyn.preprocessing.filtering import condition_emg


def _tone(freq_hz: float, fs: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def _std_steady_state(x: np.ndarray, trim: int = 1000) -> float:
    """Std over the middle of the signal only -- a narrow (e.g. +/-0.5 Hz) notch's
    filtfilt edge transients otherwise dominate a short test signal's amplitude even
    when the tone itself is being attenuated correctly in steady state."""
    return float(np.std(x[trim:-trim]))


def test_default_behaviour_matches_original_fixed_pipeline():
    """Defaults must reproduce the pipeline's original always-on 20-450 Hz bandpass +
    50 Hz x8-harmonic notch exactly -- other projects depend on this function's
    existing default behaviour, so adding parameters must not change it."""
    rng = np.random.default_rng(0)
    fs = 2000.0
    n = 12000
    data = rng.standard_normal((n, 2))

    out = condition_emg(data, fs=fs)

    assert out.shape == data.shape
    # a 50 Hz tone (inside the notch band) must be strongly attenuated by default
    tone_50 = _tone(50.0, fs, n)[:, None].repeat(2, axis=1)
    out_tone = condition_emg(tone_50, fs=fs)
    assert _std_steady_state(out_tone) < 0.1 * _std_steady_state(tone_50)


def test_apply_bandpass_false_skips_bandpass():
    fs = 2000.0
    n = 12000
    # a 10 Hz tone is outside the default 20-450 Hz passband -- removed when the
    # bandpass runs, preserved when it's disabled.
    tone_10 = _tone(10.0, fs, n)[:, None]

    filtered = condition_emg(tone_10, fs=fs, apply_notch=False)
    unfiltered = condition_emg(tone_10, fs=fs, apply_bandpass=False, apply_notch=False)

    assert _std_steady_state(filtered) < 0.2 * _std_steady_state(tone_10)
    assert _std_steady_state(unfiltered) > 0.8 * _std_steady_state(tone_10)


def test_apply_notch_false_skips_notch():
    fs = 2000.0
    n = 12000
    tone_50 = _tone(50.0, fs, n)[:, None]

    notched = condition_emg(tone_50, fs=fs, apply_bandpass=False)
    not_notched = condition_emg(tone_50, fs=fs, apply_bandpass=False, apply_notch=False)

    assert _std_steady_state(notched) < 0.1 * _std_steady_state(tone_50)
    assert _std_steady_state(not_notched) > 0.8 * _std_steady_state(tone_50)


def test_custom_bandpass_range():
    fs = 2000.0
    n = 12000
    tone_100 = _tone(100.0, fs, n)[:, None]

    # 100 Hz is inside the default range but outside a narrower custom one
    default_range = condition_emg(tone_100, fs=fs, apply_notch=False)
    narrow_range = condition_emg(
        tone_100, fs=fs, apply_notch=False, bandpass_range=(150.0, 450.0)
    )

    assert _std_steady_state(default_range) > 0.5 * _std_steady_state(tone_100)
    assert _std_steady_state(narrow_range) < 0.1 * _std_steady_state(tone_100)


def test_custom_notch_fundamental_60hz():
    fs = 2000.0
    n = 12000
    tone_60 = _tone(60.0, fs, n)[:, None]

    default_50hz_notch = condition_emg(tone_60, fs=fs, apply_bandpass=False)
    notch_60hz = condition_emg(
        tone_60, fs=fs, apply_bandpass=False, notch_fundamental_hz=60.0
    )

    # 60 Hz survives the default 50 Hz-harmonic notch set (60 isn't a harmonic of 50)
    assert _std_steady_state(default_50hz_notch) > 0.5 * _std_steady_state(tone_60)
    assert _std_steady_state(notch_60hz) < 0.1 * _std_steady_state(tone_60)


def test_returns_fresh_array_when_filters_disabled():
    """Regression: with both filters off, ``data_out`` used to alias the caller's own
    array (no filtfilt call ever ran to produce a fresh one) -- a later in-place
    ECG-channel removal would then silently mutate the caller's input."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2000, 3))
    original = data.copy()

    out = condition_emg(data, fs=1000.0, apply_bandpass=False, apply_notch=False)

    assert out is not data
    np.testing.assert_array_equal(data, original)
