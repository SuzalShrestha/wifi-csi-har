import numpy as np
import pytest

from csihar.preprocessing import detrend_moving_mean, hampel, lowpass


def test_hampel_removes_isolated_spike():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 0.1, (200, 4))
    x[100, 2] = 100.0
    cleaned = hampel(x, window=11, n_sigmas=3.0)
    assert abs(cleaned[100, 2] - 10) < 1.0
    # bulk of the signal passes through: few replacements, all of them small
    changed = cleaned[:90] != x[:90]
    assert changed.mean() < 0.05
    assert np.abs(cleaned[:90] - x[:90]).max() < 1.0


def test_detrend_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        detrend_moving_mean(np.ones((50, 2)), window=100)


def test_hampel_does_not_mutate_input():
    x = np.ones((50, 2))
    x[25, 0] = 99.0
    x_copy = x.copy()
    hampel(x)
    assert np.array_equal(x, x_copy)


def test_hampel_rejects_even_window():
    with pytest.raises(ValueError):
        hampel(np.ones((10, 1)), window=4)


def test_detrend_removes_constant_offset():
    x = np.full((300, 3), 25.0)
    out = detrend_moving_mean(x, window=51)
    assert np.abs(out).max() < 1e-9


def test_detrend_keeps_fast_oscillation():
    t = np.arange(1000) / 100.0
    slow = 5 * t                       # drift
    fast = np.sin(2 * np.pi * 5 * t)   # 5 Hz activity content
    x = (slow + fast)[:, None]
    out = detrend_moving_mean(x, window=101)
    core = out[150:-150, 0]
    assert np.corrcoef(core, fast[150:-150])[0, 1] > 0.95


def test_lowpass_attenuates_high_frequency():
    fs = 100.0
    t = np.arange(1000) / fs
    low = np.sin(2 * np.pi * 2 * t)
    high = np.sin(2 * np.pi * 40 * t)
    x = (low + high)[:, None]
    out = lowpass(x, cutoff_hz=20, fs=fs)[:, 0]
    residual_high = out - low
    assert np.abs(residual_high[100:-100]).max() < 0.1


def test_lowpass_validates_cutoff():
    with pytest.raises(ValueError):
        lowpass(np.ones((100, 1)), cutoff_hz=60, fs=100)
