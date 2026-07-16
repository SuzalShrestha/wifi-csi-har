import numpy as np
import pytest

from csihar.preprocessing import align_receivers, make_windows, resample_uniform


def jittered_stream(duration_s=10.0, fs=100.0, n_sc=4, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    ts = np.arange(0, duration_s, 1 / fs) + rng.normal(0, 0.001, int(duration_s * fs))
    values = rng.normal(20, 1, (len(ts), n_sc))
    if drop is not None:
        keep = np.ones(len(ts), bool)
        keep[drop] = False
        ts, values = ts[keep], values[keep]
    return ts, values


def test_resample_produces_uniform_grid():
    ts, values = jittered_stream()
    out, loss = resample_uniform(ts, values, fs=100.0, t_start=1.0, t_end=4.0)
    assert out.shape == (300, 4)
    assert loss < 0.05


def test_resample_epoch_scale_timestamps_yield_fixed_length():
    # Host timestamps are epoch seconds (~1.7e9); the grid length must be
    # exactly round(window_s * fs) for every start, or np.stack breaks later.
    fs, t0 = 100.0, 1.7e9
    ts = t0 + np.arange(0, 6, 1 / fs)
    values = np.ones((len(ts), 2))
    for start_offset in np.linspace(0.0, 2.0, 21):
        start = t0 + start_offset
        out, _ = resample_uniform(ts, values, fs=fs, t_start=start, t_end=start + 3.0)
        assert out.shape == (300, 2)


def test_resample_reports_loss_for_gap():
    ts, values = jittered_stream(drop=slice(200, 300))  # 1 s hole
    _, loss = resample_uniform(ts, values, fs=100.0, t_start=1.0, t_end=4.0)
    assert loss > 0.25


def test_make_windows_counts_and_rejection():
    ts, values = jittered_stream(duration_s=10.0)
    wins = make_windows(ts, values, fs=100.0, window_s=3.0, hop_s=1.5)
    assert 4 <= len(wins) <= 5
    assert all(w.values.shape == (300, 4) for w in wins)

    ts2, values2 = jittered_stream(duration_s=10.0, drop=slice(100, 400))
    wins2 = make_windows(ts2, values2, fs=100.0, window_s=3.0, hop_s=1.5,
                         max_loss=0.10)
    assert len(wins2) < len(wins)  # gap windows rejected


def test_align_receivers_matches_and_drops():
    ts, values = jittered_stream(seed=1)
    streams = {
        "rx1": make_windows(ts, values),
        "rx2": make_windows(ts + 0.02, values),   # small clock offset: fine
        "rx3": make_windows(ts[:400], values[:400]),  # short stream
    }
    aligned = align_receivers(streams, tolerance_s=0.1)
    assert all(set(g) == {"rx1", "rx2", "rx3"} for g in aligned)
    # rx3 only covers ~4s -> fewer aligned groups than rx1 windows
    assert 0 < len(aligned) < len(streams["rx1"])


def test_align_empty():
    assert align_receivers({}) == []


def test_resample_validates_input():
    with pytest.raises(ValueError):
        resample_uniform(np.array([1.0]), np.ones((1, 2)), 100, 0, 1)
