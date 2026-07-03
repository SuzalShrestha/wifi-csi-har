import numpy as np
import pytest

from csihar.preprocessing import apply_scaler, fit_scaler


def test_fit_apply_standardizes_train():
    rng = np.random.default_rng(0)
    train = rng.normal(50, 7, (20, 300, 52))
    params = fit_scaler(train)
    out = apply_scaler(train, params)
    flat = out.reshape(-1, 52)
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-4)


def test_train_stats_applied_to_test_unchanged():
    rng = np.random.default_rng(1)
    train = rng.normal(50, 7, (20, 300, 52))
    test = rng.normal(60, 7, (5, 300, 52))  # shifted distribution
    params = fit_scaler(train)
    out = apply_scaler(test, params)
    # test mean must NOT be zero — proves no refit on test data
    assert out.reshape(-1, 52).mean() > 0.5


def test_dead_subcarrier_guard():
    train = np.random.default_rng(2).normal(50, 7, (10, 100, 8))
    train[..., 3] = 0.0  # dead carrier, zero variance
    params = fit_scaler(train)
    out = apply_scaler(train, params)
    assert np.isfinite(out).all()


def test_fit_scaler_validates_shape():
    with pytest.raises(ValueError):
        fit_scaler(np.ones((10, 52)))


def test_apply_does_not_mutate():
    x = np.ones((2, 10, 4))
    x_copy = x.copy()
    apply_scaler(x, fit_scaler(x))
    assert np.array_equal(x, x_copy)


def test_subcarrier_indices_are_52():
    from csihar.preprocessing import usable_lltf_indices

    idx = usable_lltf_indices()
    assert len(idx) == 52
    assert 0 not in idx            # DC excluded
    assert not set(range(27, 38)) & set(idx)  # guard band excluded


def test_null_detector_finds_simulated_nulls():
    from csihar.parser import parse_line
    from csihar.preprocessing import detect_null_subcarriers
    from csihar.simulate import generate_lines

    frames = [
        parse_line(line, ts)
        for ts, line in generate_lines("background", duration_s=3.0, seed=3)
    ]
    amps = np.stack([f.amplitude for f in frames if f is not None])
    nulls = set(detect_null_subcarriers(amps).tolist())
    assert 0 in nulls
    assert set(range(28, 37)).issubset(nulls)
