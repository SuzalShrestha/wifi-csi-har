import numpy as np

from csihar.features import FEATURE_NAMES_PER_CHANNEL, extract_features


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 3, 100, 52)).astype(np.float32)
    feats = extract_features(X)
    expected_f = 3 * 52 * len(FEATURE_NAMES_PER_CHANNEL)
    assert expected_f == 3 * 52 * 10
    assert feats.shape == (4, expected_f)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_does_not_mutate_input():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, 100, 52)).astype(np.float32)
    before = X.copy()
    extract_features(X)
    assert np.array_equal(X, before)


def test_discriminates_spectral_content():
    """A 5 Hz sinusoidal component should raise 2-10 Hz band energy."""
    fs = 100.0
    t = np.arange(300) / fs
    rng = np.random.default_rng(2)

    constant = np.broadcast_to(1.0, (2, 300, 4)).copy() + rng.normal(
        0, 0.01, size=(2, 300, 4)
    )
    sinusoid = constant + 5.0 * np.sin(2 * np.pi * 5.0 * t)[None, :, None]

    X = np.stack([constant, sinusoid]).astype(np.float32)  # (2, 2, 300, 4)
    feats = extract_features(X, fs=fs)

    n_stats = len(FEATURE_NAMES_PER_CHANNEL)
    band_2_10_offset = FEATURE_NAMES_PER_CHANNEL.index("band_energy_2_10hz")
    # channel-major layout: stats block per (rx, subcarrier); grab rx0, sc0.
    band_idx = 0 * n_stats + band_2_10_offset
    constant_energy = feats[0, band_idx]
    sinusoid_energy = feats[1, band_idx]
    assert sinusoid_energy > constant_energy * 10


def test_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 3, 100, 52)).astype(np.float32)
    a = extract_features(X)
    b = extract_features(X)
    assert np.array_equal(a, b)
