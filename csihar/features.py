"""Per-window hand-crafted features for the classical ML baseline.

Turns a preprocessed CSI window ``(n_rx, T, S)`` into a flat feature vector
by computing, independently per (receiver, subcarrier) time series, a small
set of time-domain statistics plus band-limited spectral energies. Pure and
fully vectorized: no Python-level loop over receivers or subcarriers.

Feature ordering (fixed, documented so downstream code can slice by name):

For each of the ``n_rx * S`` channels, in channel order
``(rx0, sc0), (rx0, sc1), ..., (rx0, scS-1), (rx1, sc0), ...`` (i.e. the
channel axis is ``rx`` then ``subcarrier``, row-major over the flattened
``(n_rx, S)`` grid), the following ``len(FEATURE_NAMES_PER_CHANNEL)``
features appear consecutively:

1. ``mean``               - time-domain mean
2. ``std``                 - time-domain standard deviation (ddof=0)
3. ``mad``                  - median absolute deviation about the median
4. ``min``                  - minimum
5. ``max``                  - maximum
6. ``p25``                  - 25th percentile
7. ``p75``                  - 75th percentile
8. ``band_energy_0_2hz``    - sum |rfft|^2 for frequencies in [0, 2) Hz
9. ``band_energy_2_10hz``   - sum |rfft|^2 for frequencies in [2, 10) Hz
10. ``band_energy_10_30hz`` - sum |rfft|^2 for frequencies in [10, 30) Hz

So the output feature vector for one window is the concatenation, over
channels in the order above, of these 10 values: total width
``n_rx * S * len(FEATURE_NAMES_PER_CHANNEL)``.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES_PER_CHANNEL: tuple[str, ...] = (
    "mean",
    "std",
    "mad",
    "min",
    "max",
    "p25",
    "p75",
    "band_energy_0_2hz",
    "band_energy_2_10hz",
    "band_energy_10_30hz",
)

_BAND_EDGES_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 10.0),
    (10.0, 30.0),
)


def extract_features(X: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Extract per-channel time + spectral features from CSI windows.

    Parameters
    ----------
    X: array (n, n_rx, T, S) — n windows, n_rx receivers, T time samples,
        S subcarriers. Not mutated.
    fs: sampling rate in Hz, used to compute FFT bin frequencies.

    Returns
    -------
    array (n, F) float32, where
    ``F == n_rx * S * len(FEATURE_NAMES_PER_CHANNEL)``.
    See module docstring for the exact feature ordering.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"expected X (n, n_rx, T, S), got shape {X.shape}")
    n, n_rx, t, s = X.shape
    x = X.astype(np.float64, copy=False)

    # Time-domain stats, computed along the time axis (axis=2), vectorized
    # across (n, n_rx, S).
    mean = np.mean(x, axis=2)
    std = np.std(x, axis=2)
    median = np.median(x, axis=2)
    mad = np.median(np.abs(x - median[:, :, None, :]), axis=2)
    minimum = np.min(x, axis=2)
    maximum = np.max(x, axis=2)
    p25 = np.percentile(x, 25, axis=2)
    p75 = np.percentile(x, 75, axis=2)

    # Spectral band energies via rfft along the time axis.
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    spectrum = np.fft.rfft(x, axis=2)
    power = np.abs(spectrum) ** 2  # (n, n_rx, n_freq, S)

    bands = []
    for lo, hi in _BAND_EDGES_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(np.sum(power[:, :, mask, :], axis=2))

    # Stack per-channel stats along a new last axis in the documented order,
    # then flatten (n_rx, S, n_stats) -> F, keeping channel-major ordering.
    stats = np.stack(
        [mean, std, mad, minimum, maximum, p25, p75, *bands], axis=-1
    )  # (n, n_rx, S, n_stats)
    assert stats.shape[-1] == len(FEATURE_NAMES_PER_CHANNEL)

    features = stats.reshape(n, n_rx * s * len(FEATURE_NAMES_PER_CHANNEL))
    return features.astype(np.float32)
