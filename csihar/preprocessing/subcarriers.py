"""Subcarrier selection for ESP32 LLTF (20 MHz, 64-point FFT).

The ESP-IDF CSI buffer orders LLTF subcarriers as FFT bins 0..31 (DC and
positive frequencies) followed by bins 32..63 (negative frequencies).
802.11 L-LTF populates subcarriers -26..-1 and +1..+26 (52 total, pilots
included); DC and the guard bins carry nothing.

Because buffer ordering has bitten other projects before, we also provide an
empirical null-carrier detector — run it once on real captures and confirm it
agrees with `usable_lltf_indices()` before trusting either.
"""

from __future__ import annotations

import numpy as np

N_USABLE = 52


def usable_lltf_indices() -> np.ndarray:
    """Buffer indices of the 52 populated LLTF subcarriers (theory)."""
    positive = np.arange(1, 27)    # +1 .. +26
    negative = np.arange(38, 64)   # -26 .. -1
    return np.concatenate([positive, negative])


def detect_null_subcarriers(
    amplitudes: np.ndarray, rel_threshold: float = 0.05
) -> np.ndarray:
    """Empirically find dead subcarriers from data.

    amplitudes: (n_packets, n_subcarriers) array.
    Returns indices whose mean amplitude is below rel_threshold * overall mean
    of nonzero carriers.
    """
    if amplitudes.ndim != 2:
        raise ValueError("expected (n_packets, n_subcarriers)")
    mean_amp = amplitudes.mean(axis=0)
    scale = mean_amp[mean_amp > 0].mean() if (mean_amp > 0).any() else 0.0
    return np.where(mean_amp < rel_threshold * scale)[0]
