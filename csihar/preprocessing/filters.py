"""Denoising filters for CSI amplitude streams.

All functions are pure: input arrays are never mutated. Arrays are shaped
(n_samples, n_subcarriers) and filtered along the time axis (axis 0).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def hampel(x: np.ndarray, window: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace outlier spikes with the local median (vectorized, per column).

    window must be odd. Uses the MAD-based Hampel identifier, the standard
    first stage in CSI pipelines for impulsive RF/AGC spikes.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >= 3")
    x = np.asarray(x, dtype=np.float64)
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)) if x.ndim == 2 else pad, mode="edge")
    # sliding windows along time axis
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    med = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - med[..., None]), axis=-1)
    threshold = n_sigmas * 1.4826 * mad
    outlier = np.abs(x - med) > threshold
    return np.where(outlier, med, x)


def detrend_moving_mean(x: np.ndarray, window: int = 101) -> np.ndarray:
    """Remove slow drift (thermal/AGC) by subtracting a centered moving mean.

    Acts as a crude high-pass; keeps the motion-induced fluctuations that
    matter for HAR while removing the static channel component.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    x = np.asarray(x, dtype=np.float64)
    kernel = np.ones(window) / window
    if x.ndim == 1:
        trend = np.convolve(np.pad(x, (window // 2,), mode="edge"), kernel, "valid")
    else:
        padded = np.pad(x, ((window // 2, window // 2), (0, 0)), mode="edge")
        trend = np.apply_along_axis(lambda c: np.convolve(c, kernel, "valid"), 0, padded)
    return x - trend[: x.shape[0]]


def lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along the time axis.

    Human motion energy in CSI sits below ~20 Hz at 2.4 GHz; a 20–30 Hz cutoff
    at fs=100 Hz removes wideband noise without touching activity content.
    """
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff must be in (0, fs/2)")
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)
