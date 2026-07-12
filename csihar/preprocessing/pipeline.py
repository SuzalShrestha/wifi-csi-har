"""Fixed per-receiver preprocessing pipeline: raw amplitude stream -> windows.

Chain (matches the proposal): usable-subcarrier selection -> Hampel outlier
removal -> moving-mean detrend -> uniform resampling + windowing. Pure — the
input arrays are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import detrend_moving_mean, hampel
from .subcarriers import usable_lltf_indices
from .windowing import Window, make_windows


@dataclass(frozen=True)
class PreprocessConfig:
    """All knobs of the session -> tensor pipeline, in one committed place."""

    fs: float = 100.0
    window_s: float = 3.0
    hop_s: float = 1.5
    max_loss: float = 0.10
    hampel_window: int = 11
    hampel_sigmas: float = 3.0
    detrend_window: int = 101
    align_tolerance_s: float = 0.1
    boundary_margin_s: float = 0.5


def preprocess_stream(
    host_ts: np.ndarray, amplitudes: np.ndarray, cfg: PreprocessConfig
) -> list[Window]:
    """One receiver's raw stream -> denoised, loss-screened windows.

    host_ts: (n,) host-clock epoch seconds; amplitudes: (n, n_subcarriers)
    with the full 64-bin LLTF buffer (the 52 usable bins are selected here).
    """
    host_ts = np.asarray(host_ts, dtype=np.float64)
    amps = np.asarray(amplitudes, dtype=np.float64)
    if host_ts.ndim != 1 or amps.ndim != 2 or len(host_ts) != len(amps):
        raise ValueError("expected host_ts (n,) and amplitudes (n, n_subcarriers)")
    if len(host_ts) == 0:
        return []
    usable = usable_lltf_indices()
    if amps.shape[1] <= int(usable.max()):
        raise ValueError(
            f"amplitudes has {amps.shape[1]} subcarriers, "
            f"need the full {int(usable.max()) + 1}-bin LLTF buffer"
        )
    x = amps[:, usable]
    x = hampel(x, window=cfg.hampel_window, n_sigmas=cfg.hampel_sigmas)
    x = detrend_moving_mean(x, window=cfg.detrend_window)
    return make_windows(
        host_ts,
        x,
        fs=cfg.fs,
        window_s=cfg.window_s,
        hop_s=cfg.hop_s,
        max_loss=cfg.max_loss,
    )
