"""Per-subcarrier standardization.

Scaler statistics MUST be fit on the training split only and then applied
unchanged to validation/test — fitting on the full dataset is a leak that
inflates every published number it touches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalerParams:
    mean: np.ndarray  # (n_subcarriers,)
    std: np.ndarray   # (n_subcarriers,)


def fit_scaler(train_windows: np.ndarray) -> ScalerParams:
    """train_windows: (n_windows, n_samples, n_subcarriers)."""
    if train_windows.ndim != 3:
        raise ValueError("expected (n_windows, n_samples, n_subcarriers)")
    flat = train_windows.reshape(-1, train_windows.shape[-1])
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard dead subcarriers
    return ScalerParams(mean=flat.mean(axis=0), std=std)


def apply_scaler(windows: np.ndarray, params: ScalerParams) -> np.ndarray:
    """Returns a new standardized array; input is not mutated."""
    return ((windows - params.mean) / params.std).astype(np.float32)
