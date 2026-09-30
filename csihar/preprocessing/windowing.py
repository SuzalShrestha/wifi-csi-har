"""Uniform resampling, windowing, and multi-receiver alignment.

Packets arrive at a nominal rate (default 100 Hz) but with jitter and loss.
Strategy (matches the proposal): resample each receiver's stream onto a
uniform time grid, cut fixed-length windows, and align the three receivers at
the *window* level — per-packet correspondence across receivers is never
assumed because they run on independent clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    """One fixed-length, uniformly-sampled window from one receiver."""

    start_ts: float
    values: np.ndarray      # (n_samples, n_subcarriers), float32
    loss_fraction: float    # fraction of grid points with no packet nearby


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, float]:
    """Linearly interpolate onto a uniform grid [t_start, t_end) at fs Hz.

    Returns (resampled (n, n_subcarriers), loss_fraction) where loss_fraction
    is the share of grid points farther than 1.5/fs from any real packet.
    """
    if timestamps.ndim != 1 or len(timestamps) != len(values):
        raise ValueError("timestamps and values must align")
    if len(timestamps) < 2:
        raise ValueError("need at least 2 packets to resample")
    order = np.argsort(timestamps)
    ts, vals = timestamps[order], np.asarray(values)[order]

    # Deterministic sample count: np.arange over epoch-scale floats (~1.7e9)
    # can yield 299 or 301 points from accumulated rounding, and a single
    # off-by-one window breaks np.stack at assembly time.
    n_grid = int(round((t_end - t_start) * fs))
    grid = t_start + np.arange(n_grid) / fs
    out = np.empty((len(grid), vals.shape[1]), dtype=np.float32)
    for col in range(vals.shape[1]):
        out[:, col] = np.interp(grid, ts, vals[:, col])

    nearest = np.searchsorted(ts, grid)
    nearest = np.clip(nearest, 1, len(ts) - 1)
    dist = np.minimum(np.abs(ts[nearest] - grid), np.abs(ts[nearest - 1] - grid))
    loss = float(np.mean(dist > 1.5 / fs))
    return out, loss


def make_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    fs: float = 100.0,
    window_s: float = 3.0,
    hop_s: float = 1.5,
    max_loss: float = 0.10,
) -> list[Window]:
    """Cut uniformly-resampled, loss-screened windows from one stream."""
    if len(timestamps) == 0:
        return []
    # Sort once and hand each window only its bracketing slice. Passing the
    # whole stream made every window re-sort and copy all of it: 50 s per
    # receiver for a 25-minute session.
    order = np.argsort(timestamps)
    ts, vals = np.asarray(timestamps)[order], np.asarray(values)[order]
    t0, t1 = float(ts[0]), float(ts[-1])
    windows: list[Window] = []
    start = t0
    while start + window_s <= t1:
        lo = max(int(np.searchsorted(ts, start, "left")) - 1, 0)
        hi = min(int(np.searchsorted(ts, start + window_s, "right")) + 1, len(ts))
        resampled, loss = resample_uniform(
            ts[lo:hi], vals[lo:hi], fs, start, start + window_s
        )
        if loss <= max_loss:
            windows.append(Window(start_ts=start, values=resampled, loss_fraction=loss))
        start += hop_s
    return windows


def align_receivers(
    streams: dict[str, list[Window]], tolerance_s: float = 0.1
) -> list[dict[str, Window]]:
    """Group windows whose start times match across all receivers.

    streams: receiver id -> its windows (as produced by make_windows with the
    same window/hop settings). Returns one dict per aligned group; windows
    lacking a counterpart in any receiver are dropped.
    """
    if not streams:
        return []
    ids = sorted(streams)
    reference = streams[ids[0]]
    aligned: list[dict[str, Window]] = []
    for ref_win in reference:
        group = {ids[0]: ref_win}
        for other in ids[1:]:
            match = _closest(streams[other], ref_win.start_ts, tolerance_s)
            if match is None:
                break
            group[other] = match
        else:
            aligned.append(group)
    return aligned


def _closest(windows: list[Window], ts: float, tolerance_s: float) -> Window | None:
    best, best_dist = None, tolerance_s
    for win in windows:
        dist = abs(win.start_ts - ts)
        if dist <= best_dist:
            best, best_dist = win, dist
    return best
