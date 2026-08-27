"""Post-collection session QA: per-receiver quality checks and a CLI report.

Checks each receiver's Parquet file for capture rate, sequence gaps, RSSI
sanity, radio-configuration drift (channel/bandwidth must stay pinned, MCS
must stay dominated by one rate), and whether the empirical null-subcarrier
detector agrees with the theoretical usable-subcarrier set. Run this
immediately after every
collect-session (see `/validate-session` skill) before accepting data into
the dataset.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import detect_null_subcarriers, usable_lltf_indices


@dataclass(frozen=True)
class ReceiverReport:
    """Immutable QA summary for one receiver's Parquet file."""

    receiver_id: str
    n_frames: int
    duration_s: float
    mean_rate_hz: float
    seq_gap_fraction: float
    rssi_mean: float
    rssi_std: float
    dominant_channel: int
    dominant_bandwidth: int
    radio_config_drift: bool
    dominant_mcs: int
    mcs_mode_fraction: float
    null_subcarrier_mismatch: bool
    passed: bool


# Reported for columns absent from sessions recorded before they were persisted.
ABSENT = -1


def _too_few_frames(receiver_id: str, n_frames: int) -> ReceiverReport:
    return ReceiverReport(
        receiver_id=receiver_id,
        n_frames=n_frames,
        duration_s=0.0,
        mean_rate_hz=0.0,
        seq_gap_fraction=1.0,
        rssi_mean=float("nan"),
        rssi_std=float("nan"),
        dominant_channel=ABSENT,
        dominant_bandwidth=ABSENT,
        radio_config_drift=False,
        dominant_mcs=ABSENT,
        mcs_mode_fraction=float("nan"),
        null_subcarrier_mismatch=True,
        passed=False,
    )


def _seq_gap_fraction(seq: np.ndarray) -> float:
    """Fraction of missing sequence numbers between first and last (inclusive)."""
    lo, hi = int(seq.min()), int(seq.max())
    expected = hi - lo + 1
    if expected <= 0:
        return 1.0
    present = np.unique(seq).size
    return max(0.0, (expected - present) / expected)


def _mode_and_fraction(values: np.ndarray) -> tuple[int, float]:
    """Most common value in `values` and the fraction of rows holding it."""
    unique, counts = np.unique(values, return_counts=True)
    winner = int(np.argmax(counts))
    return int(unique[winner]), float(counts[winner] / values.size)


def _pinned_value(df: pd.DataFrame, column: str) -> tuple[int, bool]:
    """(dominant value, drifted) for a column that must be constant.

    Channel and bandwidth are pinned for the whole dataset, so any change
    inside a single session means its frames straddle two radio environments
    and the session is not usable as one unit. A column missing entirely
    (legacy session) is reported absent and not treated as drift.
    """
    if column not in df.columns:
        return ABSENT, False
    value, fraction = _mode_and_fraction(df[column].to_numpy())
    return value, fraction < 1.0


def _dominant_mcs(df: pd.DataFrame) -> tuple[int, float]:
    """(dominant MCS, fraction of frames at it).

    Unlike channel, MCS is picked per packet by the router's rate adaptation,
    so occasional changes are normal — it is the *fraction* at the dominant
    rate that says whether the link held steady. NaN when the column is absent.
    """
    if "mcs" not in df.columns:
        return ABSENT, float("nan")
    return _mode_and_fraction(df["mcs"].to_numpy())


def _null_subcarrier_mismatch(amplitude: np.ndarray) -> bool:
    """True if empirical nulls disagree with theory inside the usable band.

    Only empirical nulls that fall *inside* the theoretical usable set count
    as a mismatch — extra nulls outside it (the already-known DC/guard band)
    are expected and not a disagreement.
    """
    usable = set(usable_lltf_indices().tolist())
    empirical_nulls = set(detect_null_subcarriers(amplitude).tolist())
    return bool(empirical_nulls & usable)


def check_receiver(
    path: Path,
    min_rate_hz: float = 90.0,
    max_gap_fraction: float = 0.05,
    min_mcs_fraction: float = 0.9,
) -> ReceiverReport:
    """Load one receiver's Parquet file and compute its QA report. Pure/read-only."""
    receiver_id = path.stem
    df = pd.read_parquet(path)
    n_frames = len(df)

    if n_frames < 2:
        return _too_few_frames(receiver_id, n_frames)

    host_ts = df["host_ts"].to_numpy()
    seq = df["seq"].to_numpy()
    rssi = df["rssi"].to_numpy()

    duration_s = float(host_ts.max() - host_ts.min())
    mean_rate_hz = (n_frames - 1) / duration_s if duration_s > 0 else 0.0
    gap_fraction = _seq_gap_fraction(seq)

    channel, channel_drift = _pinned_value(df, "channel")
    bandwidth, bandwidth_drift = _pinned_value(df, "bandwidth")
    radio_config_drift = channel_drift or bandwidth_drift
    mcs, mcs_fraction = _dominant_mcs(df)
    # NaN compares False, so an absent mcs column is simply not checked.
    mcs_unstable = bool(mcs_fraction < min_mcs_fraction)

    real = np.stack(df["csi_real"].to_numpy())
    imag = np.stack(df["csi_imag"].to_numpy())
    amplitude = np.abs(real + 1j * imag)
    finite_rows = np.isfinite(amplitude).all(axis=1)
    amplitude = amplitude[finite_rows]
    mismatch = _null_subcarrier_mismatch(amplitude) if amplitude.size else True

    passed = (
        mean_rate_hz >= min_rate_hz
        and gap_fraction <= max_gap_fraction
        and not radio_config_drift
        and not mcs_unstable
        and not mismatch
    )

    return ReceiverReport(
        receiver_id=receiver_id,
        n_frames=n_frames,
        duration_s=duration_s,
        mean_rate_hz=mean_rate_hz,
        seq_gap_fraction=gap_fraction,
        rssi_mean=float(rssi.mean()),
        rssi_std=float(rssi.std()),
        dominant_channel=channel,
        dominant_bandwidth=bandwidth,
        radio_config_drift=radio_config_drift,
        dominant_mcs=mcs,
        mcs_mode_fraction=mcs_fraction,
        null_subcarrier_mismatch=mismatch,
        passed=passed,
    )


def check_session(
    session_dir: Path,
    min_rate_hz: float = 90.0,
    max_gap_fraction: float = 0.05,
    min_mcs_fraction: float = 0.9,
) -> list[ReceiverReport]:
    """Run check_receiver over every *.parquet file in a session directory."""
    paths = sorted(session_dir.glob("*.parquet"))
    return [
        check_receiver(p, min_rate_hz, max_gap_fraction, min_mcs_fraction)
        for p in paths
    ]


def _format_report(report: ReceiverReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    return (
        f"{report.receiver_id}: n_frames={report.n_frames} "
        f"duration_s={report.duration_s:.2f} rate_hz={report.mean_rate_hz:.2f} "
        f"gap_frac={report.seq_gap_fraction:.4f} rssi_mean={report.rssi_mean:.2f} "
        f"rssi_std={report.rssi_std:.2f} null_mismatch={report.null_subcarrier_mismatch} "
        f"-> {status}\n"
        f"  radio: ch={report.dominant_channel} bw={report.dominant_bandwidth} "
        f"drift={report.radio_config_drift} "
        f"mcs={report.dominant_mcs} mcs_frac={report.mcs_mode_fraction:.3f}"
    )


def _save_heatmap(path: Path, out_dir: Path) -> None:
    """Load a receiver's amplitude and save a heatmap PNG, reusing view.show_static."""
    from .storage import load_amplitudes
    from .view import show_static

    _, amplitude = load_amplitudes(path)
    finite_rows = np.isfinite(amplitude).all(axis=1)
    amplitude = amplitude[finite_rows]
    if amplitude.shape[0] < 2:
        return
    out = out_dir / f"{path.stem}_qa_heatmap.png"
    show_static(amplitude, f"CSI amplitude — {path.stem}", out)


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(
        description="Session QA: rate/gap/radio-config/subcarrier checks"
    )
    ap.add_argument("session_dir", type=Path, help="datasets/raw/<session> directory")
    ap.add_argument("--min-rate-hz", type=float, default=90.0)
    ap.add_argument("--max-gap-fraction", type=float, default=0.05)
    ap.add_argument("--min-mcs-fraction", type=float, default=0.9)
    args = ap.parse_args(argv)

    reports = check_session(
        args.session_dir,
        args.min_rate_hz,
        args.max_gap_fraction,
        args.min_mcs_fraction,
    )
    if not reports:
        print(f"no *.parquet files found in {args.session_dir}")
        return 1

    for report in reports:
        print(_format_report(report))

    for path in sorted(args.session_dir.glob("*.parquet")):
        _save_heatmap(path, args.session_dir)

    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
