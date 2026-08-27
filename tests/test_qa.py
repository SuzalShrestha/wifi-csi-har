from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csihar.parser import parse_line
from csihar.qa import ABSENT, ReceiverReport, check_receiver, check_session
from csihar.simulate import generate_lines
from csihar.storage import frames_to_dataframe


def _write_session(
    tmp_path: Path,
    activity: str = "walking",
    duration_s: float = 2.0,
    seed: int = 1,
    keep_every: int | None = None,
    receiver_id: str = "rx1",
) -> Path:
    """Build a session dir with one receiver's parquet from simulated frames."""
    lines = generate_lines(activity, duration_s=duration_s, seed=seed)
    frames = [parse_line(line, ts) for ts, line in lines]
    frames = [f for f in frames if f is not None]
    if keep_every is not None:
        frames = [f for i, f in enumerate(frames) if i % keep_every != 0]
    df = frames_to_dataframe(frames)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    df.to_parquet(session_dir / f"{receiver_id}.parquet")
    return session_dir


def test_clean_simulated_session_passes(tmp_path):
    session_dir = _write_session(tmp_path, duration_s=3.0, seed=1)
    reports = check_session(session_dir)
    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, ReceiverReport)
    assert report.n_frames > 250
    assert report.mean_rate_hz > 90.0
    assert report.seq_gap_fraction < 0.05
    assert report.passed


def test_dropped_frames_session_fails(tmp_path):
    # Drop every 3rd frame on top of the simulator's own ~2% serial drop —
    # this should blow both the rate and the gap-fraction thresholds.
    session_dir = _write_session(
        tmp_path, duration_s=3.0, seed=2, keep_every=3, receiver_id="rx1"
    )
    reports = check_session(session_dir)
    assert len(reports) == 1
    report = reports[0]
    assert not report.passed
    assert report.mean_rate_hz < 90.0 or report.seq_gap_fraction > 0.05


def test_too_few_frames_fails_gracefully(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    lines = generate_lines("walking", duration_s=2.0, seed=3)
    frames = [parse_line(line, ts) for ts, line in lines]
    frames = [f for f in frames if f is not None][:1]  # keep just one frame
    df = frames_to_dataframe(frames)
    path = session_dir / "rx1.parquet"
    df.to_parquet(path)

    report = check_receiver(path)
    assert report.n_frames == 1
    assert not report.passed


def test_check_receiver_does_not_mutate_or_write(tmp_path):
    session_dir = _write_session(tmp_path, duration_s=2.0, seed=4)
    path = session_dir / "rx1.parquet"
    before = pd.read_parquet(path)
    mtime_before = path.stat().st_mtime_ns

    check_receiver(path)

    after = pd.read_parquet(path)
    pd.testing.assert_frame_equal(before, after)
    assert path.stat().st_mtime_ns == mtime_before
    # No stray files should appear alongside the parquet.
    assert sorted(p.name for p in session_dir.iterdir()) == ["rx1.parquet"]


def test_check_session_handles_multiple_receivers(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    for i, receiver_id in enumerate(["rx1", "rx2"]):
        lines = generate_lines("walking", duration_s=2.0, seed=10 + i)
        frames = [parse_line(line, ts) for ts, line in lines]
        frames = [f for f in frames if f is not None]
        df = frames_to_dataframe(frames)
        df.to_parquet(session_dir / f"{receiver_id}.parquet")

    reports = check_session(session_dir)
    assert {r.receiver_id for r in reports} == {"rx1", "rx2"}


def _session_dataframe(duration_s: float = 3.0, seed: int = 1) -> pd.DataFrame:
    """Parsed simulator frames for one receiver, as the stored Parquet schema."""
    lines = generate_lines("walking", duration_s=duration_s, seed=seed)
    frames = [parse_line(line, ts) for ts, line in lines]
    return frames_to_dataframe([f for f in frames if f is not None])


def _write_dataframe(tmp_path: Path, df: pd.DataFrame) -> Path:
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    df.to_parquet(session_dir / "rx1.parquet")
    return session_dir / "rx1.parquet"


def test_clean_session_reports_pinned_radio_config(tmp_path):
    # The simulator emits a fixed channel 6 / BW20 / MCS 7 link.
    report = check_receiver(_write_dataframe(tmp_path, _session_dataframe()))
    assert report.dominant_channel == 6
    assert report.dominant_bandwidth == 0
    assert report.dominant_mcs == 7
    assert report.mcs_mode_fraction == 1.0
    assert not report.radio_config_drift
    assert report.passed


def test_channel_hop_mid_session_fails(tmp_path):
    # An auto-channel router moving 6 -> 11 halfway through splits the session
    # across two radio environments; it must not pass QA silently.
    df = _session_dataframe()
    channel = df["channel"].to_numpy().copy()
    channel[len(channel) // 2 :] = 11
    path = _write_dataframe(tmp_path, df.assign(channel=channel))

    report = check_receiver(path)
    assert report.radio_config_drift
    assert not report.passed


def test_bandwidth_change_fails(tmp_path):
    df = _session_dataframe()
    bandwidth = df["bandwidth"].to_numpy().copy()
    bandwidth[-50:] = 1
    path = _write_dataframe(tmp_path, df.assign(bandwidth=bandwidth))

    report = check_receiver(path)
    assert report.radio_config_drift
    assert not report.passed


def test_unstable_rate_adaptation_fails(tmp_path):
    # A fifth of the frames at a different MCS is below the 0.9 default.
    df = _session_dataframe()
    mcs = df["mcs"].to_numpy().copy()
    mcs[::5] = 5
    path = _write_dataframe(tmp_path, df.assign(mcs=mcs))

    report = check_receiver(path)
    assert report.dominant_mcs == 7
    assert report.mcs_mode_fraction == pytest.approx(0.8, abs=0.01)
    assert not report.radio_config_drift  # rate adaptation is not config drift
    assert not report.passed


def test_occasional_rate_change_still_passes(tmp_path):
    df = _session_dataframe()
    mcs = df["mcs"].to_numpy().copy()
    mcs[::50] = 5  # 2% of frames — well inside normal rate adaptation
    path = _write_dataframe(tmp_path, df.assign(mcs=mcs))

    assert check_receiver(path).passed


def test_legacy_session_without_mcs_column_still_passes(tmp_path):
    # Sessions recorded before mcs/bandwidth were persisted must stay readable:
    # the absent columns are reported as ABSENT and excluded from the checks.
    df = _session_dataframe().drop(columns=["mcs", "bandwidth"])
    report = check_receiver(_write_dataframe(tmp_path, df))

    assert report.dominant_mcs == ABSENT
    assert report.dominant_bandwidth == ABSENT
    assert report.dominant_channel == 6
    assert np.isnan(report.mcs_mode_fraction)
    assert not report.radio_config_drift
    assert report.passed
