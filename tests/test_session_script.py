import contextlib
from pathlib import Path

import pandas as pd
import pytest

from csihar import session_script
from csihar.parser import parse_line
from csihar.session_script import (
    Segment,
    TimedSegment,
    parse_script,
    run_scripted_session,
    segments_to_labels_json,
)
from csihar.simulate import generate_lines


def test_parse_script_valid():
    segments = parse_script("background:10,walking:60,sitting:60")
    assert segments == (
        Segment(label="background", duration_s=10.0),
        Segment(label="walking", duration_s=60.0),
        Segment(label="sitting", duration_s=60.0),
    )


def test_parse_script_bad_label_raises():
    with pytest.raises(ValueError, match="unknown label"):
        parse_script("cartwheeling:10")


def test_parse_script_zero_duration_raises():
    with pytest.raises(ValueError, match="duration > 0"):
        parse_script("walking:0")


def test_parse_script_negative_duration_raises():
    with pytest.raises(ValueError, match="duration > 0"):
        parse_script("walking:-5")


def test_parse_script_empty_raises():
    with pytest.raises(ValueError):
        parse_script("")


def test_segments_to_labels_json_shape():
    timed = (
        TimedSegment(label="background", start_ts=1000.0, end_ts=1010.0),
        TimedSegment(label="walking", start_ts=1010.0, end_ts=1070.0),
    )
    result = segments_to_labels_json(timed)
    assert result == {
        "segments": [
            {"label": "background", "start_ts": 1000.0, "end_ts": 1010.0},
            {"label": "walking", "start_ts": 1010.0, "end_ts": 1070.0},
        ]
    }
    # ordered, non-overlapping
    starts_ends = [(s["start_ts"], s["end_ts"]) for s in result["segments"]]
    for (s1, e1), (s2, e2) in zip(starts_ends, starts_ends[1:]):
        assert e1 <= s2


class _FakeClock:
    """Advances a fixed step each call so segment wait-loops exit quickly."""

    def __init__(self, start: float = 1000.0, step: float = 0.05):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _make_stub_read_receiver(n_frames: int = 20):
    """Build a read_receiver replacement that appends synthetic frames.

    Uses the real simulator + parser so frames are realistic CsiFrame
    objects, but never touches serial hardware.
    """
    lines = generate_lines("walking", duration_s=1.0, seed=0)
    frames = [parse_line(line, ts) for ts, line in lines]
    frames = [f for f in frames if f is not None][:n_frames]

    def stub_read_receiver(state, stop):
        for frame in frames:
            state.frames.append(frame)

    return stub_read_receiver


def test_run_scripted_session_with_fakes(tmp_path, monkeypatch):
    stub = _make_stub_read_receiver(n_frames=15)
    monkeypatch.setattr(session_script, "read_receiver", stub)

    prompts: list[str] = []

    def fake_prompt(msg: str) -> str:
        prompts.append(msg)
        return ""

    fake_clock = _FakeClock(start=1000.0, step=0.05)

    ports = {"/dev/fake1": "rx1", "/dev/fake2": "rx2"}
    script = parse_script("background:0.1,walking:0.15")

    session_dir = run_scripted_session(
        ports=ports,
        out_dir=tmp_path,
        script=script,
        subject="testsubject",
        environment="room_a",
        notes="unit test",
        prompt_fn=fake_prompt,
        clock=fake_clock,
    )

    assert session_dir.parent == tmp_path
    assert session_dir.name.endswith("testsubject_scripted")
    assert session_dir.is_dir()

    metadata_path = session_dir / "metadata.json"
    assert metadata_path.exists()

    labels_path = session_dir / "labels.json"
    assert labels_path.exists()
    import json

    labels = json.loads(labels_path.read_text())
    segments = labels["segments"]
    assert len(segments) == 2
    assert [s["label"] for s in segments] == ["background", "walking"]

    # ordered, non-overlapping, valid timestamps
    for seg in segments:
        assert seg["start_ts"] < seg["end_ts"]
    for prev, cur in zip(segments, segments[1:]):
        assert prev["end_ts"] <= cur["start_ts"]

    for receiver_id in ports.values():
        parquet_path = session_dir / f"{receiver_id}.parquet"
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) == 15

    assert len(prompts) == 2
    assert "background" in prompts[0]
    assert "walking" in prompts[1]


def test_scripted_session_runs_downlink_traffic(tmp_path, monkeypatch):
    """The pilot harness must flood the receivers, or it records at <1 Hz."""
    monkeypatch.setattr(
        session_script, "read_receiver", _make_stub_read_receiver(n_frames=5)
    )
    seen: list[tuple[str, ...]] = []

    @contextlib.contextmanager
    def spy_traffic(targets, *args, **kwargs):
        seen.append(tuple(targets or ()))
        yield None

    monkeypatch.setattr(session_script, "downlink_traffic", spy_traffic)

    run_scripted_session(
        ports={"/dev/fake1": "rx1"},
        out_dir=tmp_path,
        script=parse_script("walking:0.1"),
        subject="s1",
        environment="room_a",
        prompt_fn=lambda _: "",
        clock=_FakeClock(step=0.05),
        traffic_ips=["192.168.1.97", "192.168.1.98"],
    )

    assert seen == [("192.168.1.97", "192.168.1.98")]


def test_scripted_session_without_traffic_ips_still_runs(tmp_path, monkeypatch):
    # Simulator/offline use must not require IPs.
    monkeypatch.setattr(
        session_script, "read_receiver", _make_stub_read_receiver(n_frames=5)
    )
    session_dir = run_scripted_session(
        ports={"/dev/fake1": "rx1"},
        out_dir=tmp_path,
        script=parse_script("walking:0.1"),
        subject="s1",
        environment="room_a",
        prompt_fn=lambda _: "",
        clock=_FakeClock(step=0.05),
    )
    assert (session_dir / "labels.json").exists()


def _spy_on_run(monkeypatch) -> dict:
    """Replace run_scripted_session with a recorder, so main() can be tested
    without serial ports or a real prompt."""
    calls: dict = {}

    def fake_run(*args, **kwargs):
        calls.update(kwargs)
        return Path("session")

    monkeypatch.setattr(session_script, "run_scripted_session", fake_run)
    return calls


def test_cli_forwards_traffic_ips(tmp_path, monkeypatch):
    calls = _spy_on_run(monkeypatch)

    session_script.main(
        [
            "--port", "/dev/fake1=rx1",
            "--script", "walking:10",
            "--subject", "s1",
            "--out", str(tmp_path),
            "--traffic", "192.168.1.97",
            "--traffic", "192.168.1.98",
        ]
    )

    assert calls["traffic_ips"] == ["192.168.1.97", "192.168.1.98"]


def test_cli_warns_when_no_traffic_ips_given(tmp_path, monkeypatch, capsys):
    _spy_on_run(monkeypatch)

    session_script.main(
        [
            "--port", "/dev/fake1=rx1",
            "--script", "walking:10",
            "--subject", "s1",
            "--out", str(tmp_path),
        ]
    )

    assert "no --traffic IPs given" in capsys.readouterr().err


def test_lead_in_delays_the_labelled_span(tmp_path, monkeypatch):
    """Background needs the room empty BEFORE the label starts.

    Without this, a lone subject has to record background as its own session,
    and the model then learns the session rather than the absence of a person.
    """
    monkeypatch.setattr(
        session_script, "read_receiver", _make_stub_read_receiver(n_frames=5)
    )
    clock = _FakeClock(start=1000.0, step=1.0)

    session_dir = run_scripted_session(
        ports={"/dev/fake1": "rx1"},
        out_dir=tmp_path,
        script=parse_script("background:5"),
        subject="s1",
        environment="room_a",
        prompt_fn=lambda _: "",
        clock=clock,
        lead_in_s=10.0,
    )

    import json

    segment = json.loads((session_dir / "labels.json").read_text())["segments"][0]
    # Enter is pressed at ~1000; the label must not start until the lead-in
    # has elapsed, so the subject's walk out of the room is excluded.
    assert segment["start_ts"] >= 1010.0
    assert segment["end_ts"] - segment["start_ts"] >= 5.0


def test_zero_lead_in_starts_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session_script, "read_receiver", _make_stub_read_receiver(n_frames=5)
    )
    clock = _FakeClock(start=1000.0, step=1.0)

    session_dir = run_scripted_session(
        ports={"/dev/fake1": "rx1"},
        out_dir=tmp_path,
        script=parse_script("walking:5"),
        subject="s1",
        environment="room_a",
        prompt_fn=lambda _: "",
        clock=clock,
    )

    import json

    segment = json.loads((session_dir / "labels.json").read_text())["segments"][0]
    assert segment["start_ts"] < 1010.0


def test_cli_forwards_lead_in(tmp_path, monkeypatch):
    calls = _spy_on_run(monkeypatch)
    session_script.main(
        [
            "--port", "/dev/fake1=rx1", "--script", "background:10",
            "--subject", "s1", "--out", str(tmp_path), "--lead-in", "15",
        ]
    )
    assert calls["lead_in_s"] == 15.0
