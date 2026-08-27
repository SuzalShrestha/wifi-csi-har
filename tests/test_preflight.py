"""Tests for the pre-session hardware check."""

import pytest

from csihar.collector import ReceiverState
from csihar.parser import METADATA_COLUMNS, parse_line
from csihar.preflight import evaluate, format_result, main, run_preflight
from csihar.simulate import generate_lines

RX_IDS = ("rx1", "rx2", "rx3")


def _state(receiver_id: str, duration_s: float = 5.0, seed: int = 0, **kwargs):
    """A ReceiverState pre-filled with realistic simulated frames."""
    lines = generate_lines("walking", duration_s=duration_s, seed=seed)
    frames = [parse_line(line, ts) for ts, line in lines]
    state = ReceiverState(port=f"/dev/{receiver_id}", receiver_id=receiver_id)
    state.frames = [f for f in frames if f is not None]
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def _healthy_states():
    return [_state(rx, seed=i) for i, rx in enumerate(RX_IDS)]


def _override_field(line: str, name: str, value: str) -> str:
    """Rewrite one metadata field of a CSI_DATA line; other lines pass through.

    Indexes by name off the parser's own column list rather than a magic
    position, and skips the simulator's leading firmware log line.
    """
    if not line.startswith("CSI_DATA,"):
        return line
    head, sep, data = line.partition(',"[')
    fields = head.split(",")
    fields[1 + METADATA_COLUMNS.index(name)] = value
    return f"{','.join(fields)}{sep}{data}"


def test_healthy_rig_passes():
    result = evaluate(_healthy_states(), RX_IDS)
    assert result.passed
    assert result.missing == ()
    assert not result.channel_mismatch
    assert len(result.reports) == 3
    assert "READY TO RECORD" in format_result(result)


def test_receiver_with_no_frames_is_reported_missing():
    states = _healthy_states()
    states[1].frames = []

    result = evaluate(states, RX_IDS)

    assert result.missing == ("rx2",)
    assert not result.passed
    assert "rx2: NO FRAMES" in format_result(result)
    assert "UART USB-C port" in format_result(result)


def test_absent_receiver_is_reported_missing():
    # Only two boards plugged in when three were expected.
    result = evaluate(_healthy_states()[:2], RX_IDS)
    assert result.missing == ("rx3",)
    assert not result.passed


def test_low_rate_fails_and_blames_traffic():
    # 3 frames spread over 5 s is the no-UDP-downlink signature.
    slow = _state("rx2", seed=1)
    slow.frames = [slow.frames[0], slow.frames[len(slow.frames) // 2], slow.frames[-1]]
    states = [_healthy_states()[0], slow, _healthy_states()[2]]

    result = evaluate(states, RX_IDS)

    assert not result.passed
    assert "--traffic" in format_result(result)


def test_channel_mismatch_fails():
    states = _healthy_states()
    # rx3 joined an AP on a different channel — its windows cannot be aligned
    # with the others'.
    frames = [
        parse_line(_override_field(line, "channel", "11"), ts)
        for ts, line in generate_lines("walking", duration_s=5.0, seed=9)
    ]
    states[2].frames = [f for f in frames if f is not None]

    result = evaluate(states, RX_IDS)

    assert result.channels == (6, 11)
    assert result.channel_mismatch
    assert not result.passed
    assert "different channels" in format_result(result)


def test_excess_parse_errors_fail():
    states = _healthy_states()
    states[0].parse_errors = 500
    result = evaluate(states, RX_IDS, max_parse_errors=10)
    assert not result.passed


def test_run_preflight_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def stub_read_receiver(state, stop):
        state.frames = _state(state.receiver_id).frames

    monkeypatch.setattr("csihar.preflight.read_receiver", stub_read_receiver)

    result = run_preflight({"/dev/f1": "rx1"}, duration_s=0.05)

    assert result.passed
    assert list(tmp_path.iterdir()) == []


def test_cli_exits_nonzero_when_not_ready(monkeypatch, capsys):
    def stub_read_receiver(state, stop):
        state.frames = []

    monkeypatch.setattr("csihar.preflight.read_receiver", stub_read_receiver)

    code = main(["--port", "/dev/f1=rx1", "--duration", "0.05"])

    assert code == 1
    out = capsys.readouterr()
    assert "NOT READY" in out.out
    assert "no --traffic IPs given" in out.err
