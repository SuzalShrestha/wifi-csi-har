"""Tests for the UDP downlink traffic generator."""

import importlib
import socket
import time

import pytest

from csihar.traffic import TrafficGenerator, downlink_traffic


def test_requires_targets():
    with pytest.raises(ValueError):
        TrafficGenerator([])


def test_sends_paced_udp_to_all_targets():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    port = rx.getsockname()[1]

    gen = TrafficGenerator(
        ["127.0.0.1"], rate_hz=200.0, payload_bytes=64, port=port
    ).start()
    time.sleep(0.5)
    gen.stop()

    received = 0
    try:
        while True:
            data, _ = rx.recvfrom(1024)
            assert len(data) == 64
            received += 1
    except socket.timeout:
        pass
    rx.close()

    # ~100 expected in 0.5 s at 200 Hz; loose bounds — pacing, not precision
    assert 50 <= received <= 150
    assert received <= gen.sent  # loopback may drop under load; never exceeds


def test_downlink_traffic_is_a_noop_without_targets():
    for targets in (None, []):
        with downlink_traffic(targets) as gen:
            assert gen is None


def test_downlink_traffic_runs_for_the_block_then_stops():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    port = rx.getsockname()[1]

    with downlink_traffic(["127.0.0.1"], rate_hz=200.0) as gen:
        assert gen is not None
        gen.port = port
        time.sleep(0.2)
        assert gen._thread.is_alive()

    assert not gen._thread.is_alive()
    rx.close()


def test_downlink_traffic_stops_even_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with downlink_traffic(["127.0.0.1"]) as gen:
            captured = gen
            raise RuntimeError("boom")
    assert not captured._thread.is_alive()


@pytest.mark.parametrize(
    "module_name",
    ["collector", "session_script", "view", "realtime", "dashboard"],
)
def test_every_live_entry_point_exposes_traffic(module_name, capsys):
    """Regression: a live-capture CLI without --traffic records at <1 Hz.

    session_script (the pilot-dataset harness), realtime --live and dashboard
    --live all shipped without the flag, which would have silently produced
    worthless sessions against a real router.
    """
    module = importlib.import_module(f"csihar.{module_name}")
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    # Match the rendered flag, not a substring — "--trafficXX" must not pass.
    assert "--traffic IP" in capsys.readouterr().out
