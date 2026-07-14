"""Tests for the UDP downlink traffic generator."""

import socket
import time

import pytest

from csihar.traffic import TrafficGenerator


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
