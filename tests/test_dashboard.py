"""Tests for the Phase 5 demo dashboard (csihar/dashboard.py).

Uses a stub engine (records feeds, returns canned Predictions on tick) and a
small synthetic Frame iterator, so no torch checkpoint or real model is
needed. Requires the optional ``demo`` extra (fastapi/uvicorn/httpx).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from csihar.dashboard import create_app  # noqa: E402


@dataclass
class _Prediction:
    ts: float
    raw_label: str
    confidence: float
    smoothed_label: str


@dataclass
class _StubEngine:
    """Records fed frames; tick() returns from a preset script."""

    tick_script: list[_Prediction] = field(default_factory=list)
    feeds: list[tuple[str, float, np.ndarray]] = field(default_factory=list)
    _tick_calls: int = 0

    def feed(self, rx_id: str, host_ts: float, amplitude: np.ndarray) -> None:
        self.feeds.append((rx_id, host_ts, amplitude))

    def tick(self, t_end: float):
        if self._tick_calls < len(self.tick_script):
            pred = self.tick_script[self._tick_calls]
            self._tick_calls += 1
            return pred
        return None


def _synthetic_frames(n_frames: int = 6, rx_ids=("rx1", "rx2")):
    """Yield (rx_id, host_ts, amplitude(64,)) frames, alternating receivers,
    spaced enough apart to trigger multiple hop ticks.

    A tiny real sleep between frames is deliberate: it paces the background
    drain thread against wall-clock time, the same way the production
    sources (``simulate_source``/``replay_source``) pace themselves via
    ``sleep_fn=time.sleep``. Without it, an unpaced iterator can race to
    completion (and publish "end" to zero registered clients) before a test
    websocket client has connected.
    """
    for i in range(n_frames):
        rx = rx_ids[i % len(rx_ids)]
        ts = i * 0.5
        amp = np.full(64, float(i + 1), dtype=np.float32)
        time.sleep(0.05)
        yield (rx, ts, amp)


def _make_app(tick_script, n_frames=6):
    engine = _StubEngine(tick_script=tick_script)

    def source_factory():
        return _synthetic_frames(n_frames=n_frames)

    app = create_app(engine, source_factory, hop_s=1.0, heatmap_hz=100.0)
    return app, engine


def _collect_messages(client: TestClient, max_messages: int = 200, timeout_s: float = 5.0):
    messages = []
    with client.websocket_connect("/ws") as ws:
        deadline = time.time() + timeout_s
        while time.time() < deadline and len(messages) < max_messages:
            try:
                msg = ws.receive_json()
            except Exception:
                break
            messages.append(msg)
            if msg.get("type") == "end":
                break
    return messages


def test_get_root_returns_html():
    app, _ = _make_app(tick_script=[])
    client = TestClient(app)
    with client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "WiFi CSI HAR" in resp.text


def test_websocket_receives_csi_messages_with_52_amplitudes():
    app, _ = _make_app(tick_script=[])
    client = TestClient(app)
    with client:
        messages = _collect_messages(client)
    csi_msgs = [m for m in messages if m["type"] == "csi"]
    assert len(csi_msgs) > 0
    for m in csi_msgs:
        assert len(m["amplitude"]) == 52
        assert m["rx_id"] in ("rx1", "rx2")
        assert isinstance(m["ts"], float)


def test_websocket_receives_prediction_messages_with_fall_flag():
    tick_script = [
        _Prediction(ts=1.0, raw_label="walking", confidence=0.9, smoothed_label="walking"),
        _Prediction(ts=2.0, raw_label="falling", confidence=0.95, smoothed_label="falling"),
    ]
    app, _ = _make_app(tick_script=tick_script)
    client = TestClient(app)
    with client:
        messages = _collect_messages(client)

    pred_msgs = [m for m in messages if m["type"] == "prediction"]
    assert len(pred_msgs) == 2

    walking_msg = pred_msgs[0]
    assert walking_msg["raw_label"] == "walking"
    assert walking_msg["smoothed_label"] == "walking"
    assert walking_msg["confidence"] == pytest.approx(0.9)
    assert walking_msg["fall"] is False

    falling_msg = pred_msgs[1]
    assert falling_msg["smoothed_label"] == "falling"
    assert falling_msg["fall"] is True


def test_websocket_receives_end_message_on_source_exhaustion():
    app, _ = _make_app(tick_script=[])
    client = TestClient(app)
    with client:
        messages = _collect_messages(client)
    assert messages[-1]["type"] == "end"


def test_engine_feed_called_for_every_frame():
    app, engine = _make_app(tick_script=[], n_frames=6)
    client = TestClient(app)
    with client:
        _collect_messages(client)
    assert len(engine.feeds) == 6


def test_two_concurrent_clients_both_receive_messages():
    tick_script = [
        _Prediction(ts=1.0, raw_label="sitting", confidence=0.8, smoothed_label="sitting"),
    ]
    app, _ = _make_app(tick_script=tick_script)
    client = TestClient(app)
    with client:
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            msgs1, msgs2 = [], []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                got_one = False
                for ws, bucket in ((ws1, msgs1), (ws2, msgs2)):
                    try:
                        msg = ws.receive_json()
                        bucket.append(msg)
                        got_one = True
                    except Exception:
                        pass
                    if bucket and bucket[-1].get("type") == "end":
                        continue
                if (msgs1 and msgs1[-1].get("type") == "end") and (
                    msgs2 and msgs2[-1].get("type") == "end"
                ):
                    break
                if not got_one:
                    break

    assert any(m["type"] == "csi" for m in msgs1)
    assert any(m["type"] == "csi" for m in msgs2)


def test_background_thread_shuts_down_cleanly():
    # Regression guard: app context manager (triggers startup/shutdown
    # events) must return promptly rather than hanging on the worker thread.
    app, _ = _make_app(tick_script=[])
    client = TestClient(app)
    start = time.time()
    with client:
        with client.websocket_connect("/ws") as ws:
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
    elapsed = time.time() - start
    assert elapsed < 15.0
