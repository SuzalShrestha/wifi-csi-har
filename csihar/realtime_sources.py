"""Frame sources for the Phase 5 real-time engine: replay, simulate, live.

Split out of ``realtime.py`` to keep that module under the line budget. Each
source yields ``(rx_id, host_ts, amplitude)`` tuples in host-clock order,
mirroring how ``RealtimeEngine.feed`` expects data.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .parser import parse_line
from .simulate import generate_lines
from .storage import load_amplitudes

Frame = tuple[str, float, np.ndarray]  # (rx_id, host_ts, amplitude)


def replay_source(
    session_dir: Path,
    speed: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Iterator[Frame]:
    """Replay a stored session's receiver parquets merged in host_ts order."""
    session_dir = Path(session_dir)
    parquets = sorted(session_dir.glob("*.parquet"))
    if not parquets:
        raise ValueError(f"no receiver parquet files under {session_dir}")

    merged: list[tuple[float, str, np.ndarray]] = []
    for path in parquets:
        rx_id = path.stem
        host_ts, amplitude = load_amplitudes(path)
        for ts, amp in zip(host_ts, amplitude):
            merged.append((float(ts), rx_id, amp.astype(np.float32)))
    merged.sort(key=lambda item: item[0])

    prev_ts: float | None = None
    for ts, rx_id, amp in merged:
        if prev_ts is not None and speed > 0:
            gap = ts - prev_ts
            if gap > 0:
                sleep_fn(gap / speed)
        prev_ts = ts
        yield rx_id, ts, amp


def simulate_source(
    activity: str,
    duration_s: float,
    rx_ids: tuple[str, ...] = ("rx1", "rx2", "rx3"),
    t0: float = 0.0,
    sleep_fn: Callable[[float], None] = lambda _: None,
) -> Iterator[Frame]:
    """Generate a synthetic multi-receiver stream (no hardware required)."""
    merged: list[tuple[float, str, np.ndarray]] = []
    for i, rx in enumerate(rx_ids):
        lines = generate_lines(activity=activity, duration_s=duration_s, seed=1000 * i)
        for ts, line in lines:
            frame = parse_line(line, host_ts=t0 + ts)
            if frame is not None:
                merged.append((frame.host_ts, rx, frame.amplitude))
    merged.sort(key=lambda item: item[0])

    prev_ts: float | None = None
    for ts, rx_id, amp in merged:
        if prev_ts is not None:
            gap = ts - prev_ts
            if gap > 0:
                sleep_fn(gap)
        prev_ts = ts
        yield rx_id, ts, amp


def live_source(ports: dict[str, str]) -> Iterator[Frame]:
    """Read live CSI frames from serial ports: {device_path: rx_id}.

    One reader thread per port pushes parsed frames into a shared queue; this
    generator drains the queue. Only exercised with real hardware attached.
    """
    import serial

    from .collector import BAUD_RATE

    q: queue.Queue[Frame] = queue.Queue()
    stop = threading.Event()

    def _reader(port: str, rx_id: str) -> None:
        with serial.Serial(port, BAUD_RATE, timeout=1) as conn:
            conn.reset_input_buffer()
            while not stop.is_set():
                raw = conn.readline()
                if not raw:
                    continue
                host_ts = time.time()
                frame = parse_line(raw.decode("utf-8", errors="replace"), host_ts)
                if frame is None:
                    continue
                q.put((rx_id, frame.host_ts, frame.amplitude))

    threads = [
        threading.Thread(target=_reader, args=(port, rx_id), daemon=True)
        for port, rx_id in ports.items()
    ]
    for t in threads:
        t.start()
    try:
        while True:
            yield q.get()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3)
