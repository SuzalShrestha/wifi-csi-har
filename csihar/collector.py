"""Multi-receiver CSI collector.

One thread per serial port reads CSI_DATA lines, timestamps them on arrival
(host clock — the shared reference for window-level alignment), and parses
them into frames. The main thread periodically reports rates/loss and writes
one Parquet file per receiver at session end.

Usage:
    python -m csihar.collector --port /dev/cu.usbmodem101=rx1 \
        --port /dev/cu.usbmodem102=rx2 --port /dev/cu.usbmodem103=rx3 \
        --label walking --subject sujal --env room_a \
        --duration 60 --out datasets/raw
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import serial

from .parser import CsiFrame, CsiParseError, parse_line
from .storage import frames_to_dataframe, write_session_metadata
from .traffic import add_traffic_argument, downlink_traffic

BAUD_RATE = 921600  # 115200 cannot sustain 100 pkt/s * ~700 B/line


@dataclass
class ReceiverState:
    port: str
    receiver_id: str
    frames: list[CsiFrame] = field(default_factory=list)
    parse_errors: int = 0
    last_seq: int | None = None
    seq_gaps: int = 0


def read_receiver(state: ReceiverState, stop: threading.Event) -> None:
    """Serial read loop for one receiver. Appends to state (thread-owned)."""
    with serial.Serial(state.port, BAUD_RATE, timeout=1) as conn:
        conn.reset_input_buffer()
        while not stop.is_set():
            raw = conn.readline()
            if not raw:
                continue
            host_ts = time.time()
            try:
                frame = parse_line(raw.decode("utf-8", errors="replace"), host_ts)
            except CsiParseError:
                state.parse_errors += 1
                continue
            if frame is None:
                continue
            if state.last_seq is not None and frame.seq > state.last_seq + 1:
                state.seq_gaps += frame.seq - state.last_seq - 1
            state.last_seq = frame.seq
            state.frames.append(frame)


def run_session(
    ports: dict[str, str],
    out_dir: Path,
    label: str,
    subject: str,
    environment: str,
    duration_s: float,
    notes: str = "",
    traffic_ips: list[str] | None = None,
) -> Path:
    session_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{subject}_{label}"
    session_dir = out_dir / session_name
    write_session_metadata(
        session_dir, label=label, subject=subject, environment=environment,
        receivers=ports, notes=notes,
    )

    stop = threading.Event()
    states = [ReceiverState(port=p, receiver_id=r) for p, r in ports.items()]
    threads = [
        threading.Thread(target=read_receiver, args=(s, stop), daemon=True)
        for s in states
    ]

    # Router sends sparse frames at DSSS rates (no CSI on ESP32-S3); steady
    # UDP downlink forces OFDM/HT rates and ~100 Hz CSI. See csihar/traffic.py.
    with downlink_traffic(traffic_ips):
        for t in threads:
            t.start()

        start = time.time()
        try:
            while time.time() - start < duration_s:
                time.sleep(2.0)
                _print_status(states, time.time() - start)
        except KeyboardInterrupt:
            print("\nstopping early (Ctrl-C)")
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=3)

    for s in states:
        if not s.frames:
            print(f"WARNING: {s.receiver_id} ({s.port}) captured 0 frames")
            continue
        path = session_dir / f"{s.receiver_id}.parquet"
        frames_to_dataframe(s.frames).to_parquet(path)
        elapsed = s.frames[-1].host_ts - s.frames[0].host_ts
        rate = len(s.frames) / elapsed if elapsed > 0 else 0.0
        print(
            f"{s.receiver_id}: {len(s.frames)} frames @ {rate:.1f} Hz, "
            f"{s.seq_gaps} seq gaps, {s.parse_errors} parse errors -> {path}"
        )
    return session_dir


def _print_status(states: list[ReceiverState], elapsed: float) -> None:
    parts = [f"{s.receiver_id}:{len(s.frames)}" for s in states]
    print(f"[{elapsed:6.1f}s] frames {' '.join(parts)}", end="\r", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect CSI from ESP32-S3 receivers")
    ap.add_argument(
        "--port", action="append", required=True, metavar="DEV=ID",
        help="serial port and receiver id, e.g. /dev/cu.usbmodem101=rx1 (repeatable)",
    )
    ap.add_argument(
        "--label", required=True,
        # falling is excluded: a whole-session falling label is mostly
        # standing/lying between falls. session_script cues each fall.
        choices=("walking", "sitting", "standing", "lying", "background"),
        help="activity label for this session (record falls with "
        "csihar.session_script, which cues each fall)",
    )
    ap.add_argument("--subject", required=True)
    ap.add_argument("--env", default="room_a")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    ap.add_argument("--out", type=Path, default=Path("datasets/raw"))
    ap.add_argument("--notes", default="")
    add_traffic_argument(ap)
    args = ap.parse_args(argv)

    ports: dict[str, str] = {}
    for spec in args.port:
        dev, sep, rid = spec.partition("=")
        if not sep or not rid:
            ap.error(f"--port needs DEV=ID form, got {spec!r}")
        ports[dev] = rid

    session_dir = run_session(
        ports, args.out, args.label, args.subject, args.env, args.duration,
        args.notes, traffic_ips=args.traffic,
    )
    print(f"\nsession saved: {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
