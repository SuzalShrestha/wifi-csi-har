"""UDP downlink traffic generator for CSI capture.

The router only transmits at OFDM/HT rates — the ones whose LLTF the
ESP32-S3 CSI engine can use — when there is sustained downlink data.
Sparse small frames (the firmware's own ping replies, beacons) go out at
DSSS/CCK rates and produce no CSI at all. Measured on real hardware
(2026-07-13): firmware ping alone yields <1 Hz CSI; 100 pkt/s of 200-byte
UDP to the board's IP yields a steady 100 Hz.

Run standalone:
    python -m csihar.traffic 192.168.1.79 192.168.1.80 --rate 100

or pass ``--traffic <ip>`` to the collector / viewer, which embed a
TrafficGenerator for the session duration.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time

DEFAULT_RATE_HZ = 100.0
DEFAULT_PAYLOAD_BYTES = 200
DISCARD_PORT = 9  # nothing listens; the radio transmission is the point


class TrafficGenerator:
    """Background thread sending paced UDP packets to each target IP."""

    def __init__(
        self,
        targets: list[str],
        rate_hz: float = DEFAULT_RATE_HZ,
        payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
        port: int = DISCARD_PORT,
    ) -> None:
        if not targets:
            raise ValueError("need at least one target IP")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.targets = list(targets)
        self.interval_s = 1.0 / rate_hz
        self.payload = b"\x00" * payload_bytes
        self.port = port
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            for ip in self.targets:
                try:
                    sock.sendto(self.payload, (ip, self.port))
                    self.sent += 1
                except OSError:
                    pass  # transient (e.g. ARP miss, network drop); keep pacing
            next_tick += self.interval_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind; don't burst to catch up
        sock.close()

    def start(self) -> "TrafficGenerator":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", help="receiver IP addresses")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="pkt/s per target")
    ap.add_argument("--payload", type=int, default=DEFAULT_PAYLOAD_BYTES, help="bytes")
    args = ap.parse_args(argv)

    gen = TrafficGenerator(args.targets, rate_hz=args.rate, payload_bytes=args.payload).start()
    print(f"sending {args.rate:g} pkt/s x {len(args.targets)} target(s); Ctrl-C to stop")
    try:
        while True:
            time.sleep(5)
            print(f"  sent {gen.sent} packets", end="\r", flush=True)
    except KeyboardInterrupt:
        gen.stop()
        print(f"\nstopped after {gen.sent} packets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
