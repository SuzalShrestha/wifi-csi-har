"""Pre-session hardware check: prove the rig works BEFORE recording anything.

Two bring-up sessions have been lost to failure modes that look like success
until the data is inspected — a board on the wrong USB port (the native port
also emits CSI, at 115200, and is not validated at 100 Hz) and a capture with
no UDP downlink (CSI collapses to <1 Hz). Both are cheap to detect in ten
seconds and expensive to discover after a subject has performed for 25
minutes.

Runs a short live capture through the *same* checks as post-session QA
(``qa.check_dataframe``), plus the cross-receiver checks that only make sense
before a session: every receiver present, all on one channel, all at rate.

    python -m csihar.preflight \
        --port /dev/cu.usbmodem5B5E0807741=rx1 \
        --port /dev/cu.usbmodem5C842982391=rx2 \
        --port /dev/cu.usbmodem5C842990031=rx3 \
        --traffic 192.168.1.97 --traffic 192.168.1.98 --traffic 192.168.1.99

Exit code 0 means the rig is ready to record; anything else means fix it
first. Writes nothing to disk.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .collector import ReceiverState, read_receiver
from .qa import ABSENT, ReceiverReport, check_dataframe
from .storage import frames_to_dataframe
from .traffic import add_traffic_argument, downlink_traffic

__all__ = ["PreflightResult", "evaluate", "run_preflight", "main"]

DEFAULT_DURATION_S = 10.0


@dataclass(frozen=True)
class PreflightResult:
    """Per-receiver QA plus the cross-receiver checks, and the verdict."""

    reports: tuple[ReceiverReport, ...]
    missing: tuple[str, ...]        # receivers that produced no frames at all
    parse_errors: dict[str, int]
    channels: tuple[int, ...]       # distinct channels seen across receivers
    passed: bool

    @property
    def channel_mismatch(self) -> bool:
        """Receivers on different channels cannot be aligned into one window."""
        return len([c for c in self.channels if c != ABSENT]) > 1


def evaluate(
    states: list[ReceiverState],
    expected_ids: tuple[str, ...],
    min_rate_hz: float = 90.0,
    max_parse_errors: int = 10,
) -> PreflightResult:
    """Turn captured receiver states into a verdict. Pure — no I/O."""
    by_id = {s.receiver_id: s for s in states}
    missing = tuple(
        rx for rx in expected_ids if not by_id.get(rx) or not by_id[rx].frames
    )

    reports = []
    for rx in expected_ids:
        state = by_id.get(rx)
        if state is None or not state.frames:
            continue
        reports.append(
            check_dataframe(
                rx, frames_to_dataframe(state.frames), min_rate_hz=min_rate_hz
            )
        )

    channels = tuple(sorted({r.dominant_channel for r in reports}))
    parse_errors = {s.receiver_id: s.parse_errors for s in states}
    channel_mismatch = len([c for c in channels if c != ABSENT]) > 1

    passed = (
        not missing
        and bool(reports)
        and all(r.passed for r in reports)
        and not channel_mismatch
        and all(n <= max_parse_errors for n in parse_errors.values())
    )
    return PreflightResult(
        reports=tuple(reports),
        missing=missing,
        parse_errors=parse_errors,
        channels=channels,
        passed=passed,
    )


def run_preflight(
    ports: dict[str, str],
    duration_s: float = DEFAULT_DURATION_S,
    traffic_ips: list[str] | None = None,
    min_rate_hz: float = 90.0,
) -> PreflightResult:
    """Capture for `duration_s` on every port, then evaluate. Writes nothing."""
    stop = threading.Event()
    states = [ReceiverState(port=p, receiver_id=r) for p, r in ports.items()]
    threads = [
        threading.Thread(target=read_receiver, args=(s, stop), daemon=True)
        for s in states
    ]

    with downlink_traffic(traffic_ips):
        for t in threads:
            t.start()
        try:
            time.sleep(duration_s)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=3)

    return evaluate(states, tuple(ports.values()), min_rate_hz=min_rate_hz)


def format_result(result: PreflightResult) -> str:
    lines = []
    for report in result.reports:
        status = "ok" if report.passed else "FAIL"
        errors = result.parse_errors.get(report.receiver_id, 0)
        lines.append(
            f"{report.receiver_id}: {report.mean_rate_hz:6.1f} Hz  "
            f"rssi {report.rssi_mean:6.1f}  ch {report.dominant_channel}  "
            f"gaps {report.seq_gap_fraction:.3f}  parse_err {errors}  -> {status}"
        )
    for rx in result.missing:
        lines.append(f"{rx}: NO FRAMES -> FAIL")

    lines.append("")
    if result.missing:
        lines.append(
            "A receiver produced nothing. Check it is on the UART USB-C port "
            "(not the native port), powered, and joined to the router."
        )
    if any(r.mean_rate_hz < 90.0 for r in result.reports):
        lines.append(
            "Low rate usually means no UDP downlink — pass --traffic with "
            "every receiver's current IP. It can also mean 115200 baud (wrong "
            "USB port) or a saturated hub."
        )
    if result.channel_mismatch:
        lines.append(
            f"Receivers are on different channels {result.channels} — they "
            "cannot be aligned into one window. Rejoin them to the same AP."
        )
    lines.append("READY TO RECORD" if result.passed else "NOT READY — fix the above")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m csihar.preflight",
        description="Verify all receivers are streaming before a session.",
    )
    ap.add_argument(
        "--port", action="append", required=True, metavar="DEV=ID",
        help="serial port and receiver id, e.g. /dev/cu.usbmodem101=rx1 "
        "(repeatable)",
    )
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    ap.add_argument("--min-rate-hz", type=float, default=90.0)
    add_traffic_argument(ap)
    args = ap.parse_args(argv)

    ports: dict[str, str] = {}
    for spec in args.port:
        dev, sep, rid = spec.partition("=")
        if not sep or not rid:
            ap.error(f"--port needs DEV=ID form, got {spec!r}")
        ports[dev] = rid

    if not args.traffic:
        print(
            "WARNING: no --traffic IPs given; CSI will read <1 Hz against a "
            "real router and this check will fail for the wrong reason.",
            file=sys.stderr,
        )

    print(f"capturing {args.duration:.0f}s on {len(ports)} receivers...")
    result = run_preflight(
        ports, args.duration, args.traffic, min_rate_hz=args.min_rate_hz
    )
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
