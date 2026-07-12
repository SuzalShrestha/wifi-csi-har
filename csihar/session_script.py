"""Guided scripted-session runner: one continuous recording, many labels.

Unlike collector.run_session (one label per whole session), this steps an
operator through a timed sequence of activities while recording continuously
across all receivers, then emits a `labels.json` sidecar mapping host-clock
time ranges to activity labels. Downstream preprocessing slices windows out
of the parquet using these ranges instead of a single per-session label.

Usage:
    python -m csihar.session_script \
        --port /dev/cu.usbmodem101=rx1 --port /dev/cu.usbmodem102=rx2 \
        --port /dev/cu.usbmodem103=rx3 \
        --script "background:10,walking:60,sitting:60" \
        --subject sujal --env room_a --out datasets/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .collector import BAUD_RATE, ReceiverState, read_receiver
from .storage import frames_to_dataframe, write_session_metadata

VALID_LABELS = frozenset(
    {"walking", "sitting", "standing", "lying", "falling", "background"}
)

_COUNTDOWN_INTERVAL_S = 5.0
_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class Segment:
    label: str
    duration_s: float


@dataclass(frozen=True)
class TimedSegment:
    label: str
    start_ts: float
    end_ts: float


def parse_script(spec: str) -> tuple[Segment, ...]:
    """Parse "label:seconds,label:seconds,..." into ordered Segments.

    Raises ValueError for unknown labels or non-positive durations.
    """
    segments = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        label, sep, raw_duration = part.partition(":")
        label = label.strip()
        if not sep:
            raise ValueError(f"segment {part!r} must be in label:seconds form")
        if label not in VALID_LABELS:
            raise ValueError(
                f"unknown label {label!r}; valid labels are {sorted(VALID_LABELS)}"
            )
        try:
            duration_s = float(raw_duration)
        except ValueError as exc:
            raise ValueError(
                f"segment {part!r} has a non-numeric duration"
            ) from exc
        if duration_s <= 0:
            raise ValueError(f"segment {part!r} must have duration > 0")
        segments.append(Segment(label=label, duration_s=duration_s))
    if not segments:
        raise ValueError("script must contain at least one segment")
    return tuple(segments)


def segments_to_labels_json(segments_with_times: tuple[TimedSegment, ...]) -> dict:
    """Build the labels.json dict per the shared contract (pure, testable)."""
    return {
        "segments": [
            {
                "label": s.label,
                "start_ts": s.start_ts,
                "end_ts": s.end_ts,
            }
            for s in segments_with_times
        ]
    }


def write_labels_json(
    session_dir: Path, segments_with_times: tuple[TimedSegment, ...]
) -> Path:
    path = session_dir / "labels.json"
    path.write_text(json.dumps(segments_to_labels_json(segments_with_times), indent=2))
    return path


def _wait_segment(duration_s: float, label: str, clock: Callable[[], float]) -> None:
    """Block until `duration_s` of clock() time has elapsed, printing a
    countdown. Uses a poll loop (not a single sleep) so a fake clock that
    jumps forward exits immediately without a real sleep.
    """
    start = clock()
    next_announce = _COUNTDOWN_INTERVAL_S
    while True:
        elapsed = clock() - start
        remaining = duration_s - elapsed
        if remaining <= 0:
            break
        if elapsed >= next_announce or remaining <= _COUNTDOWN_INTERVAL_S:
            print(f"  {label}: {max(remaining, 0):.0f}s remaining", flush=True)
            next_announce += _COUNTDOWN_INTERVAL_S
        time.sleep(min(_POLL_INTERVAL_S, max(remaining, 0)))


def run_scripted_session(
    ports: dict[str, str],
    out_dir: Path,
    script: tuple[Segment, ...],
    subject: str,
    environment: str,
    notes: str = "",
    prompt_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.time,
) -> Path:
    """Record continuously while stepping the operator through `script`.

    Reuses collector.read_receiver/ReceiverState so serial handling is
    identical to a plain session. Writes one parquet per receiver plus
    metadata.json (label="scripted") and labels.json with per-segment
    start/end host-clock timestamps.
    """
    session_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{subject}_scripted"
    session_dir = out_dir / session_name
    write_session_metadata(
        session_dir, label="scripted", subject=subject, environment=environment,
        receivers=ports, notes=notes,
    )

    stop = threading.Event()
    states = [ReceiverState(port=p, receiver_id=r) for p, r in ports.items()]
    threads = [
        threading.Thread(target=read_receiver, args=(s, stop), daemon=True)
        for s in states
    ]
    for t in threads:
        t.start()

    timed_segments: list[TimedSegment] = []
    try:
        for segment in script:
            prompt_fn(
                f"NEXT: {segment.label} for {segment.duration_s:.0f}s — "
                "press Enter when subject is ready"
            )
            start_ts = clock()
            _wait_segment(segment.duration_s, segment.label, clock)
            end_ts = clock()
            timed_segments.append(
                TimedSegment(label=segment.label, start_ts=start_ts, end_ts=end_ts)
            )
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
        print(f"{s.receiver_id}: {len(s.frames)} frames -> {path}")

    write_labels_json(session_dir, tuple(timed_segments))
    return session_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Record a guided scripted CSI session (multi-segment labels)"
    )
    ap.add_argument(
        "--port", action="append", required=True, metavar="DEV=ID",
        help="serial port and receiver id, e.g. /dev/cu.usbmodem101=rx1 (repeatable)",
    )
    ap.add_argument(
        "--script", required=True,
        help='comma-separated "label:seconds" segments, e.g. '
        '"background:10,walking:60,sitting:60"',
    )
    ap.add_argument("--subject", required=True)
    ap.add_argument("--env", default="room_a")
    ap.add_argument("--out", type=Path, default=Path("datasets/raw"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    ports: dict[str, str] = {}
    for spec in args.port:
        dev, sep, rid = spec.partition("=")
        if not sep or not rid:
            ap.error(f"--port needs DEV=ID form, got {spec!r}")
        ports[dev] = rid

    try:
        script = parse_script(args.script)
    except ValueError as exc:
        ap.error(str(exc))
        return 2

    session_dir = run_scripted_session(
        ports, args.out, script, args.subject, args.env, args.notes,
    )
    print(f"\nsession saved: {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
