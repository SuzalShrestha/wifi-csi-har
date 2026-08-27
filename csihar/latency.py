"""End-to-end latency benchmark for the streaming pipeline (Phase 1 item 6).

Measures *compute cost*, which is independent of model accuracy — so it runs
against an untrained model while no checkpoint exists yet, and against a real
checkpoint later for the Phase 5 item 6 numbers. Stages timed:

    parse      per CSI_DATA line: serial text -> CsiFrame
    ingest     per frame: append_frame (copy-on-write buffer append + prune)
    window     per tick: subcarrier select, hampel, detrend, resample, stack
    inference  per tick: normalize, forward pass, softmax

``window`` + ``inference`` are the tick path — the compute that must finish
inside one hop or the stream falls behind. ``RealtimeEngine`` adds only
majority-vote smoothing over ``vote_k`` (<= 5) entries on top of these, which
is not timed separately because it is a bounded Counter over a handful of
strings.

The dominant *observed* delay is structural rather than computational: a label
describes a window that closed ``window_s`` ago, so a correct end-to-end
figure is that budget plus the tick compute. Both are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .dataset import LABEL_NAMES
from .models import build_model
from .models.inference import ModelBundle, load_checkpoint, predict
from .parser import parse_line
from .preprocessing import PreprocessConfig
from .realtime import StreamBuffers, append_frame, make_realtime_window
from .realtime_sources import Frame, replay_source, simulate_source
from .simulate import generate_lines

__all__ = [
    "StageTiming", "LatencyReport",
    "summarize", "untrained_bundle", "measure", "main",
]

STAGE_NAMES: tuple[str, ...] = ("parse", "ingest", "window", "inference")


@dataclass(frozen=True)
class StageTiming:
    """Timing distribution for one pipeline stage, in milliseconds."""

    name: str
    n_calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True)
class LatencyReport:
    """Per-stage timings plus the real-time budgets they are judged against."""

    stages: tuple[StageTiming, ...]
    n_frames: int
    n_ticks: int
    n_receivers: int
    window_s: float
    hop_s: float
    frame_budget_ms: float      # wall time available per incoming frame
    tick_budget_ms: float       # wall time available per tick (= hop_s)
    tick_compute_p95_ms: float  # window + inference at the 95th percentile
    end_to_end_p95_ms: float    # buffering delay + tick compute
    realtime_capable: bool

    def by_name(self, name: str) -> StageTiming:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


def summarize(name: str, samples_s: Iterable[float]) -> StageTiming:
    """Percentile summary of per-call durations (seconds in, milliseconds out)."""
    ms = np.asarray(list(samples_s), dtype=np.float64) * 1000.0
    if ms.size == 0:
        return StageTiming(name, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    return StageTiming(
        name=name,
        n_calls=int(ms.size),
        mean_ms=float(ms.mean()),
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        max_ms=float(ms.max()),
    )


def untrained_bundle(
    model_name: str,
    n_rx: int,
    n_time: int,
    n_subcarriers: int,
    label_names: tuple[str, ...] = LABEL_NAMES,
) -> ModelBundle:
    """A randomly-initialised bundle with the real architecture and shapes.

    Latency depends on the architecture and the window shape, not on the
    weights, so this gives a truthful compute measurement before any model has
    been trained. It must never be used for anything that reports accuracy.
    """
    config = {
        "n_rx": n_rx,
        "n_time": n_time,
        "n_subcarriers": n_subcarriers,
        "n_classes": len(label_names),
        "chunk_len": 25,
    }
    model = build_model(
        model_name,
        n_rx=n_rx,
        n_time=n_time,
        n_subcarriers=n_subcarriers,
        n_classes=len(label_names),
    )
    model.eval()
    return ModelBundle(
        model=model,
        model_name=model_name,
        label_names=tuple(label_names),
        norm_mean=np.zeros((n_rx, n_subcarriers), dtype=np.float32),
        norm_std=np.ones((n_rx, n_subcarriers), dtype=np.float32),
        config=config,
    )


def measure_parse(n_lines: int = 2000, seed: int = 0) -> list[float]:
    """Time parse_line over byte-exact firmware-format lines from the simulator."""
    lines = generate_lines("walking", duration_s=n_lines / 100.0, seed=seed)
    durations: list[float] = []
    for host_ts, line in lines:
        start = time.perf_counter()
        parse_line(line, host_ts)
        durations.append(time.perf_counter() - start)
    return durations


def measure(
    source: Iterator[Frame],
    *,
    pre_cfg: PreprocessConfig,
    bundle: ModelBundle | None = None,
    model_name: str = "cnn",
    hop_s: float | None = None,
    max_ticks: int | None = None,
) -> LatencyReport:
    """Run a frame source through the tick path, timing each stage.

    ``bundle`` defaults to an untrained model matched to the first window's
    shape. The ``parse`` stage is always timed against simulator-generated
    lines (byte-exact firmware format) because recorded sessions store parsed
    frames, not the raw serial text. Returns a report; performs no I/O.
    """
    hop_s = pre_cfg.hop_s if hop_s is None else hop_s
    buffers: StreamBuffers = {}
    ingest_s: list[float] = []
    window_s_times: list[float] = []
    inference_s: list[float] = []
    tick_compute_s: list[float] = []

    rx_seen: set[str] = set()
    next_tick: float | None = None
    n_frames = 0
    done = False

    for rx_id, host_ts, amplitude in source:
        if done:
            break
        start = time.perf_counter()
        buffers = append_frame(
            buffers, rx_id, host_ts, amplitude, window_s=pre_cfg.window_s
        )
        ingest_s.append(time.perf_counter() - start)
        rx_seen.add(rx_id)
        n_frames += 1

        if next_tick is None:
            next_tick = host_ts + pre_cfg.window_s
        while host_ts >= next_tick:
            start = time.perf_counter()
            window = make_realtime_window(
                buffers, tuple(rx_seen), pre_cfg, next_tick
            )
            window_elapsed = time.perf_counter() - start
            next_tick += hop_s
            if window is None:
                continue
            window_s_times.append(window_elapsed)

            if bundle is None:
                n_rx, n_time, n_sub = window.shape
                bundle = untrained_bundle(model_name, n_rx, n_time, n_sub)

            start = time.perf_counter()
            predict(bundle, window)
            inference_elapsed = time.perf_counter() - start
            inference_s.append(inference_elapsed)
            tick_compute_s.append(window_elapsed + inference_elapsed)

            if max_ticks is not None and len(tick_compute_s) >= max_ticks:
                done = True
                break

    n_receivers = max(len(rx_seen), 1)
    # Frames arrive at fs per receiver, so the aggregate budget is shared.
    frame_budget_ms = 1000.0 / (pre_cfg.fs * n_receivers)
    tick_budget_ms = hop_s * 1000.0

    stages = (
        summarize("parse", measure_parse()),
        summarize("ingest", ingest_s),
        summarize("window", window_s_times),
        summarize("inference", inference_s),
    )
    tick_compute_p95 = summarize("tick", tick_compute_s).p95_ms
    ingest_p95 = stages[1].p95_ms
    parse_p95 = stages[0].p95_ms

    realtime_capable = bool(
        tick_compute_p95 <= tick_budget_ms
        and (parse_p95 + ingest_p95) <= frame_budget_ms
    )

    return LatencyReport(
        stages=stages,
        n_frames=n_frames,
        n_ticks=len(tick_compute_s),
        n_receivers=n_receivers,
        window_s=pre_cfg.window_s,
        hop_s=hop_s,
        frame_budget_ms=frame_budget_ms,
        tick_budget_ms=tick_budget_ms,
        tick_compute_p95_ms=tick_compute_p95,
        end_to_end_p95_ms=pre_cfg.window_s * 1000.0 + tick_compute_p95,
        realtime_capable=realtime_capable,
    )


def format_report(report: LatencyReport) -> str:
    header = (
        f"{'stage':<10} {'n':>7} {'mean_ms':>9} {'p50_ms':>9} "
        f"{'p95_ms':>9} {'max_ms':>9}"
    )
    rows = [
        f"{s.name:<10} {s.n_calls:>7} {s.mean_ms:>9.3f} {s.p50_ms:>9.3f} "
        f"{s.p95_ms:>9.3f} {s.max_ms:>9.3f}"
        for s in report.stages
    ]
    verdict = "REAL-TIME OK" if report.realtime_capable else "TOO SLOW"
    return "\n".join(
        [
            f"frames={report.n_frames} ticks={report.n_ticks} "
            f"receivers={report.n_receivers} "
            f"window_s={report.window_s} hop_s={report.hop_s}",
            header,
            "-" * len(header),
            *rows,
            "",
            f"per-frame budget : {report.frame_budget_ms:.3f} ms "
            f"({report.n_receivers} rx sharing the host)",
            f"per-tick budget  : {report.tick_budget_ms:.1f} ms (one hop)",
            f"tick compute p95 : {report.tick_compute_p95_ms:.1f} ms",
            f"end-to-end p95   : {report.end_to_end_p95_ms:.1f} ms "
            f"({report.window_s * 1000:.0f} ms buffering + compute)",
            f"verdict          : {verdict}",
        ]
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m csihar.latency",
        description="Measure per-stage latency of the streaming CSI pipeline.",
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--simulate", metavar="ACTIVITY", default="walking",
        help="benchmark on synthetic frames (default: walking)",
    )
    src.add_argument(
        "--replay", type=Path, metavar="SESSION_DIR",
        help="benchmark on a recorded session instead of simulated frames",
    )
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds of simulated stream (ignored with --replay)")
    ap.add_argument("--receivers", type=int, default=3,
                    help="number of simulated receivers")
    ap.add_argument("--model", default="cnn", choices=("cnn", "cnn_lstm"))
    ap.add_argument("--checkpoint", type=Path,
                    help="use a trained checkpoint instead of an untrained model")
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--hop-s", type=float, default=1.5)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pre_cfg = PreprocessConfig(window_s=args.window_s, hop_s=args.hop_s)

    if args.replay is not None:
        source = replay_source(args.replay, speed=0.0, sleep_fn=lambda _: None)
    else:
        rx_ids = tuple(f"rx{i + 1}" for i in range(args.receivers))
        source = simulate_source(args.simulate, args.duration, rx_ids=rx_ids)

    bundle = load_checkpoint(args.checkpoint) if args.checkpoint else None
    report = measure(
        source,
        pre_cfg=pre_cfg,
        bundle=bundle,
        model_name=args.model,
        hop_s=args.hop_s,
        max_ticks=args.max_ticks,
    )

    if bundle is None:
        print(f"[untrained {args.model} — compute only, says nothing about accuracy]")
    print(format_report(report))

    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2))
        print(f"wrote {args.json}")

    return 0 if report.realtime_capable else 1


if __name__ == "__main__":
    sys.exit(main())
