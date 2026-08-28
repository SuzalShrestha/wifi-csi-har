"""Phase 5 streaming inference: live/replayed CSI frames -> smoothed labels.

Pipeline per receiver, matching ``preprocessing.pipeline.preprocess_stream``
exactly: usable subcarrier selection -> hampel -> detrend_moving_mean ->
resample_uniform. Windows across receivers are stacked (n_rx, T, S) and fed
to ``models.inference.predict``. Raw per-tick predictions are smoothed by a
simple majority vote over a short rolling history.

``RealtimeEngine`` is the one stateful object in this module (a streaming
accumulator, documented exception to the immutability rule); everything else
(``smooth_predictions``, ``append_frame``, ``make_realtime_window``) is a pure
function over immutable inputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .models.inference import ModelBundle, load_checkpoint, predict
from .preprocessing import PreprocessConfig, detrend_moving_mean, hampel, resample_uniform
from .preprocessing.subcarriers import usable_lltf_indices
from .realtime_sources import Frame, live_source, replay_source, simulate_source
from .traffic import add_traffic_argument, downlink_traffic

__all__ = [
    "SmootherConfig", "smooth_predictions",
    "StreamBuffers", "append_frame", "make_realtime_window",
    "Prediction", "RealtimeEngine",
    "Frame", "replay_source", "simulate_source", "live_source",
    "main",
]

# ------------------------------------------------------------- smoothing


@dataclass(frozen=True)
class SmootherConfig:
    """Rolling majority-vote smoothing over the last ``vote_k`` predictions."""

    vote_k: int = 5
    min_confidence: float = 0.6


def smooth_predictions(
    history: tuple[tuple[str, float], ...], cfg: SmootherConfig
) -> str:
    """Majority label over the last ``vote_k`` (raw_label, confidence) pairs.

    Entries below ``cfg.min_confidence`` ABSTAIN: they are dropped from the
    vote rather than voting for "unknown". Letting them vote as a label meant
    two uncertain windows could veto three confident ones, and measured on the
    pilot session that made the smoother strictly worse than no smoothing at
    all - same 29 label changes as the raw stream, but accuracy 0.856 vs
    0.918, with 10% of outputs "unknown". Abstaining gives 0.905 and 19
    changes. Only when NO recent entry is confident does this say "unknown".

    Ties break toward the most recent tied label. Empty history -> "unknown".
    """
    if not history:
        return "unknown"
    recent = history[-cfg.vote_k :]
    votes = [label for label, conf in recent if conf >= cfg.min_confidence]
    if not votes:
        return "unknown"
    counts = Counter(votes)
    best_count = max(counts.values())
    tied = {label for label, c in counts.items() if c == best_count}
    for label in reversed(votes):
        if label in tied:
            return label
    return "unknown"  # unreachable, kept for type-checker satisfaction


# ------------------------------------------------------- buffering + windows

# rx_id -> list of (host_ts, raw amplitude (64,)) samples, oldest first.
StreamBuffers = dict[str, list[tuple[float, np.ndarray]]]


def append_frame(
    buffers: StreamBuffers,
    rx_id: str,
    host_ts: float,
    amplitude: np.ndarray,
    *,
    window_s: float = 3.0,
    now: float | None = None,
) -> StreamBuffers:
    """Return a NEW buffers dict with one sample appended and old ones pruned.

    Only the touched receiver's list is copied (copy-on-write); other
    receivers' lists are shared by reference with the input dict, so this is
    O(len(touched_list)) per call, not O(total buffered samples).
    """
    new_buffers = dict(buffers)
    existing = list(new_buffers.get(rx_id, ()))
    existing.append((float(host_ts), np.asarray(amplitude, dtype=np.float32)))
    cutoff = (now if now is not None else host_ts) - (window_s + 2.0)
    existing = [(ts, amp) for ts, amp in existing if ts >= cutoff]
    new_buffers[rx_id] = existing
    return new_buffers


def _slice_window(
    samples: list[tuple[float, np.ndarray]], t_end: float, window_s: float
) -> tuple[np.ndarray, np.ndarray] | None:
    lo = t_end - window_s - 0.5
    in_range = [(ts, amp) for ts, amp in samples if lo <= ts <= t_end]
    if len(in_range) < 2:
        return None
    min_ts = min(ts for ts, _ in in_range)
    if min_ts > t_end - window_s + 0.1:
        return None
    ts_arr = np.array([ts for ts, _ in in_range], dtype=np.float64)
    amp_arr = np.stack([amp for _, amp in in_range]).astype(np.float64)
    return ts_arr, amp_arr


def make_realtime_window(
    buffers: StreamBuffers,
    rx_ids: tuple[str, ...],
    cfg: PreprocessConfig,
    t_end: float,
) -> np.ndarray | None:
    """Assemble one (n_rx, T, S) window across receivers, or None if unready.

    Applies, per receiver and IN THIS ORDER (matches
    ``preprocessing.pipeline.preprocess_stream``): usable subcarrier
    selection -> hampel -> detrend_moving_mean -> resample_uniform. Any
    receiver failing coverage/loss checks aborts the whole window.
    """
    usable = usable_lltf_indices()
    per_rx: list[np.ndarray] = []
    for rx in sorted(rx_ids):
        samples = buffers.get(rx, [])
        sliced = _slice_window(samples, t_end, cfg.window_s)
        if sliced is None:
            return None
        ts_arr, amp_arr = sliced
        if amp_arr.shape[1] <= int(usable.max()):
            return None
        x = amp_arr[:, usable]
        x = hampel(x, window=cfg.hampel_window, n_sigmas=cfg.hampel_sigmas)
        x = detrend_moving_mean(x, window=cfg.detrend_window)
        resampled, loss = resample_uniform(
            ts_arr, x, fs=cfg.fs, t_start=t_end - cfg.window_s, t_end=t_end
        )
        if loss > cfg.max_loss:
            return None
        per_rx.append(resampled.astype(np.float32))
    return np.stack(per_rx).astype(np.float32)


# --------------------------------------------------------------------- engine


@dataclass(frozen=True)
class Prediction:
    ts: float
    raw_label: str
    confidence: float
    smoothed_label: str
    # A fall is a transient (1-2 windows): it can lose every majority vote,
    # so the safety-critical alert bypasses smoothing on a confident raw hit.
    fall_alert: bool = False


class RealtimeEngine:
    """Streaming accumulator: buffers + prediction history (stateful by design).

    The pure building blocks above (``append_frame``, ``make_realtime_window``,
    ``smooth_predictions``) stay pure and independently testable; this class
    just wires them together and holds state across ``feed``/``tick`` calls.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        pre_cfg: PreprocessConfig,
        smoother: SmootherConfig,
    ) -> None:
        self.bundle = bundle
        self.pre_cfg = pre_cfg
        self.smoother = smoother
        self._buffers: StreamBuffers = {}
        self._history: list[tuple[str, float]] = []

    def feed(self, rx_id: str, host_ts: float, amplitude: np.ndarray) -> None:
        self._buffers = append_frame(
            self._buffers, rx_id, host_ts, amplitude,
            window_s=self.pre_cfg.window_s,
        )

    def tick(self, t_end: float) -> Prediction | None:
        rx_ids = tuple(self._buffers.keys())
        if not rx_ids:
            return None
        window = make_realtime_window(self._buffers, rx_ids, self.pre_cfg, t_end)
        if window is None:
            return None
        labels, confidences = predict(self.bundle, window)
        raw_label = labels[0]
        confidence = float(confidences[0].max())
        self._history.append((raw_label, confidence))
        del self._history[: -self.smoother.vote_k]  # bound memory on long runs
        smoothed = smooth_predictions(tuple(self._history), self.smoother)
        fall_alert = smoothed == "falling" or (
            raw_label == "falling" and confidence >= self.smoother.min_confidence
        )
        return Prediction(
            ts=t_end, raw_label=raw_label, confidence=confidence,
            smoothed_label=smoothed, fall_alert=fall_alert,
        )


# ------------------------------------------------------------------------ CLI
# Sources (replay_source, simulate_source, live_source) live in
# realtime_sources.py to keep this module focused and under the line budget.


def _run_stream(
    source: Iterator[Frame],
    engine: RealtimeEngine,
    hop_s: float,
    csv_writer: csv.writer | None,
) -> None:
    next_tick: float | None = None
    for rx_id, host_ts, amplitude in source:
        engine.feed(rx_id, host_ts, amplitude)
        if next_tick is None:
            next_tick = host_ts + hop_s
        while host_ts >= next_tick:
            pred = engine.tick(next_tick)
            if pred is not None:
                print(
                    f"{pred.ts:.2f} {pred.raw_label}->{pred.smoothed_label} "
                    f"({pred.confidence:.2f})"
                )
                if csv_writer is not None:
                    csv_writer.writerow(
                        [pred.ts, pred.raw_label, pred.confidence, pred.smoothed_label]
                    )
            next_tick += hop_s


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m csihar.realtime",
        description="Streaming CSI HAR inference over a replayed, simulated, "
        "or live source.",
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--replay", type=Path, default=None, metavar="SESSION_DIR")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--simulate", default=None, metavar="ACTIVITY")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument(
        "--live", action="append", default=[], metavar="DEV=ID",
        help="serial port and receiver id, e.g. /dev/cu.usbmodem101=rx1",
    )
    ap.add_argument("--hop", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=None, metavar="predictions.csv")
    add_traffic_argument(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    bundle = load_checkpoint(args.checkpoint)
    # Window length must match training: the CNN accepts any T (adaptive
    # pooling), so a mismatched window would be silently mis-scored.
    pre_cfg = PreprocessConfig(
        window_s=int(bundle.config["n_time"]) / PreprocessConfig.fs
    )
    smoother = SmootherConfig()
    engine = RealtimeEngine(bundle, pre_cfg, smoother)

    if args.replay is not None:
        source = replay_source(args.replay, speed=args.speed)
    elif args.simulate is not None:
        source = simulate_source(args.simulate, duration_s=args.duration)
    elif args.live:
        ports: dict[str, str] = {}
        for spec in args.live:
            dev, sep, rid = spec.partition("=")
            if not sep or not rid:
                raise SystemExit(f"--live needs DEV=ID form, got {spec!r}")
            ports[dev] = rid
        source = live_source(ports)
    else:
        raise SystemExit("one of --replay, --simulate, --live is required")

    if args.out is not None:
        fh = open(args.out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["ts", "raw_label", "confidence", "smoothed_label"])
    else:
        fh, writer = None, None

    try:
        with downlink_traffic(args.traffic):
            _run_stream(source, engine, args.hop, writer)
    finally:
        if fh is not None:
            fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
