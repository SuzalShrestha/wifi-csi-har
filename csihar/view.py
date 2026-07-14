"""CSI amplitude heatmap viewer.

Replay mode (works today, no hardware):
    python -m csihar.view --replay datasets/raw/<session>/rx1.parquet
    python -m csihar.view --simulate walking

Live mode (Phase 1 milestone M1):
    python -m csihar.view --live /dev/cu.usbmodem101
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .parser import parse_line, CsiParseError
from .preprocessing import usable_lltf_indices


def show_static(amplitudes: np.ndarray, title: str, out: Path | None = None) -> None:
    usable = amplitudes[:, usable_lltf_indices()]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        usable.T, aspect="auto", origin="lower", cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("packet index (time →)")
    ax.set_ylabel("subcarrier (52 usable LLTF)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|CSI| amplitude")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        print(f"saved {out}")
    else:
        plt.show()


def replay(path: Path, out: Path | None) -> None:
    from .storage import load_amplitudes

    _, amplitudes = load_amplitudes(path)
    show_static(amplitudes, f"CSI amplitude — {path.name}", out)


def simulate(activity: str, out: Path | None) -> None:
    from .simulate import generate_lines

    amps = []
    for ts, line in generate_lines(activity, duration_s=10.0):
        frame = parse_line(line, ts)
        if frame is not None:
            amps.append(frame.amplitude)
    show_static(np.stack(amps), f"CSI amplitude — simulated {activity}", out)


def live(port: str, baud: int = 921600, history: int = 500) -> None:
    import serial

    buffer: deque[np.ndarray] = deque(maxlen=history)
    usable = usable_lltf_indices()

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        np.zeros((len(usable), history)), aspect="auto", origin="lower",
        cmap="viridis", vmin=0, vmax=40,
    )
    ax.set_title(f"live CSI — {port}")
    ax.set_xlabel("packets (rolling)")
    ax.set_ylabel("subcarrier")

    with serial.Serial(port, baud, timeout=1) as conn:
        conn.reset_input_buffer()
        last_draw = 0.0
        while plt.fignum_exists(fig.number):
            raw = conn.readline()
            if not raw:
                continue
            try:
                frame = parse_line(raw.decode("utf-8", errors="replace"), time.time())
            except CsiParseError:
                continue
            if frame is None:
                continue
            buffer.append(frame.amplitude[usable])
            now = time.time()
            if now - last_draw > 0.1 and buffer:
                data = np.stack(buffer).T
                padded = np.zeros((len(usable), history))
                padded[:, -data.shape[1]:] = data
                image.set_data(padded)
                image.set_clim(0, max(1.0, float(data.max())))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_draw = now


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CSI heatmap viewer")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay", type=Path, help="parquet file to plot")
    group.add_argument("--simulate", metavar="ACTIVITY", help="synthetic data")
    group.add_argument("--live", metavar="PORT", help="serial port for live view")
    ap.add_argument("--out", type=Path, help="save PNG instead of showing")
    ap.add_argument(
        "--traffic", metavar="IP",
        help="board IP to flood with UDP while viewing (needed with a real "
        "router or CSI drops to <1 Hz; see csihar/traffic.py)",
    )
    args = ap.parse_args(argv)

    traffic = None
    if args.traffic:
        from .traffic import TrafficGenerator

        traffic = TrafficGenerator([args.traffic]).start()
    try:
        if args.replay:
            replay(args.replay, args.out)
        elif args.simulate:
            simulate(args.simulate, args.out)
        else:
            live(args.live)
    finally:
        if traffic is not None:
            traffic.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
