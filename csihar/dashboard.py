"""Phase 5 demo dashboard: FastAPI + WebSocket wrapper around RealtimeEngine.

Serves a single offline HTML page (``csihar/static/dashboard.html``) plus a
``/ws`` endpoint that streams live CSI heatmap samples and smoothed
predictions produced by a background thread draining a ``Frame`` source
through the engine's ``feed``/``tick`` calls (mirrors ``_run_stream`` in
``csihar/realtime.py``).

``fastapi``/``uvicorn`` are only ever imported here, guarded by a
try/except, so ``import csihar`` keeps working without the ``demo`` extra
installed — ``create_app``/``main`` simply raise ``ModuleNotFoundError`` if
called without it. (They must be resolvable at *module* import time, not
just lazily inside a function, because FastAPI resolves the ``WebSocket``
type annotation via the endpoint function's ``__globals__`` — with
``from __future__ import annotations`` active, a name only bound inside a
nested function's local scope is invisible to that lookup and the parameter
silently degrades into a query parameter instead of the websocket.)

``_Broadcaster`` is the one stateful object here (a fan-out registry of
per-client queues guarded by a lock) — the same documented exception to the
immutability rule that ``RealtimeEngine`` itself uses.
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from .preprocessing.subcarriers import usable_lltf_indices
from .realtime_sources import Frame, live_source, replay_source, simulate_source

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:  # demo extra not installed
    FastAPI = None  # type: ignore[assignment,misc]
    WebSocket = None  # type: ignore[assignment,misc]
    WebSocketDisconnect = None  # type: ignore[assignment,misc]
    HTMLResponse = None  # type: ignore[assignment,misc]

__all__ = ["DashboardConfig", "create_app", "main"]

_STATIC_DIR = Path(__file__).parent / "static"


class Engine(Protocol):
    """Duck-typed engine interface — anything with feed/tick works (tests
    inject a stub; production uses ``csihar.realtime.RealtimeEngine``)."""

    def feed(self, rx_id: str, host_ts: float, amplitude: np.ndarray) -> None: ...
    def tick(self, t_end: float) -> Any: ...


@dataclass(frozen=True)
class DashboardConfig:
    hop_s: float = 1.5
    heatmap_hz: float = 10.0


class _Broadcaster:
    """Fan-out registry of per-client queues (stateful by design, guarded by
    a lock) — mirrors the documented exception ``RealtimeEngine`` takes for
    its own streaming state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[int, "queue.Queue[Any]"] = {}
        self._next_id = 0

    def register(self) -> tuple[int, "queue.Queue[Any]"]:
        q: "queue.Queue[Any]" = queue.Queue()
        with self._lock:
            client_id = self._next_id
            self._next_id += 1
            self._clients[client_id] = q
        return client_id, q

    def unregister(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)

    def publish(self, message: Any) -> None:
        with self._lock:
            targets = list(self._clients.values())
        for q in targets:
            q.put(message)


def _csi_message(rx_id: str, host_ts: float, amplitude: np.ndarray) -> dict:
    usable = usable_lltf_indices()
    amp = np.asarray(amplitude, dtype=np.float64)
    subsampled = amp[usable] if amp.shape[0] > int(usable.max()) else amp
    return {
        "type": "csi",
        "rx_id": rx_id,
        "ts": float(host_ts),
        "amplitude": [round(float(v), 1) for v in subsampled.tolist()],
    }


def _prediction_message(pred: Any) -> dict:
    return {
        "type": "prediction",
        "ts": float(pred.ts),
        "raw_label": str(pred.raw_label),
        "smoothed_label": str(pred.smoothed_label),
        "confidence": float(pred.confidence),
        "fall": pred.smoothed_label == "falling",
    }


def _drain_source(
    source: Iterator[Frame],
    engine: Engine,
    broadcaster: _Broadcaster,
    stop_event: threading.Event,
    hop_s: float,
    heatmap_hz: float,
) -> None:
    """Background worker: feed the engine, throttle heatmap frames, tick on
    schedule, and broadcast results. Mirrors ``_run_stream`` in
    ``csihar/realtime.py`` for the tick-scheduling pattern."""
    next_tick: float | None = None
    min_csi_gap = (1.0 / heatmap_hz) if heatmap_hz > 0 else 0.0
    last_csi_ts: dict[str, float] = {}

    for rx_id, host_ts, amplitude in source:
        if stop_event.is_set():
            return
        engine.feed(rx_id, host_ts, amplitude)

        last = last_csi_ts.get(rx_id)
        if last is None or (host_ts - last) >= min_csi_gap:
            last_csi_ts[rx_id] = host_ts
            broadcaster.publish(_csi_message(rx_id, host_ts, amplitude))

        if next_tick is None:
            next_tick = host_ts + hop_s
        while host_ts >= next_tick:
            pred = engine.tick(next_tick)
            if pred is not None:
                broadcaster.publish(_prediction_message(pred))
            next_tick += hop_s

    broadcaster.publish({"type": "end"})


def create_app(
    engine: Engine,
    source_factory: Callable[[], Iterator[Frame]],
    *,
    hop_s: float = 1.5,
    heatmap_hz: float = 10.0,
):
    """Build the FastAPI app. Requires the ``demo`` extra (fastapi/uvicorn)
    to be installed; raises ``ModuleNotFoundError`` otherwise."""
    if FastAPI is None:
        raise ModuleNotFoundError(
            "fastapi is required for csihar.dashboard; "
            'install it with `pip install -e ".[demo]"`'
        )

    broadcaster = _Broadcaster()
    stop_event = threading.Event()

    @contextlib.asynccontextmanager
    async def _lifespan(_: "FastAPI"):
        source = source_factory()
        worker = threading.Thread(
            target=_drain_source,
            args=(source, engine, broadcaster, stop_event, hop_s, heatmap_hz),
            daemon=True,
        )
        worker.start()
        try:
            yield
        finally:
            stop_event.set()
            worker.join(timeout=3)

    app = FastAPI(title="WiFi CSI HAR Dashboard", lifespan=_lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        from starlette.websockets import WebSocketState

        await websocket.accept()
        client_id, q = broadcaster.register()
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                message = await _get_from_queue(q)
                if message is None:
                    continue  # poll timeout: recheck connection state
                await websocket.send_json(message)
                if message.get("type") == "end":
                    # Source exhausted: nothing further will ever be queued,
                    # so stop reading rather than block forever on q.get()
                    # (which would also stall a clean client disconnect).
                    break
        except WebSocketDisconnect:
            pass
        finally:
            broadcaster.unregister(client_id)

    return app


async def _get_from_queue(
    q: "queue.Queue[Any]", poll_timeout: float = 0.5
) -> Any:
    """Blocking queue.get(timeout=...) run off the event loop thread pool.

    Returns None on a poll timeout (no message ready) so the caller can
    periodically recheck the websocket's connection state rather than
    blocking forever on a queue that a disconnected/finished stream will
    never fill again.
    """
    import anyio

    def _get() -> Any:
        try:
            return q.get(timeout=poll_timeout)
        except queue.Empty:
            return None

    return await anyio.to_thread.run_sync(_get)


# ------------------------------------------------------------------------ CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m csihar.dashboard",
        description="Live CSI HAR dashboard: heatmap + predicted activity "
        "over a replayed, simulated, or live source.",
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from .models.inference import load_checkpoint
    from .preprocessing import PreprocessConfig
    from .realtime import RealtimeEngine, SmootherConfig

    args = _build_arg_parser().parse_args(argv)
    bundle = load_checkpoint(args.checkpoint)
    pre_cfg = PreprocessConfig()
    smoother = SmootherConfig()
    engine = RealtimeEngine(bundle, pre_cfg, smoother)

    if args.replay is not None:
        def source_factory() -> Iterator[Frame]:
            return replay_source(args.replay, speed=args.speed)
    elif args.simulate is not None:
        def source_factory() -> Iterator[Frame]:
            return simulate_source(args.simulate, duration_s=args.duration)
    elif args.live:
        ports: dict[str, str] = {}
        for spec in args.live:
            dev, sep, rid = spec.partition("=")
            if not sep or not rid:
                raise SystemExit(f"--live needs DEV=ID form, got {spec!r}")
            ports[dev] = rid

        def source_factory() -> Iterator[Frame]:
            return live_source(ports)
    else:
        raise SystemExit("one of --replay, --simulate, --live is required")

    app = create_app(engine, source_factory, hop_s=args.hop)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
