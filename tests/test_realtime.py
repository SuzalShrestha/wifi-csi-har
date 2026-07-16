import numpy as np
import pytest

from csihar.dataset import LABEL_NAMES, assemble_dataset
from csihar.models.inference import load_checkpoint, predict
from csihar.parser import parse_line
from csihar.preprocessing import PreprocessConfig
from csihar.realtime import (
    Prediction,
    RealtimeEngine,
    SmootherConfig,
    append_frame,
    make_realtime_window,
    smooth_predictions,
)
from csihar.realtime_sources import replay_source
from csihar.simulate import generate_lines, write_session
from csihar.train import TrainConfig, train_model

# ------------------------------------------------------------- smoothing


def test_smooth_predictions_empty_history_is_unknown():
    assert smooth_predictions((), SmootherConfig()) == "unknown"


def test_smooth_predictions_low_confidence_counts_as_unknown():
    cfg = SmootherConfig(vote_k=5, min_confidence=0.6)
    history = (
        ("walking", 0.5),
        ("walking", 0.55),
        ("walking", 0.4),
        ("sitting", 0.9),
        ("sitting", 0.8),
    )
    # 3 low-confidence "walking" entries -> "unknown"; 2 "sitting" pass.
    # unknown(3) beats sitting(2).
    assert smooth_predictions(history, cfg) == "unknown"


def test_smooth_predictions_majority_wins():
    cfg = SmootherConfig(vote_k=5, min_confidence=0.6)
    history = (
        ("walking", 0.9),
        ("walking", 0.9),
        ("walking", 0.9),
        ("sitting", 0.9),
        ("sitting", 0.9),
    )
    assert smooth_predictions(history, cfg) == "walking"


def test_smooth_predictions_tie_breaks_to_most_recent():
    cfg = SmootherConfig(vote_k=4, min_confidence=0.6)
    history = (
        ("walking", 0.9),
        ("walking", 0.9),
        ("sitting", 0.9),
        ("sitting", 0.9),
    )
    # 2-2 tie: most recent among tied labels is "sitting" (last entry).
    assert smooth_predictions(history, cfg) == "sitting"


def test_smooth_predictions_only_last_vote_k_considered():
    cfg = SmootherConfig(vote_k=3, min_confidence=0.6)
    history = (
        ("sitting", 0.9),
        ("sitting", 0.9),
        ("sitting", 0.9),
        ("sitting", 0.9),
        ("walking", 0.9),
        ("walking", 0.9),
        ("walking", 0.9),
    )
    # only the last 3 entries (all "walking") count.
    assert smooth_predictions(history, cfg) == "walking"


# ------------------------------------------------------- buffering + windows


def _feed_session_into_buffers(activity: str, duration_s: float, seed: int = 0):
    """Build StreamBuffers from simulate.generate_lines for 3 receivers."""
    buffers: dict = {}
    rx_ids = ("rx1", "rx2", "rx3")
    for i, rx in enumerate(rx_ids):
        lines = generate_lines(activity=activity, duration_s=duration_s, seed=seed + 1000 * i)
        for ts, line in lines:
            frame = parse_line(line, host_ts=ts)
            if frame is not None:
                buffers = append_frame(buffers, rx, frame.host_ts, frame.amplitude)
    return buffers, rx_ids


def test_make_realtime_window_full_coverage():
    cfg = PreprocessConfig()  # window_s=3.0, fs=100 -> T=300, 52 usable
    buffers, rx_ids = _feed_session_into_buffers("walking", duration_s=5.0)
    t_end = 4.0  # well within the buffered 5s, full 3s window available
    window = make_realtime_window(buffers, rx_ids, cfg, t_end)
    assert window is not None
    assert window.shape == (3, 300, 52)
    assert window.dtype == np.float32


def test_make_realtime_window_missing_receiver_returns_none():
    cfg = PreprocessConfig()
    buffers, rx_ids = _feed_session_into_buffers("walking", duration_s=5.0)
    # Drop one receiver's buffer entirely.
    incomplete = dict(buffers)
    del incomplete[rx_ids[0]]
    window = make_realtime_window(incomplete, rx_ids, cfg, 4.0)
    assert window is None


def test_make_realtime_window_truncated_coverage_returns_none():
    cfg = PreprocessConfig()  # window_s=3.0
    buffers, rx_ids = _feed_session_into_buffers("walking", duration_s=5.0)
    # Only keep the last ~1s of samples per receiver (t in [3.0, 4.0]),
    # much less than the needed 3s window ending at t_end=4.0.
    truncated = {}
    for rx, samples in buffers.items():
        truncated[rx] = [(ts, amp) for ts, amp in samples if ts >= 3.0]
    window = make_realtime_window(truncated, rx_ids, cfg, 4.0)
    assert window is None


# -------------------------------------------------------- fall alert fast path


def test_fall_alert_fires_on_confident_raw_fall_and_history_is_bounded(monkeypatch):
    import csihar.realtime as rt

    cfg = PreprocessConfig()
    buffers, _ = _feed_session_into_buffers("walking", duration_s=5.0)
    script = iter(
        [("walking", 0.9), ("walking", 0.9), ("falling", 0.9), ("falling", 0.3)]
    )

    def fake_predict(bundle, window):
        label, conf = next(script)
        dist = np.full((1, 6), (1.0 - conf) / 5.0, dtype=np.float32)
        dist[0, 0] = conf
        return [label], dist

    monkeypatch.setattr(rt, "predict", fake_predict)
    engine = RealtimeEngine(
        bundle=None, pre_cfg=cfg,
        smoother=SmootherConfig(vote_k=3, min_confidence=0.6),
    )
    engine._buffers = buffers

    p1 = engine.tick(4.0)
    p2 = engine.tick(4.0)
    p3 = engine.tick(4.0)
    p4 = engine.tick(4.0)
    assert p1.fall_alert is False and p2.fall_alert is False
    # A single confident fall window loses the 3-vote majority (smoothed stays
    # "walking") but must still raise the safety-critical alert.
    assert p3.smoothed_label == "walking"
    assert p3.fall_alert is True
    # Low-confidence fall: no alert.
    assert p4.fall_alert is False
    # History is pruned to vote_k, so long demos don't grow memory unbounded.
    assert len(engine._history) == 3


# ---------------------------------------------------------- engine e2e smoke


@pytest.fixture(scope="module")
def sim_checkpoint(tmp_path_factory):
    """Train a tiny checkpoint exactly like test_train.py's smoke fixture."""
    root = tmp_path_factory.mktemp("raw")
    seed = 0
    for subject in ("s1", "s2"):
        for label in ("background", "walking"):
            write_session(
                root, label=label, subject=subject, duration_s=8.0, seed=seed
            )
            seed += 1
    ds = assemble_dataset(root, PreprocessConfig())

    ckpt_dir = tmp_path_factory.mktemp("ckpt")
    cfg = TrainConfig(
        model="cnn",
        split="random",
        epochs=8,
        batch_size=8,
        patience=8,
        results_csv=str(ckpt_dir / "results" / "dl.csv"),
        checkpoints_dir=str(ckpt_dir / "checkpoints"),
        figures_dir=str(ckpt_dir / "figures"),
    )
    report, checkpoint_path = train_model(ds, cfg)
    assert report.accuracy > 0.7
    return load_checkpoint(checkpoint_path)


def _stream_session_and_collect_smoothed(
    bundle, session_dir, hop_s: float = 1.5
) -> list[str]:
    # The smoke checkpoint is a tiny, few-epoch model: its argmax is accurate
    # (trivially separable simulated data) but softmax stays under-confident
    # (~0.2 on a 6-way head). Use a low min_confidence here so the vote
    # exercises real label agreement instead of being swamped by "unknown" —
    # SmootherConfig's default (0.6) is tuned for a properly-trained model.
    smoother = SmootherConfig(vote_k=5, min_confidence=0.1)
    engine = RealtimeEngine(bundle, PreprocessConfig(), smoother)
    source = replay_source(session_dir, sleep_fn=lambda _: None)
    smoothed: list[str] = []
    next_tick = None
    for rx_id, host_ts, amplitude in source:
        engine.feed(rx_id, host_ts, amplitude)
        if next_tick is None:
            next_tick = host_ts + hop_s
        while host_ts >= next_tick:
            pred = engine.tick(next_tick)
            if pred is not None:
                assert isinstance(pred, Prediction)
                smoothed.append(pred.smoothed_label)
            next_tick += hop_s
    return smoothed


def test_engine_streams_walking_session_majority_walking(sim_checkpoint, tmp_path):
    session_dir = write_session(
        tmp_path, label="walking", subject="s3", duration_s=10.0, seed=999
    )
    smoothed = _stream_session_and_collect_smoothed(sim_checkpoint, session_dir)
    assert len(smoothed) > 0
    frac_walking = sum(1 for lbl in smoothed if lbl == "walking") / len(smoothed)
    assert frac_walking >= 0.6, f"only {frac_walking:.2f} predicted walking: {smoothed}"


def test_engine_streams_background_session_majority_background_or_unknown(
    sim_checkpoint, tmp_path
):
    session_dir = write_session(
        tmp_path, label="background", subject="s4", duration_s=10.0, seed=888
    )
    smoothed = _stream_session_and_collect_smoothed(sim_checkpoint, session_dir)
    assert len(smoothed) > 0
    frac_ok = sum(1 for lbl in smoothed if lbl in ("background", "unknown")) / len(smoothed)
    assert frac_ok >= 0.6, f"only {frac_ok:.2f} predicted background/unknown: {smoothed}"
