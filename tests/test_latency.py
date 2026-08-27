import json
import math

import numpy as np
import pytest

from csihar.latency import (
    LatencyReport,
    StageTiming,
    main,
    measure,
    summarize,
    untrained_bundle,
)
from csihar.models.inference import predict
from csihar.preprocessing import PreprocessConfig
from csihar.realtime_sources import simulate_source


def _report(duration_s: float = 8.0, receivers: int = 2, **kwargs) -> LatencyReport:
    cfg = PreprocessConfig(window_s=3.0, hop_s=1.5)
    rx_ids = tuple(f"rx{i + 1}" for i in range(receivers))
    source = simulate_source("walking", duration_s, rx_ids=rx_ids)
    return measure(source, pre_cfg=cfg, **kwargs)


def test_summarize_converts_seconds_to_milliseconds():
    stage = summarize("demo", [0.001, 0.002, 0.003, 0.004])
    assert isinstance(stage, StageTiming)
    assert stage.n_calls == 4
    assert stage.mean_ms == pytest.approx(2.5)
    assert stage.p50_ms == pytest.approx(2.5)
    assert stage.max_ms == pytest.approx(4.0)


def test_summarize_empty_is_nan_not_an_error():
    stage = summarize("demo", [])
    assert stage.n_calls == 0
    assert math.isnan(stage.mean_ms)
    assert math.isnan(stage.p95_ms)


def test_untrained_bundle_is_accepted_by_predict():
    bundle = untrained_bundle("cnn", n_rx=3, n_time=300, n_subcarriers=52)
    window = np.zeros((3, 300, 52), dtype=np.float32)

    labels, confidences = predict(bundle, window)

    assert len(labels) == 1
    assert labels[0] in bundle.label_names
    assert confidences.shape == (1, len(bundle.label_names))
    assert confidences.sum() == pytest.approx(1.0, abs=1e-5)


def test_measure_reports_every_stage():
    report = _report()
    assert tuple(s.name for s in report.stages) == (
        "parse", "ingest", "window", "inference",
    )
    assert report.n_frames > 0
    assert report.n_ticks > 0
    assert report.n_receivers == 2
    assert all(s.n_calls > 0 for s in report.stages)


def test_end_to_end_latency_is_buffering_plus_compute():
    # The window length dominates: a label always describes a window that
    # closed window_s ago, so end-to-end must include that budget.
    report = _report()
    assert report.end_to_end_p95_ms == pytest.approx(
        report.window_s * 1000.0 + report.tick_compute_p95_ms
    )
    assert report.end_to_end_p95_ms > report.window_s * 1000.0


def test_pipeline_keeps_up_with_the_stream_on_this_host():
    report = _report()
    assert report.tick_compute_p95_ms < report.tick_budget_ms
    assert report.realtime_capable


def test_max_ticks_caps_the_run():
    report = _report(duration_s=20.0, max_ticks=2)
    assert report.n_ticks == 2


def test_frame_budget_is_shared_across_receivers():
    one = _report(receivers=1)
    three = _report(receivers=3)
    assert one.frame_budget_ms == pytest.approx(3 * three.frame_budget_ms)


def test_by_name_raises_for_an_unknown_stage():
    report = _report()
    assert report.by_name("inference").name == "inference"
    with pytest.raises(KeyError):
        report.by_name("nope")


def test_cli_writes_json_and_exits_zero(tmp_path, capsys):
    out = tmp_path / "latency.json"
    code = main(
        [
            "--simulate", "walking", "--duration", "8", "--receivers", "2",
            "--max-ticks", "2", "--json", str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["n_ticks"] == 2
    assert {s["name"] for s in payload["stages"]} == {
        "parse", "ingest", "window", "inference",
    }
    assert "untrained" in capsys.readouterr().out
