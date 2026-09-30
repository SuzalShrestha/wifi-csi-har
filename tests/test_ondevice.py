"""Budget analyzer for the on-device (ESP32-S3) inference stretch goal."""

from __future__ import annotations

import pytest

from csihar.ondevice import (
    ARENA_SAFETY_FACTOR,
    FLASH_BYTES,
    format_budget,
    main,
    profile_model,
)


def budget(model: str = "cnn", n_rx: int = 1, n_time: int = 300):
    return profile_model(
        model, n_rx=n_rx, n_time=n_time, n_subcarriers=52, n_classes=6
    )


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        profile_model("resnet", n_rx=1, n_time=300, n_subcarriers=52, n_classes=6)


def test_int8_weights_are_a_quarter_of_float32():
    b = budget()
    assert b.weights_float32 == b.n_params * 4
    assert b.weights_int8 == b.n_params
    assert b.arena_float32 == b.arena_int8 * 4


def test_parameter_count_matches_the_built_model():
    from csihar.models import build_model

    model = build_model("cnn", n_rx=1, n_time=300, n_subcarriers=52, n_classes=6)
    assert budget().n_params == sum(p.numel() for p in model.parameters())


def test_every_traced_layer_is_reported():
    """A silent hook failure would understate the arena, which is the whole
    point of the analysis -- so assert the trunk was actually traced."""
    b = budget()
    kinds = [lb.kind for lb in b.layers]
    assert kinds.count("Conv2d") == 4  # the CNN's four conv blocks
    assert "Linear" in kinds
    assert all(lb.out_shape[0] == 1 for lb in b.layers)  # batch of one


def test_arena_is_the_peak_live_working_set_not_the_total():
    """Summing every tensor would massively overstate the requirement; the
    arena only has to hold the tensors live at one op."""
    b = budget()
    live = [lb.live_bytes_int8 for lb in b.layers]
    assert b.arena_int8 == max(live)
    assert b.arena_int8 < sum(live)


def test_arena_peaks_in_the_first_conv_block():
    """The finding the report rests on: the constraint is the full-resolution
    first feature map, not the weights. If a future architecture change moves
    the peak, the recommendation (stride the front end) stops following."""
    b = budget()
    peak = max(b.layers, key=lambda lb: lb.live_bytes_int8)
    assert peak.name.startswith("features.0")
    assert b.weights_int8 < b.arena_int8


def test_shorter_windows_shrink_the_arena_proportionally():
    full, half = budget(n_time=300), budget(n_time=150)
    assert full.n_params == half.n_params  # adaptive pooling: weights unchanged
    assert half.arena_int8 == pytest.approx(full.arena_int8 / 2, rel=0.02)
    assert half.macs == pytest.approx(full.macs / 2, rel=0.05)


def test_receiver_count_barely_moves_the_budget():
    """Receivers are input channels, so n_rx only widens the first conv --
    which is why the on-device constraint is data availability (a board has
    only its own CSI), not memory."""
    one, three = budget(n_rx=1), budget(n_rx=3)
    assert three.n_params > one.n_params
    assert three.n_params - one.n_params < 0.01 * one.n_params


def test_weights_fit_flash_but_arena_exceeds_internal_sram():
    """Pins the measured conclusion: flash is not the constraint, RAM is."""
    b = budget()
    assert b.fits_flash
    assert b.weights_int8 < FLASH_BYTES
    assert not b.fits_internal_sram


def test_safety_factor_is_applied():
    b = budget()
    assert b.arena_with_safety == int(b.arena_int8 * ARENA_SAFETY_FACTOR)
    assert b.arena_with_safety > b.arena_int8


def test_cnn_lstm_is_profiled_including_its_lstm():
    b = budget(model="cnn_lstm")
    assert any(lb.kind == "LSTM" for lb in b.layers)
    assert b.macs > 0


def test_format_budget_reports_both_verdicts():
    text = format_budget(budget(), per_layer=True)
    assert "weights fit: YES" in text
    assert "NO - needs PSRAM" in text
    assert "features.0.0" in text  # per-layer table rendered


def test_cli_runs(capsys):
    assert main(["--model", "cnn", "--n-rx", "1"]) == 0
    assert "parameters" in capsys.readouterr().out
