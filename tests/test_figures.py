"""Tests for csihar.figures — parsing, aggregation, and figure generation."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from csihar.evaluate import RESULT_COLUMNS
from csihar.figures import (
    generate_all,
    load_results,
    parse_notes,
    plot_ablation,
    plot_falling_recall,
    plot_split_comparison,
    split_family,
    variant_value,
)


def _write_results(
    path: Path, rows: list[dict]
) -> Path:
    """Write a results CSV with the real RESULT_COLUMNS header."""
    defaults = {
        "timestamp": "2026-07-12T00:00:00", "git_sha": "abc1234",
        "model": "cnn", "split": "random", "seed": 0, "config": "{}",
        "accuracy": "0.9000", "macro_f1": "0.8800", "falling_recall": "0.9500",
        "n_train": 100, "n_test": 25, "notes": "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**defaults, **row})
    return path


# ------------------------------------------------------------------ parsing


def test_parse_notes_roundtrip():
    assert parse_notes("ablation=rate variant=50Hz") == {
        "ablation": "rate", "variant": "50Hz",
    }


def test_parse_notes_ignores_malformed_tokens():
    assert parse_notes("plain =x y= a=b") == {"a": "b"}


def test_parse_notes_empty_and_nan():
    assert parse_notes("") == {}
    assert parse_notes(float("nan")) == {}  # pandas NaN cell


def test_parse_notes_survives_an_appended_coverage_note():
    """`train` joins run notes with a degenerate-split warning using "; ".

    Splitting on whitespace alone left the semicolon glued to the value
    ("25Hz;"), which made `variant_value` raise on window/rate rows -- i.e.
    `make figures` crashed on exactly the runs the coverage guard flags.
    """
    notes = (
        "ablation=rate variant=25Hz; DEGENERATE SPLIT - test missing "
        "4 class(es): lying, sitting, standing, walking"
    )
    parsed = parse_notes(notes)
    assert parsed == {"ablation": "rate", "variant": "25Hz"}
    assert variant_value(parsed["ablation"], parsed["variant"]) == 25.0


def test_parse_notes_coverage_note_does_not_split_a_series():
    """The receivers ablation did not raise -- it silently produced a second
    series ("rx0" and "rx0;") for one variant, which is worse than a crash."""
    clean = parse_notes("ablation=receivers variant=rx0")
    flagged = parse_notes("ablation=receivers variant=rx0; DEGENERATE SPLIT - x")
    assert clean == flagged


def test_variant_value_receivers():
    assert variant_value("receivers", "rx0") == 1.0
    assert variant_value("receivers", "rx0+rx1+rx2") == 3.0


def test_variant_value_window_and_rate():
    assert variant_value("window", "1.5s") == 1.5
    assert variant_value("rate", "50Hz") == 50.0


def test_variant_value_unknown_ablation():
    with pytest.raises(ValueError, match="unknown ablation"):
        variant_value("bogus", "x")


def test_split_family_collapses_loso_folds():
    assert split_family("cross-subject:s1") == "cross-subject"
    assert split_family("random") == "random"


# ------------------------------------------------------------------ loading


def test_load_results_missing_files_gives_empty(tmp_path):
    df = load_results([tmp_path / "nope.csv"])
    assert df.empty


def test_load_results_concatenates_and_adds_split_fam(tmp_path):
    a = _write_results(tmp_path / "a.csv", [{"split": "cross-subject:s1"}])
    b = _write_results(tmp_path / "b.csv", [{"split": "random"}])
    df = load_results([a, b])
    assert len(df) == 2
    assert set(df["split_fam"]) == {"cross-subject", "random"}


# ----------------------------------------------------------------- plotting


def test_plot_split_comparison_writes_png(tmp_path):
    csv_path = _write_results(
        tmp_path / "dl.csv",
        [
            {"model": "cnn", "split": "random", "seed": s, "macro_f1": 0.9 - s * 0.01}
            for s in range(3)
        ]
        + [{"model": "cnn_lstm", "split": "cross-session", "macro_f1": 0.7}],
    )
    out = plot_split_comparison(load_results([csv_path]), tmp_path / "fig.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_plot_split_comparison_empty_returns_none(tmp_path):
    assert plot_split_comparison(pd.DataFrame(), tmp_path / "fig.png") is None
    assert not (tmp_path / "fig.png").exists()


def test_plot_falling_recall_writes_png(tmp_path):
    csv_path = _write_results(
        tmp_path / "dl.csv", [{"falling_recall": "0.8000"}]
    )
    out = plot_falling_recall(load_results([csv_path]), tmp_path / "fr.png")
    assert out is not None and out.exists()


def test_plot_ablation_writes_png(tmp_path):
    rows = [
        {"notes": f"ablation=rate variant={hz}Hz", "seed": s,
         "macro_f1": 0.5 + hz / 250, "split": split}
        for hz in (25, 50, 100)
        for s in (0, 1)
        for split in ("random", "cross-session")
    ]
    csv_path = _write_results(tmp_path / "ablation_rate.csv", rows)
    out = plot_ablation(load_results([csv_path]), "rate", tmp_path / "ab.png")
    assert out is not None and out.exists()


def test_plot_ablation_no_matching_rows_returns_none(tmp_path):
    csv_path = _write_results(tmp_path / "r.csv", [{"notes": ""}])
    assert plot_ablation(load_results([csv_path]), "rate", tmp_path / "x.png") is None


def test_plot_ablation_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown ablation"):
        plot_ablation(pd.DataFrame({"notes": []}), "bogus", tmp_path / "x.png")


# ------------------------------------------------------------------- driver


def test_generate_all_empty_dir_skips_everything(tmp_path):
    outcomes = generate_all(tmp_path / "results", tmp_path / "figs")
    assert outcomes and all(path is None for path in outcomes.values())
    assert not (tmp_path / "figs").exists()


def test_generate_all_produces_available_figures(tmp_path):
    results = tmp_path / "results"
    _write_results(results / "dl.csv", [{"model": "cnn"}])
    _write_results(
        results / "ablation_receivers.csv",
        [
            {"notes": f"ablation=receivers variant={v}", "macro_f1": 0.6 + i * 0.1}
            for i, v in enumerate(("rx0", "rx0+rx1", "rx0+rx1+rx2"))
        ],
    )
    outcomes = generate_all(results, tmp_path / "figs")
    assert outcomes["split_comparison"] is not None
    assert outcomes["falling_recall"] is not None
    assert outcomes["ablation_receivers"] is not None
    assert outcomes["ablation_window"] is None
    assert outcomes["ablation_rate"] is None
    for path in outcomes.values():
        if path is not None:
            assert path.exists() and path.stat().st_size > 0


def test_cli_runs_on_empty_dir(tmp_path, capsys):
    from csihar.figures import main

    assert main(["--results-dir", str(tmp_path), "--out", str(tmp_path / "f")]) == 0
    out = capsys.readouterr().out
    assert "nothing generated" in out
