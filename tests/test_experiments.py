import argparse
import csv
import json

import numpy as np
import pytest

from csihar.dataset import HarDataset, assemble_dataset
from csihar.experiments import (
    AblationResult,
    crop_window,
    decimate_time,
    format_summary,
    load_config_file,
    rate_variants,
    receiver_variants,
    resolve_settings,
    run_ablation,
    subset_receivers,
    window_variants,
)
from csihar.preprocessing import PreprocessConfig
from csihar.simulate import write_session
from csihar.train import TrainConfig

# ------------------------------------------------------------------ fixtures


def make_ds(n: int = 6, n_rx: int = 3, n_time: int = 300, n_sub: int = 8) -> HarDataset:
    """Small hand-built dataset for fast transform tests (no simulation)."""
    rng = np.random.default_rng(42)
    return HarDataset(
        X=rng.normal(size=(n, n_rx, n_time, n_sub)).astype(np.float32),
        y=(np.arange(n) % 2).astype(np.int64),
        subjects=np.array([f"s{i % 2 + 1}" for i in range(n)], dtype=str),
        sessions=np.array([f"sess{i % 3}" for i in range(n)], dtype=str),
        environments=np.array(["room_a"] * n, dtype=str),
    )


def snapshot(ds: HarDataset) -> dict[str, np.ndarray]:
    return {
        name: getattr(ds, name).copy()
        for name in ("X", "y", "subjects", "sessions", "environments")
    }


def assert_unchanged(ds: HarDataset, snap: dict[str, np.ndarray]) -> None:
    for name, before in snap.items():
        assert np.array_equal(getattr(ds, name), before), f"{name} was mutated"


def assert_metadata_passthrough(out: HarDataset, ds: HarDataset) -> None:
    assert np.array_equal(out.y, ds.y)
    assert np.array_equal(out.subjects, ds.subjects)
    assert np.array_equal(out.sessions, ds.sessions)
    assert np.array_equal(out.environments, ds.environments)
    assert out.label_names == ds.label_names


@pytest.fixture(scope="module")
def sim_dataset(tmp_path_factory):
    """s1,s2 x background,walking -> ~16 trivially separable windows."""
    root = tmp_path_factory.mktemp("raw")
    seed = 0
    for subject in ("s1", "s2"):
        for label in ("background", "walking"):
            write_session(
                root, label=label, subject=subject, duration_s=8.0, seed=seed
            )
            seed += 1
    return assemble_dataset(root, PreprocessConfig())


# ----------------------------------------------------------- subset_receivers


def test_subset_receivers_shapes_values_and_purity():
    ds = make_ds()
    snap = snapshot(ds)
    out = subset_receivers(ds, (2, 0))
    assert out is not ds
    assert out.X.shape == (6, 2, 300, 8)
    assert np.array_equal(out.X, ds.X[:, [2, 0], :, :])
    assert_metadata_passthrough(out, ds)
    assert_unchanged(ds, snap)
    # mutating the result must not touch the input either (no shared buffers)
    out.X[:] = 0.0
    assert_unchanged(ds, snap)


def test_subset_receivers_single_and_full():
    ds = make_ds()
    assert subset_receivers(ds, (1,)).X.shape == (6, 1, 300, 8)
    full = subset_receivers(ds, (0, 1, 2))
    assert np.array_equal(full.X, ds.X)


@pytest.mark.parametrize("bad", [(), (0, 0), (3,), (-1,), (0, 5)])
def test_subset_receivers_rejects_bad_indices(bad):
    ds = make_ds()
    with pytest.raises(ValueError):
        subset_receivers(ds, bad)


# --------------------------------------------------------------- decimate_time


def test_decimate_time_shapes_values_and_purity():
    ds = make_ds()
    snap = snapshot(ds)
    out = decimate_time(ds, 2)
    assert out.X.shape == (6, 3, 150, 8)
    assert np.array_equal(out.X, ds.X[:, :, ::2, :])
    assert_metadata_passthrough(out, ds)
    assert_unchanged(ds, snap)


def test_decimate_time_factor_one_is_a_new_equal_dataset():
    ds = make_ds()
    out = decimate_time(ds, 1)
    assert out is not ds and out.X is not ds.X
    assert np.array_equal(out.X, ds.X)


def test_decimate_time_rejects_bad_factor():
    ds = make_ds(n_time=300)
    with pytest.raises(ValueError, match="integer >= 1"):
        decimate_time(ds, 0)
    with pytest.raises(ValueError, match="integer >= 1"):
        decimate_time(ds, 1.5)
    with pytest.raises(ValueError, match="minimum"):
        decimate_time(ds, 50)  # 300/50 = 6 < 8 remaining samples


# ---------------------------------------------------------------- crop_window


def test_crop_window_center_crop_and_purity():
    ds = make_ds(n_time=300)
    snap = snapshot(ds)
    out = crop_window(ds, 1.0, fs=100.0)
    assert out.X.shape == (6, 3, 100, 8)
    assert np.array_equal(out.X, ds.X[:, :, 100:200, :])  # centered
    assert_metadata_passthrough(out, ds)
    assert_unchanged(ds, snap)


def test_crop_window_full_length_and_odd_remainder():
    ds = make_ds(n_time=300)
    assert np.array_equal(crop_window(ds, 3.0).X, ds.X)
    out = crop_window(ds, 2.99)  # 299 samples -> start floor((300-299)/2) = 0
    assert out.X.shape[2] == 299
    assert np.array_equal(out.X, ds.X[:, :, 0:299, :])


def test_crop_window_rejects_bad_args():
    ds = make_ds(n_time=300)
    with pytest.raises(ValueError, match="re-assemble"):
        crop_window(ds, 4.0)  # longer than the assembled window
    with pytest.raises(ValueError):
        crop_window(ds, 0.0)
    with pytest.raises(ValueError):
        crop_window(ds, -1.0)
    with pytest.raises(ValueError):
        crop_window(ds, 1.0, fs=0.0)


# --------------------------------------------------------- variant enumeration


def test_receiver_variants_enumeration():
    variants = receiver_variants(3)
    assert len(variants) == 7  # 3 singles + 3 pairs + 1 full
    names = [name for name, _ in variants]
    assert names == [
        "rx0", "rx1", "rx2", "rx0+rx1", "rx0+rx2", "rx1+rx2", "rx0+rx1+rx2",
    ]
    assert dict(variants)["rx0+rx2"] == (0, 2)
    assert len(receiver_variants(1)) == 1
    with pytest.raises(ValueError):
        receiver_variants(0)


def test_window_variants_capped_by_dataset_length():
    assert [n for n, _ in window_variants(300)] == ["1s", "1.5s", "2s", "3s"]
    assert [n for n, _ in window_variants(150)] == ["1s", "1.5s"]
    with pytest.raises(ValueError):
        window_variants(50)  # nothing fits


def test_rate_variants_names_and_factors():
    assert rate_variants() == (("100Hz", 1), ("50Hz", 2), ("25Hz", 4))


# ----------------------------------------------------- driver arg validation


def test_run_ablation_validates_args():
    ds = make_ds()
    kwargs = dict(model="cnn", split="random", seeds=(0,), epochs=1)
    with pytest.raises(ValueError, match="unknown ablation"):
        run_ablation(ds, ablation="antennas", **kwargs)
    with pytest.raises(ValueError, match="unknown model"):
        run_ablation(ds, ablation="rate", **{**kwargs, "model": "mlp"})
    with pytest.raises(ValueError, match="unknown split"):
        run_ablation(ds, ablation="rate", **{**kwargs, "split": "temporal"})
    with pytest.raises(ValueError, match="seeds"):
        run_ablation(ds, ablation="rate", **{**kwargs, "seeds": ()})
    with pytest.raises(ValueError, match="epochs"):
        run_ablation(ds, ablation="rate", **{**kwargs, "epochs": 0})
    with pytest.raises(ValueError, match="unknown variant"):
        run_ablation(ds, ablation="rate", variants=("13Hz",), **kwargs)


# -------------------------------------------------------------- CLI settings


def test_resolve_settings_config_file_with_cli_overrides(tmp_path):
    config = {
        "data": "datasets/assembled/pilot.npz",
        "ablation": "rate",
        "model": "cnn",
        "splits": ["random", "cross-session"],
        "seeds": [0, 1, 2],
        "epochs": 30,
        "results_csv": "experiments/results/ablation_rate.csv",
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config))

    from csihar.experiments import _build_arg_parser

    args = _build_arg_parser().parse_args(
        ["--config", str(path), "--epochs", "2", "--split", "random"]
    )
    settings = resolve_settings(args)
    assert settings["data"] == "datasets/assembled/pilot.npz"
    assert settings["ablation"] == "rate"
    assert settings["epochs"] == 2  # CLI wins
    assert settings["splits"] == ("random",)  # CLI wins over config splits
    assert settings["seeds"] == (0, 1, 2)
    assert settings["results_csv"] == "experiments/results/ablation_rate.csv"


def test_resolve_settings_requires_data_and_ablation():
    ns = argparse.Namespace(
        config=None, data=None, ablation="rate", model=None, split=None,
        seeds=None, epochs=None, results_csv=None, checkpoints_dir=None,
        figures_dir=None, variants=None,
    )
    with pytest.raises(ValueError, match="--data"):
        resolve_settings(ns)
    ns2 = argparse.Namespace(**{**vars(ns), "data": "ds.npz", "ablation": None})
    with pytest.raises(ValueError, match="--ablation"):
        resolve_settings(ns2)


def test_load_config_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"ablation": "rate", "learning_rate": 0.1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config_file(path)


def test_committed_configs_are_loadable_and_consistent():
    from pathlib import Path

    for name in ("receivers", "window", "rate"):
        cfg = load_config_file(
            Path(__file__).resolve().parents[1]
            / "experiments" / "configs" / f"ablation_{name}.json"
        )
        assert cfg["ablation"] == name
        assert cfg["model"] == "cnn"
        assert cfg["splits"] == ["random", "cross-session"]
        assert cfg["seeds"] == [0, 1, 2]
        assert cfg["epochs"] == 30


def test_train_config_notes_roundtrip():
    cfg = TrainConfig(notes="ablation=rate variant=50Hz")
    assert TrainConfig.from_json(cfg.to_json()) == cfg


# ---------------------------------------------------------- end-to-end smoke


def test_run_ablation_rate_smoke(sim_dataset, tmp_path):
    results_csv = tmp_path / "results" / "ablation_rate.csv"
    results = run_ablation(
        sim_dataset,
        ablation="rate",
        model="cnn",
        split="random",
        seeds=(0,),
        epochs=1,
        results_csv=results_csv,
        checkpoints_dir=tmp_path / "checkpoints",
        figures_dir=tmp_path / "figures",
    )
    assert len(results) == 3
    assert all(isinstance(r, AblationResult) for r in results)
    assert {r.variant for r in results} == {"100Hz", "50Hz", "25Hz"}
    assert all(r.seed == 0 and r.subject is None for r in results)

    # provenance rows landed, one per variant, notes column populated
    with open(results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert {row["notes"] for row in rows} == {
        "ablation=rate variant=100Hz",
        "ablation=rate variant=50Hz",
        "ablation=rate variant=25Hz",
    }
    assert all(row["model"] == "cnn" and row["split"] == "random" for row in rows)

    # per-variant checkpoint dirs -> no filename collisions across variants
    for variant in ("100Hz", "50Hz", "25Hz"):
        assert (
            tmp_path / "checkpoints" / "rate" / variant / "cnn_random_0.pt"
        ).exists()
        assert (
            tmp_path / "figures" / "rate" / variant / "cm_cnn_random.png"
        ).exists()

    # summary table renders one line per run
    summary = format_summary(results)
    assert "50Hz" in summary and "falling_recall" in summary
    assert len(summary.splitlines()) == 2 + len(results)


def test_run_ablation_cross_subject_loso_smoke(sim_dataset, tmp_path):
    results_csv = tmp_path / "ablation_receivers.csv"
    results = run_ablation(
        sim_dataset,
        ablation="receivers",
        model="cnn",
        split="cross-subject",
        seeds=(0,),
        epochs=1,
        results_csv=results_csv,
        checkpoints_dir=tmp_path / "checkpoints",
        figures_dir=tmp_path / "figures",
        variants=("rx0",),  # restrict for speed; LOSO still loops subjects
    )
    assert {r.subject for r in results} == {"s1", "s2"}
    assert all(r.variant == "rx0" for r in results)

    with open(results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {row["split"] for row in rows} == {
        "cross-subject:s1", "cross-subject:s2",
    }
    assert all(row["notes"] == "ablation=receivers variant=rx0" for row in rows)
    for subject in ("s1", "s2"):
        assert (
            tmp_path / "checkpoints" / "receivers" / "rx0"
            / f"cnn_cross-subject_{subject}_0.pt"
        ).exists()
