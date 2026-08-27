import csv
import json
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import torch

from csihar import train as train_module
from csihar.dataset import LABEL_NAMES, assemble_dataset
from csihar.models import build_model
from csihar.models.inference import (
    CHECKPOINT_KEYS,
    load_checkpoint,
    predict,
    save_checkpoint,
)
from csihar.preprocessing import PreprocessConfig
from csihar.simulate import write_session
from csihar.train import (
    TrainConfig,
    _class_weights,
    _grouped_val_split,
    norm_apply,
    norm_fit,
    resolve_device,
    train_model,
)

# ---------------------------------------------------------------- TrainConfig


def test_train_config_json_roundtrip():
    cfg = TrainConfig(model="cnn_lstm", split="cross-subject",
                      test_subject="s2", epochs=3, seed=7)
    again = TrainConfig.from_json(cfg.to_json())
    assert again == cfg
    assert json.loads(cfg.to_json())["test_subject"] == "s2"


# -------------------------------------------------------------- normalization


def test_norm_fit_shapes_and_zero_std_guard():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(10, 3, 20, 8)).astype(np.float32)
    X[:, 1, :, 3] = 42.0  # dead channel -> zero std
    mean, std = norm_fit(X)
    assert mean.shape == (3, 8) and std.shape == (3, 8)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert std[1, 3] == 1.0  # guard: zero std replaced, no divide-by-zero
    assert mean[1, 3] == pytest.approx(42.0)
    assert (std > 0).all()


def test_norm_fit_train_only_semantics():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(20, 2, 15, 6)).astype(np.float32)
    train_idx = np.arange(15)
    X_shifted = X.copy()
    X_shifted[15:] += 100.0  # shifted test windows must not leak into stats
    mean_train, _ = norm_fit(X_shifted[train_idx])
    mean_full, _ = norm_fit(X_shifted)
    assert np.allclose(mean_train, norm_fit(X[train_idx])[0])
    assert not np.allclose(mean_train, mean_full)


def test_norm_apply_standardizes_and_does_not_mutate():
    rng = np.random.default_rng(2)
    X = rng.normal(3.0, 4.0, size=(30, 2, 25, 5)).astype(np.float32)
    before = X.copy()
    mean, std = norm_fit(X)
    Z = norm_apply(X, mean, std)
    assert Z is not X
    assert np.array_equal(X, before)  # pure function, no mutation
    assert Z.dtype == np.float32
    assert np.abs(Z.mean(axis=(0, 2))).max() < 1e-3
    assert np.abs(Z.std(axis=(0, 2)) - 1.0).max() < 1e-3


def test_norm_fit_rejects_bad_shapes():
    with pytest.raises(ValueError):
        norm_fit(np.zeros((5, 10, 4), np.float32))
    with pytest.raises(ValueError):
        norm_fit(np.zeros((0, 3, 10, 4), np.float32))


# ------------------------------------------------------- checkpoint contract


@pytest.mark.parametrize("name,extra", [("cnn", {}), ("cnn_lstm", {"chunk_len": 16})])
def test_checkpoint_roundtrip_and_predict(tmp_path, name, extra):
    torch.manual_seed(0)
    config = {"n_rx": 2, "n_time": 64, "n_subcarriers": 32, "n_classes": 6, **extra}
    model = build_model(name, **config)
    rng = np.random.default_rng(3)
    mean = rng.normal(0, 1, size=(2, 32)).astype(np.float32)
    std = rng.uniform(0.5, 2.0, size=(2, 32)).astype(np.float32)

    path = save_checkpoint(
        tmp_path / f"{name}.pt", model=model, model_name=name,
        label_names=LABEL_NAMES, norm_mean=mean, norm_std=std, config=config,
    )
    assert path.exists()

    # exact on-disk contract Phase 5 depends on
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload.keys()) == set(CHECKPOINT_KEYS)

    bundle = load_checkpoint(path)
    assert bundle.model_name == name
    assert bundle.label_names == LABEL_NAMES
    assert not bundle.model.training  # eval mode
    assert np.allclose(bundle.norm_mean, mean)
    assert np.allclose(bundle.norm_std, std)
    assert bundle.config["n_classes"] == 6

    # single window and batch, confidences are a softmax distribution
    x_one = rng.normal(0, 1, size=(2, 64, 32)).astype(np.float32)
    labels, conf = predict(bundle, x_one)
    assert labels[0] in LABEL_NAMES and len(labels) == 1
    assert conf.shape == (1, 6)
    x_batch = rng.normal(0, 1, size=(5, 2, 64, 32)).astype(np.float32)
    labels, conf = predict(bundle, x_batch)
    assert len(labels) == 5 and all(lbl in LABEL_NAMES for lbl in labels)
    assert conf.shape == (5, 6)
    assert np.allclose(conf.sum(axis=1), 1.0, atol=1e-5)
    assert (conf >= 0).all()


def test_save_checkpoint_validates_contract(tmp_path):
    model = build_model("cnn", n_rx=1, n_time=64, n_subcarriers=32, n_classes=6)
    mean = np.zeros((1, 32), np.float32)
    std = np.ones((1, 32), np.float32)
    good = {"n_rx": 1, "n_time": 64, "n_subcarriers": 32, "n_classes": 6}
    with pytest.raises(ValueError, match="missing required"):
        save_checkpoint(tmp_path / "a.pt", model=model, model_name="cnn",
                        label_names=LABEL_NAMES, norm_mean=mean, norm_std=std,
                        config={"n_rx": 1})
    with pytest.raises(ValueError, match="chunk_len"):
        save_checkpoint(tmp_path / "b.pt", model=model, model_name="cnn_lstm",
                        label_names=LABEL_NAMES, norm_mean=mean, norm_std=std,
                        config=good)
    with pytest.raises(ValueError, match="shape"):
        save_checkpoint(tmp_path / "c.pt", model=model, model_name="cnn",
                        label_names=LABEL_NAMES, norm_mean=np.zeros((3, 32)),
                        norm_std=std, config=good)


def test_predict_rejects_wrong_shapes(tmp_path):
    model = build_model("cnn", n_rx=2, n_time=64, n_subcarriers=32, n_classes=6)
    path = save_checkpoint(
        tmp_path / "m.pt", model=model, model_name="cnn",
        label_names=LABEL_NAMES,
        norm_mean=np.zeros((2, 32), np.float32),
        norm_std=np.ones((2, 32), np.float32),
        config={"n_rx": 2, "n_time": 64, "n_subcarriers": 32, "n_classes": 6},
    )
    bundle = load_checkpoint(path)
    with pytest.raises(ValueError):
        predict(bundle, np.zeros((64, 32), np.float32))  # 2-D
    with pytest.raises(ValueError):
        predict(bundle, np.zeros((3, 64, 32), np.float32))  # wrong n_rx
    with pytest.raises(ValueError):
        # wrong T: adaptive pooling would silently accept it otherwise
        predict(bundle, np.zeros((2, 128, 32), np.float32))


# ------------------------------------------------- validation split + weights


def test_grouped_val_split_holds_out_whole_sessions():
    y = np.tile(np.array([0, 1]), 20)  # both classes in every session
    sessions = np.repeat(np.array(["a", "b", "c", "d"]), 10)
    fit_pos, val_pos = _grouped_val_split(y, sessions, val_fraction=0.25, seed=0)
    assert len(val_pos) > 0 and len(fit_pos) > 0
    assert not set(sessions[fit_pos]) & set(sessions[val_pos])  # no straddling
    assert set(y[fit_pos].tolist()) == {0, 1}  # fit keeps every class
    assert len(fit_pos) + len(val_pos) == len(y)


def test_grouped_val_split_single_session_falls_back_stratified():
    y = np.tile(np.array([0, 1]), 10)
    sessions = np.full(20, "only")
    with pytest.warns(UserWarning, match="single session"):
        fit_pos, val_pos = _grouped_val_split(y, sessions, 0.2, seed=0)
    assert len(fit_pos) + len(val_pos) == 20 and len(val_pos) > 0


def test_class_weights_inverse_frequency():
    y = np.array([0] * 90 + [5] * 10)
    w = _class_weights(y, n_classes=6)
    assert w.shape == (6,)
    assert w[5] > w[0]  # scarce class weighted up
    assert (w[np.array([1, 2, 3, 4])] == 1.0).all()  # absent classes neutral
    # weighted total count preserved: sum_c w_c * n_c == n
    assert float(w[0] * 90 + w[5] * 10) == pytest.approx(100.0)


# ---------------------------------------------------------- end-to-end smoke


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


def test_train_model_smoke(sim_dataset, tmp_path):
    cfg = TrainConfig(
        model="cnn",
        split="random",
        epochs=8,
        batch_size=8,
        patience=8,
        results_csv=str(tmp_path / "results" / "dl.csv"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        figures_dir=str(tmp_path / "figures"),
    )
    report, checkpoint_path = train_model(sim_dataset, cfg)

    # walking vs background in simulation is trivially separable
    assert report.accuracy > 0.7

    # checkpoint exists, honors the contract, and is usable for inference
    assert checkpoint_path.exists()
    assert checkpoint_path.name == "cnn_random_0.pt"
    bundle = load_checkpoint(checkpoint_path)
    labels, conf = predict(bundle, sim_dataset.X[:2])
    assert len(labels) == 2 and conf.shape == (2, len(LABEL_NAMES))

    # provenance row landed in the results CSV
    with open(cfg.results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["model"] == "cnn"
    assert rows[0]["split"] == "random"
    assert rows[0]["seed"] == "0"
    assert float(rows[0]["accuracy"]) == pytest.approx(report.accuracy, abs=1e-3)
    # Config round-trips exactly, except `device`, which is deliberately
    # rewritten to the device that actually ran (see the resolved-device test).
    written = TrainConfig.from_json(rows[0]["config"])
    assert written == replace(cfg, device=written.device)
    assert written.device == str(resolve_device(cfg.device))

    # confusion matrix figure regenerated from code
    assert (tmp_path / "figures" / "cm_cnn_random.png").exists()


def test_train_model_cross_subject_requires_subject(sim_dataset):
    cfg = TrainConfig(model="cnn", split="cross-subject", epochs=1)
    with pytest.raises(ValueError, match="test_subject"):
        train_model(sim_dataset, cfg)


def test_train_model_rejects_unknown_names(sim_dataset):
    with pytest.raises(ValueError, match="unknown model"):
        train_model(sim_dataset, TrainConfig(model="mlp", epochs=1))
    with pytest.raises(ValueError, match="unknown split"):
        train_model(sim_dataset, TrainConfig(split="temporal", epochs=1))


def test_resolve_device_honours_an_explicit_name():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_uses_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")


def test_device_is_part_of_the_committed_config():
    # Every knob of a run must be reproducible from the logged config.
    assert "device" in TrainConfig().to_json()
    assert TrainConfig.from_json(TrainConfig(device="cpu").to_json()).device == "cpu"


def test_checkpoint_is_saved_on_cpu(sim_dataset, tmp_path):
    """A GPU-trained checkpoint has to load on the laptop that runs the demo."""
    cfg = TrainConfig(
        epochs=1, batch_size=8, patience=8, device="cpu",
        results_csv=str(tmp_path / "r.csv"),
        checkpoints_dir=str(tmp_path / "ckpt"),
        figures_dir=str(tmp_path / "fig"),
    )
    _, path = train_model(sim_dataset, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert all(t.device.type == "cpu" for t in payload["state_dict"].values())


def test_results_row_records_the_resolved_device_not_auto(sim_dataset, tmp_path):
    """A config column saying "auto" cannot be compared across machines.

    CPU and MPS produce different accuracy on identical data and seeds, so the
    row has to name the hardware that actually ran, not the request for one.
    """
    cfg = TrainConfig(
        model="cnn", split="random", epochs=1, batch_size=8, patience=1,
        device="auto",
        results_csv=str(tmp_path / "results" / "dl.csv"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        figures_dir=str(tmp_path / "figures"),
    )
    train_model(sim_dataset, cfg)

    with open(cfg.results_csv, newline="") as fh:
        row = next(csv.DictReader(fh))
    recorded = json.loads(row["config"])["device"]
    assert recorded != "auto"
    assert recorded == str(resolve_device("auto"))


def test_degenerate_split_is_warned_and_written_into_the_results_row(
    sim_dataset, tmp_path
):
    """The DL path needs the same guardrail baseline.py has.

    A cross-session test set holding one class scores ~0.97 accuracy while the
    model has learned nothing; without the note the CSV looks like a result.
    """
    ds = sim_dataset
    train_idx = np.flatnonzero(ds.y == ds.label_names.index("background"))
    test_idx = np.flatnonzero(ds.y != ds.label_names.index("background"))
    assert len(train_idx) and len(test_idx)

    cfg = TrainConfig(
        model="cnn", split="random", epochs=1, batch_size=8, patience=1,
        device="cpu", notes="pilot",
        results_csv=str(tmp_path / "results" / "dl.csv"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        figures_dir=str(tmp_path / "figures"),
    )

    def _degenerate(dataset, config):
        return train_idx, test_idx, "cross-session"

    with mock.patch.object(train_module, "_resolve_split", _degenerate):
        with pytest.warns(UserWarning, match="DEGENERATE SPLIT"):
            train_model(ds, cfg)

    with open(cfg.results_csv, newline="") as fh:
        row = next(csv.DictReader(fh))
    assert "DEGENERATE SPLIT" in row["notes"]
    assert "pilot" in row["notes"]  # the operator's own note survives
