"""Training engine for the deep models (Phase 4) + CLI.

Discipline encoded here, not left to convention:

- normalization statistics are fit on training windows only (validation and
  test see train-fitted stats);
- validation is carved from the TRAIN side of the split, stratified, so the
  test set is never touched until the single final evaluation;
- every run is seeded, appends a provenance row via ``evaluate.append_result``,
  saves a contract-conformant checkpoint, and regenerates its confusion
  matrix figure from code.

CLI:
    python -m csihar.train --data dataset.npz --model cnn --split random
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    HarDataset,
    load_dataset,
    label_coverage_note,
    split_coverage_note,
    split_cross_session,
    split_cross_subject,
    split_random,
)
from .evaluate import MetricsReport, append_result, compute_metrics, save_confusion_matrix
from .models import MODEL_NAMES, build_model
from .models.inference import save_checkpoint

SPLIT_NAMES: tuple[str, ...] = ("random", "cross-session", "cross-subject")
_DEFAULT_CHUNK_LEN = 25


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a compute device. "auto" prefers CUDA, then Apple MPS, then CPU.

    Training on this project's laptop is CPU-only; the real runs happen on
    Colab/Kaggle GPUs. Without this the model and its batches stay on the CPU
    no matter what hardware is attached, and a GPU runtime buys nothing.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class TrainConfig:
    """One committed place for every knob of a training run."""

    model: str = "cnn"
    split: str = "random"
    test_subject: str | None = None
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    patience: int = 5
    val_fraction: float = 0.1
    class_weighted: bool = True
    device: str = "auto"
    results_csv: str = "experiments/results/dl.csv"
    checkpoints_dir: str = "experiments/checkpoints"
    figures_dir: str = "docs/figures"
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrainConfig":
        return cls(**json.loads(text))


# ------------------------------------------------------------ normalization


def norm_fit(
    X: np.ndarray, idx: np.ndarray | None = None, chunk: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(receiver, subcarrier) mean/std over TRAIN windows only.

    X: (n, n_rx, T, S) -> mean, std each (n_rx, S) float32. Dead channels
    (zero std) get std 1.0 so normalization never divides by zero. ``idx``
    selects the train windows; they are read ``chunk`` at a time so a
    multi-GB dataset is never copied just to take its statistics.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"expected (n, n_rx, T, S), got shape {X.shape}")
    idx = np.arange(X.shape[0]) if idx is None else np.asarray(idx)
    if len(idx) == 0:
        raise ValueError("cannot fit normalization on zero windows")
    total = np.zeros((X.shape[1], X.shape[3]))
    total_sq = np.zeros_like(total)
    for start in range(0, len(idx), chunk):
        part = X[idx[start:start + chunk]].astype(np.float64)
        total += part.sum(axis=(0, 2))
        total_sq += np.square(part).sum(axis=(0, 2))
    count = len(idx) * X.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def norm_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize (n, n_rx, T, S) with (n_rx, S) stats; returns a NEW array."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"expected (n, n_rx, T, S), got shape {X.shape}")
    return ((X - mean[None, :, None, :]) / std[None, :, None, :]).astype(np.float32)


# ------------------------------------------------------------------- splits


def _resolve_split(
    ds: HarDataset, cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    """Map cfg.split to (train_idx, test_idx, split description for the CSV)."""
    if cfg.split == "random":
        train_idx, test_idx = split_random(ds, seed=cfg.seed)
        return train_idx, test_idx, "random"
    if cfg.split == "cross-session":
        train_idx, test_idx = split_cross_session(ds, seed=cfg.seed)
        return train_idx, test_idx, "cross-session"
    if cfg.split == "cross-subject":
        if not cfg.test_subject:
            raise ValueError("split 'cross-subject' requires cfg.test_subject")
        train_idx, test_idx = split_cross_subject(ds, cfg.test_subject)
        return train_idx, test_idx, f"cross-subject:{cfg.test_subject}"
    raise ValueError(f"unknown split {cfg.split!r}; expected one of {SPLIT_NAMES}")


def _stratified_val_split(
    y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Positions (into y) for fit/val; ~val_fraction per class goes to val.

    Classes with a single window stay entirely in fit. Never empties fit.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    fit_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        pos = rng.permutation(np.where(y == cls)[0])
        n_val = 0 if len(pos) < 2 else min(
            max(1, int(round(len(pos) * val_fraction))), len(pos) - 1
        )
        val_parts.append(pos[:n_val])
        fit_parts.append(pos[n_val:])
    fit_pos = np.sort(np.concatenate(fit_parts)).astype(np.int64)
    val_pos = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return fit_pos, val_pos


def _grouped_val_split(
    y: np.ndarray, groups: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Positions for fit/val where val is whole held-out groups (sessions).

    For cross-session/cross-subject evaluation, early stopping must see the
    same kind of distribution shift as the test set. A window-level random
    validation split shares activity bouts with fit (temporal leakage), so
    val macro-F1 tracks memorization and picks the wrong epoch. Falls back
    to the stratified split (with a warning) when train has < 2 groups.
    """
    unique = np.unique(groups)
    if len(unique) < 2:
        warnings.warn(
            "train side has a single session; validation falls back to a "
            "stratified window split (temporally leaky — early stopping "
            "will be optimistic)"
        )
        return _stratified_val_split(y, val_fraction, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    target = val_fraction * len(y)
    all_classes = set(y.tolist())
    val_groups: set[str] = set()
    count = 0
    for group in order:
        if val_groups and count >= target:
            break
        candidate = val_groups | {str(group)}
        fit_mask = ~np.isin(groups, list(candidate))
        # fit must keep every class or the model never learns it
        if set(y[fit_mask].tolist()) == all_classes:
            val_groups = candidate
            count = int(np.isin(groups, list(candidate)).sum())
    if not val_groups:
        warnings.warn(
            "no session can be held out without losing a class from fit; "
            "validation falls back to a stratified window split"
        )
        return _stratified_val_split(y, val_fraction, seed)
    val_mask = np.isin(groups, list(val_groups))
    return (
        np.where(~val_mask)[0].astype(np.int64),
        np.where(val_mask)[0].astype(np.int64),
    )


def _class_weights(y_fit: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency loss weights (falling is scarce and safety-critical).

    weight_c = n_fit / (n_present_classes * count_c); absent classes get 1.0
    (they never contribute to the loss anyway).
    """
    counts = np.bincount(y_fit, minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(n_classes, dtype=np.float64)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights.astype(np.float32))


# ----------------------------------------------------------------- training


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Batches of X[idx], normalized per batch.

    Normalizing up front held a float32 copy of every split next to X; at the
    M3 target (25k windows, 4.7 GB) that is ~3x the dataset and OOMs Colab.
    """
    idx = np.asarray(idx)

    def collate(positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = idx[positions]
        return (
            torch.from_numpy(norm_apply(X[rows], mean, std)),
            torch.from_numpy(np.asarray(y[rows], dtype=np.int64)),
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    # A stray size-1 last batch breaks BatchNorm in train mode; drop it only
    # when a full batch remains, so tiny smoke datasets still train.
    drop_last = shuffle and len(idx) > batch_size and len(idx) % batch_size == 1
    return DataLoader(
        range(len(idx)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
        num_workers=0,
        collate_fn=collate,
    )


def _predict_classes(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(device)).argmax(dim=1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | None = None,
) -> float:
    device = device or torch.device("cpu")
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        total_n += len(yb)
    return total_loss / max(total_n, 1)


def _checkpoint_name(cfg: TrainConfig) -> str:
    if cfg.split == "cross-subject":
        return f"{cfg.model}_cross-subject_{cfg.test_subject}_{cfg.seed}.pt"
    return f"{cfg.model}_{cfg.split}_{cfg.seed}.pt"


def _model_config(cfg: TrainConfig, X: np.ndarray, n_classes: int) -> dict:
    config = {
        "n_rx": int(X.shape[1]),
        "n_time": int(X.shape[2]),
        "n_subcarriers": int(X.shape[3]),
        "n_classes": n_classes,
    }
    if cfg.model == "cnn_lstm":
        config["chunk_len"] = _DEFAULT_CHUNK_LEN
    return config


def _trained_preprocess(ds: HarDataset) -> dict:
    """{"preprocess": cfg dict} from the dataset's provenance, or {}."""
    if not ds.provenance:
        return {}
    return {"preprocess": json.loads(ds.provenance)["preprocess"]}


def train_model(ds: HarDataset, cfg: TrainConfig) -> tuple[MetricsReport, Path]:
    """Train per cfg, evaluate ONCE on test, persist checkpoint + results row.

    Returns (test MetricsReport, checkpoint path).
    """
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"unknown model {cfg.model!r}; expected {MODEL_NAMES}")
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    _seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"device: {device}")
    # Results rows must say which device ran, not "auto" - CPU and MPS give
    # different accuracy on identical data and seeds, so an unresolved
    # config column makes the row uncomparable to any other row.
    cfg = replace(cfg, device=str(device))

    train_idx, test_idx, split_desc = _resolve_split(ds, cfg)
    y_train = ds.y[train_idx]
    if cfg.split == "random":
        fit_pos, val_pos = _stratified_val_split(y_train, cfg.val_fraction, cfg.seed)
    else:
        # Regime-matched early stopping: hold out whole sessions for val so
        # model selection sees the same shift as the cross-* test split.
        fit_pos, val_pos = _grouped_val_split(
            y_train, ds.sessions[train_idx], cfg.val_fraction, cfg.seed
        )
    fit_idx = train_idx[fit_pos]
    val_idx = train_idx[val_pos]
    if len(val_idx) == 0:
        warnings.warn("validation set is empty; early stopping uses fit windows")
        val_idx = fit_idx

    # Train-only normalization stats; val/test are transformed, never fitted.
    mean, std = norm_fit(ds.X, fit_idx)
    loader = partial(_make_loader, ds.X, ds.y, mean=mean, std=std,
                     batch_size=cfg.batch_size, seed=cfg.seed)
    train_loader = loader(fit_idx, shuffle=True)
    val_loader = loader(val_idx, shuffle=False)
    test_loader = loader(test_idx, shuffle=False)

    n_classes = len(ds.label_names)
    model_config = _model_config(cfg, ds.X, n_classes)
    model = build_model(cfg.model, **model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    weights = _class_weights(ds.y[fit_idx], n_classes) if cfg.class_weighted else None
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device)
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_best = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_pred = _predict_classes(model, val_loader, device)
        val_f1 = compute_metrics(ds.y[val_idx], val_pred, ds.label_names).macro_f1
        print(
            f"epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss {train_loss:.4f}  val_macro_f1 {val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                print(f"early stop at epoch {epoch} (patience {cfg.patience})")
                break

    model.load_state_dict(best_state)
    test_pred = _predict_classes(model, test_loader, device)
    report = compute_metrics(ds.y[test_idx], test_pred, ds.label_names)

    coverage = split_coverage_note(ds, train_idx, test_idx)
    if coverage:
        warnings.warn(f"{coverage} - this run's metrics are uninterpretable")
    # Separate severity: an unpopulated class does not invalidate the run, it
    # lowers the best macro-F1 the run could possibly reach.
    ceiling = label_coverage_note(ds)
    if ceiling:
        warnings.warn(
            f"{ceiling} - macro-F1 is not comparable to runs where every "
            "class has data"
        )

    # Checkpoints must load on any machine — the realtime engine and the
    # dashboard run on the laptop's CPU, not wherever training happened.
    model = model.to("cpu")
    checkpoint_path = save_checkpoint(
        Path(cfg.checkpoints_dir) / _checkpoint_name(cfg),
        model=model,
        model_name=cfg.model,
        label_names=ds.label_names,
        norm_mean=mean,
        norm_std=std,
        # Serving must preprocess exactly as training did (detrend window,
        # etc.); realtime/dashboard read this back.
        config={**model_config, **_trained_preprocess(ds)},
    )
    append_result(
        Path(cfg.results_csv),
        model=cfg.model,
        split=split_desc,
        seed=cfg.seed,
        config=cfg.to_json(),
        report=report,
        n_train=len(train_idx),
        n_test=len(test_idx),
        notes="; ".join(
            part for part in (cfg.notes, coverage, ceiling) if part
        ),
    )
    figure_path = Path(cfg.figures_dir) / (
        f"cm_{cfg.model}_{split_desc.replace(':', '_')}.png"
    )
    save_confusion_matrix(
        ds.y[test_idx], test_pred, ds.label_names, figure_path,
        title=f"{cfg.model} / {split_desc} / seed {cfg.seed}",
    )

    falling = (
        "n/a" if report.falling_recall is None else f"{report.falling_recall:.4f}"
    )
    print(
        f"test [{split_desc}]  accuracy {report.accuracy:.4f}  "
        f"macro_f1 {report.macro_f1:.4f}  falling_recall {falling}  "
        f"(n_train {len(train_idx)}, n_test {len(test_idx)})"
    )
    print(f"checkpoint: {checkpoint_path}")
    return report, checkpoint_path


# ---------------------------------------------------------------------- CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m csihar.train",
        description="Train a CSI HAR deep model on an assembled dataset (.npz).",
    )
    parser.add_argument("--data", required=True, help="dataset .npz path")
    parser.add_argument("--model", choices=MODEL_NAMES, default="cnn")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="random")
    parser.add_argument(
        "--test-subject", default=None,
        help="held-out subject (required for --split cross-subject)",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--device", default=TrainConfig.device,
        help='"auto" (cuda > mps > cpu), or an explicit torch device such as '
        '"cuda", "mps", "cpu"',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = TrainConfig(
        model=args.model,
        split=args.split,
        test_subject=args.test_subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    ds = load_dataset(Path(args.data))
    train_model(ds, cfg)


if __name__ == "__main__":
    main()
