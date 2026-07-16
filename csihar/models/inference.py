"""Checkpoint contract + inference bundle — the Phase 5 real-time interface.

A checkpoint is a ``torch.save`` dict with EXACTLY these keys:

    {"state_dict", "model_name", "label_names", "norm_mean", "norm_std",
     "config"}

where ``config`` must carry ``n_rx``, ``n_time``, ``n_subcarriers``,
``n_classes`` (plus ``chunk_len`` for ``cnn_lstm``) so ``build_model`` can
rebuild the architecture sight-unseen. Normalization arrays are stored as
torch tensors so ``torch.load(..., weights_only=True)`` works — never widen
this format without updating the Phase 5 loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import MODEL_NAMES, build_model

REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "n_rx", "n_time", "n_subcarriers", "n_classes",
)
CHECKPOINT_KEYS: tuple[str, ...] = (
    "state_dict", "model_name", "label_names", "norm_mean", "norm_std", "config",
)


@dataclass(frozen=True)
class ModelBundle:
    """Everything the real-time system needs to turn raw windows into labels."""

    model: nn.Module              # rebuilt, weights loaded, eval mode
    model_name: str
    label_names: tuple[str, ...]
    norm_mean: np.ndarray         # (n_rx, n_subcarriers) float32
    norm_std: np.ndarray          # (n_rx, n_subcarriers) float32
    config: dict


def _validate_config(model_name: str, config: dict) -> None:
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"checkpoint config missing required keys: {missing}")
    if model_name == "cnn_lstm" and "chunk_len" not in config:
        raise ValueError("cnn_lstm checkpoint config must include 'chunk_len'")


def _validate_norm_shapes(
    norm_mean: np.ndarray, norm_std: np.ndarray, config: dict
) -> None:
    expected = (int(config["n_rx"]), int(config["n_subcarriers"]))
    for name, arr in (("norm_mean", norm_mean), ("norm_std", norm_std)):
        if tuple(arr.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected} (n_rx, n_subcarriers), "
                f"got {tuple(arr.shape)}"
            )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_name: str,
    label_names: tuple[str, ...],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    config: dict,
) -> Path:
    """Write a contract-conformant checkpoint; returns the path written."""
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"unknown model_name {model_name!r}; expected one of {MODEL_NAMES}"
        )
    _validate_config(model_name, config)
    norm_mean = np.asarray(norm_mean, dtype=np.float32)
    norm_std = np.asarray(norm_std, dtype=np.float32)
    _validate_norm_shapes(norm_mean, norm_std, config)
    if int(config["n_classes"]) != len(label_names):
        raise ValueError(
            f"config n_classes ({config['n_classes']}) != "
            f"len(label_names) ({len(label_names)})"
        )
    payload = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "label_names": list(label_names),
        # torch tensors, not numpy arrays: keeps weights_only=True loadable.
        "norm_mean": torch.from_numpy(norm_mean.copy()),
        "norm_std": torch.from_numpy(norm_std.copy()),
        "config": dict(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path) -> ModelBundle:
    """Rebuild the model from a checkpoint and return an eval-mode bundle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older/foreign checkpoints (e.g. numpy arrays inside) need the
        # permissive path; only ever load checkpoints you produced.
        payload = torch.load(path, map_location="cpu", weights_only=False)

    missing = [k for k in CHECKPOINT_KEYS if k not in payload]
    if missing:
        raise ValueError(f"checkpoint {path} missing keys: {missing}")
    model_name = str(payload["model_name"])
    if model_name not in MODEL_NAMES:
        raise ValueError(f"checkpoint {path} has unknown model {model_name!r}")
    config = dict(payload["config"])
    _validate_config(model_name, config)

    model = build_model(
        model_name,
        n_rx=int(config["n_rx"]),
        n_time=int(config["n_time"]),
        n_subcarriers=int(config["n_subcarriers"]),
        n_classes=int(config["n_classes"]),
        chunk_len=int(config.get("chunk_len", 25)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    norm_mean = np.asarray(
        torch.as_tensor(payload["norm_mean"]).numpy(), dtype=np.float32
    )
    norm_std = np.asarray(
        torch.as_tensor(payload["norm_std"]).numpy(), dtype=np.float32
    )
    _validate_norm_shapes(norm_mean, norm_std, config)
    return ModelBundle(
        model=model,
        model_name=model_name,
        label_names=tuple(str(name) for name in payload["label_names"]),
        norm_mean=norm_mean,
        norm_std=norm_std,
        config=config,
    )


def predict(bundle: ModelBundle, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Classify raw (un-normalized) windows.

    x: (n_rx, T, S) for one window or (B, n_rx, T, S) for a batch.
    Returns (labels, confidences) where confidences is the (B, n_classes)
    softmax distribution — each row sums to 1; labels are the argmax names.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError(
            f"expected (n_rx, T, S) or (B, n_rx, T, S), got shape {x.shape}"
        )
    n_rx = int(bundle.config["n_rx"])
    n_time = int(bundle.config["n_time"])
    n_sub = int(bundle.config["n_subcarriers"])
    if x.shape[1] != n_rx or x.shape[2] != n_time or x.shape[3] != n_sub:
        # The CNN's adaptive pooling accepts any T, so a wrong window length
        # would otherwise be scored silently on a shifted distribution.
        raise ValueError(
            f"expected n_rx={n_rx}, n_time={n_time}, n_subcarriers={n_sub}, "
            f"got shape {x.shape}"
        )
    z = (x - bundle.norm_mean[None, :, None, :]) / bundle.norm_std[None, :, None, :]
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(torch.from_numpy(z.astype(np.float32)))
        confidences = torch.softmax(logits, dim=1).numpy()
    labels = [bundle.label_names[i] for i in confidences.argmax(axis=1)]
    return labels, confidences
