"""Shared evaluation and results reporting — the single scoring path.

Every model (classical baseline, CNN, CNN-LSTM, ...) reports through this
module so numbers are computed identically everywhere. Policy encoded here:
macro-F1 is the headline metric, falling recall is always broken out
(safety-critical class), and every results row carries enough provenance
(config id, split, seed, git SHA, timestamp) to regenerate it.
"""

from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("docs/figures")


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricsReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    falling_recall: float | None  # None if 'falling' absent from y_true


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: tuple[str, ...]
) -> MetricsReport:
    labels = np.arange(len(label_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: ClassMetrics(
            precision=float(precision[i]), recall=float(recall[i]),
            f1=float(f1[i]), support=int(support[i]),
        )
        for i, name in enumerate(label_names)
    }
    falling_recall = None
    if "falling" in label_names:
        idx = label_names.index("falling")
        if (np.asarray(y_true) == idx).any():
            falling_recall = float(recall[idx])
    return MetricsReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                zero_division=0)),
        per_class=per_class,
        falling_recall=falling_recall,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: tuple[str, ...],
    out_path: Path,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.arange(len(label_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, label_names, rotation=45, ha="right")
    ax.set_yticks(labels, label_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


RESULT_COLUMNS = (
    "timestamp", "git_sha", "model", "split", "seed", "config",
    "accuracy", "macro_f1", "falling_recall",
    "n_train", "n_test", "notes",
)


def append_result(
    csv_path: Path,
    *,
    model: str,
    split: str,
    seed: int,
    config: str,
    report: MetricsReport,
    n_train: int,
    n_test: int,
    notes: str = "",
) -> Path:
    """Append one provenance-complete results row; creates file + header."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "model": model,
        "split": split,
        "seed": seed,
        "config": config,
        "accuracy": f"{report.accuracy:.4f}",
        "macro_f1": f"{report.macro_f1:.4f}",
        "falling_recall": (
            "" if report.falling_recall is None else f"{report.falling_recall:.4f}"
        ),
        "n_train": n_train,
        "n_test": n_test,
        "notes": notes,
    }
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return csv_path
