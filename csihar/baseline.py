"""Classical ML baseline — the bar every deep-learning model must beat.

Two hand-crafted-feature + sklearn pipelines (SVM-RBF, Random Forest) scored
through the shared ``csihar.evaluate`` module on all three mandatory splits.
Features are extracted once per window (pure function of that window, so no
leakage is possible); only the scaler/classifier is fit on train rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import HarDataset, iter_cross_subject, load_dataset, split_cross_session, split_random
from .evaluate import MetricsReport, append_result, compute_metrics, save_confusion_matrix
from .features import extract_features

DEFAULT_RESULTS_CSV = Path("experiments/results/baseline.csv")
DEFAULT_FIGURES_DIR = Path("docs/figures")


def _make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=-1
            )),
        ]),
    }


def _score_fold(
    model: Pipeline,
    features: np.ndarray,
    ds: HarDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, MetricsReport]:
    """Fit model on train rows only; return (y_true, y_pred, report)."""
    model.fit(features[train_idx], ds.y[train_idx])
    y_pred = model.predict(features[test_idx])
    y_true = ds.y[test_idx]
    report = compute_metrics(y_true, y_pred, ds.label_names)
    return y_true, y_pred, report


def run_baseline(
    ds: HarDataset,
    split: str,
    seed: int = 0,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[tuple[str, str, MetricsReport]]:
    """Run both baseline models on one split; log results + confusion matrices.

    split: "random", "cross-session", or "cross-subject" (LOSO — one row per
    held-out subject fold plus a pooled-metrics summary row).

    Returns a list of (model_name, split_desc, MetricsReport) tuples, one per
    row written (folds included for cross-subject).
    """
    results_csv = Path(results_csv)
    figures_dir = Path(figures_dir)
    features = extract_features(ds.X)
    models = _make_models(seed)

    rows: list[tuple[str, str, MetricsReport]] = []

    if split == "random":
        train_idx, test_idx = split_random(ds, seed=seed)
        for name, model in models.items():
            y_true, y_pred, report = _score_fold(
                model, features, ds, train_idx, test_idx
            )
            append_result(
                results_csv, model=name, split="random", seed=seed,
                config="baseline", report=report,
                n_train=len(train_idx), n_test=len(test_idx), notes="",
            )
            save_confusion_matrix(
                y_true, y_pred, ds.label_names,
                figures_dir / f"cm_{name}_random.png",
                title=f"{name} — random split",
            )
            rows.append((name, "random", report))

    elif split == "cross-session":
        train_idx, test_idx = split_cross_session(ds, seed=seed)
        for name, model in models.items():
            y_true, y_pred, report = _score_fold(
                model, features, ds, train_idx, test_idx
            )
            append_result(
                results_csv, model=name, split="cross-session", seed=seed,
                config="baseline", report=report,
                n_train=len(train_idx), n_test=len(test_idx), notes="",
            )
            save_confusion_matrix(
                y_true, y_pred, ds.label_names,
                figures_dir / f"cm_{name}_cross-session.png",
                title=f"{name} — cross-session split",
            )
            rows.append((name, "cross-session", report))

    elif split == "cross-subject":
        for name, base_model in models.items():
            all_true: list[np.ndarray] = []
            all_pred: list[np.ndarray] = []
            for subject, train_idx, test_idx in iter_cross_subject(ds):
                model = _make_models(seed)[name]
                y_true, y_pred, report = _score_fold(
                    model, features, ds, train_idx, test_idx
                )
                append_result(
                    results_csv, model=name, split="cross-subject", seed=seed,
                    config="baseline", report=report,
                    n_train=len(train_idx), n_test=len(test_idx),
                    notes=f"fold={subject}",
                )
                rows.append((name, f"cross-subject:{subject}", report))
                all_true.append(y_true)
                all_pred.append(y_pred)

            pooled_true = np.concatenate(all_true)
            pooled_pred = np.concatenate(all_pred)
            pooled_report = compute_metrics(pooled_true, pooled_pred, ds.label_names)
            append_result(
                results_csv, model=name, split="cross-subject", seed=seed,
                config="baseline", report=pooled_report,
                n_train=-1, n_test=len(pooled_true), notes="LOSO pooled",
            )
            save_confusion_matrix(
                pooled_true, pooled_pred, ds.label_names,
                figures_dir / f"cm_{name}_cross-subject.png",
                title=f"{name} — cross-subject (LOSO pooled)",
            )
            rows.append((name, "cross-subject:LOSO pooled", pooled_report))

    else:
        raise ValueError(
            f"unknown split {split!r}; expected 'random', 'cross-session', "
            "or 'cross-subject'"
        )

    return rows


def _print_summary(rows: list[tuple[str, str, MetricsReport]]) -> None:
    header = f"{'model':<15} {'split':<28} {'accuracy':>9} {'macro_f1':>9} {'falling_recall':>14}"
    print(header)
    print("-" * len(header))
    for model_name, split_desc, report in rows:
        falling = "" if report.falling_recall is None else f"{report.falling_recall:.4f}"
        print(
            f"{model_name:<15} {split_desc:<28} {report.accuracy:>9.4f} "
            f"{report.macro_f1:>9.4f} {falling:>14}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classical ML baseline for CSI-HAR")
    parser.add_argument("--data", type=Path, required=True, help="path to dataset.npz")
    parser.add_argument(
        "--split", choices=["random", "cross-session", "cross-subject", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args(argv)

    ds = load_dataset(args.data)
    splits = (
        ["random", "cross-session", "cross-subject"]
        if args.split == "all" else [args.split]
    )

    all_rows: list[tuple[str, str, MetricsReport]] = []
    for split in splits:
        all_rows.extend(
            run_baseline(
                ds, split, seed=args.seed,
                results_csv=args.results_csv, figures_dir=args.figures_dir,
            )
        )
    _print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
