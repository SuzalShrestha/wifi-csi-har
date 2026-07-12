"""Phase 6 figure regeneration — every report figure comes from this module.

Repo rule: no hand-made plots. ``make figures`` (-> ``python -m csihar.figures``)
reads the provenance CSVs under ``experiments/results/`` and regenerates every
defense figure deterministically into ``docs/figures/``. Missing CSVs are
skipped with a message, not an error, so the pipeline runs at any project
stage (before real data exists, most figures are simply skipped).

Figures produced (when their source CSVs exist):

- ``split_comparison.png``  — macro-F1 per model across the three splits
- ``falling_recall.png``    — falling recall per model across splits
- ``ablation_receivers.png`` / ``ablation_window.png`` / ``ablation_rate.png``
  — macro-F1 vs receiver count / window length / packet rate

CLI:
    python -m csihar.figures [--results-dir experiments/results]
                             [--out docs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("docs/figures")

# Model-results CSVs pooled for the split-comparison figures. dl.csv is
# train.py's default; baseline.csv is baseline.py's.
MODEL_RESULT_FILES: tuple[str, ...] = ("dl.csv", "baseline.csv")
ABLATION_NAMES: tuple[str, ...] = ("receivers", "window", "rate")
SPLIT_FAMILIES: tuple[str, ...] = ("random", "cross-session", "cross-subject")

_DPI = 150


# ------------------------------------------------------------------ parsing


def parse_notes(notes: str) -> dict[str, str]:
    """``"ablation=rate variant=50Hz"`` -> ``{"ablation": ..., "variant": ...}``.

    Tokens without ``=`` are ignored; empty/NaN notes give an empty dict.
    """
    if not isinstance(notes, str) or not notes.strip():
        return {}
    out: dict[str, str] = {}
    for token in notes.split():
        key, sep, value = token.partition("=")
        if sep and key and value:
            out[key] = value
    return out


def variant_value(ablation: str, variant: str) -> float:
    """Numeric x-axis value of an ablation variant name.

    receivers: ``"rx0+rx2"`` -> 2 (receiver count); window: ``"1.5s"`` -> 1.5;
    rate: ``"50Hz"`` -> 50.0.
    """
    if ablation == "receivers":
        return float(len(variant.split("+")))
    if ablation == "window":
        return float(variant.rstrip("s"))
    if ablation == "rate":
        return float(variant.rstrip("Hz").rstrip("hz"))
    raise ValueError(f"unknown ablation {ablation!r}; expected {ABLATION_NAMES}")


def split_family(split: str) -> str:
    """Collapse per-fold split labels: ``"cross-subject:s1"`` -> ``"cross-subject"``."""
    return split.partition(":")[0]


def load_results(paths: list[Path]) -> pd.DataFrame:
    """Concatenate existing results CSVs; adds ``split_fam``. Empty if none."""
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["split_fam"] = df["split"].astype(str).map(split_family)
    return df


# ----------------------------------------------------------------- plotting


def _agg(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """mean/std of ``value`` over seeds/folds, per (model, split_fam)."""
    grouped = (
        df.dropna(subset=[value])
        .groupby(["model", "split_fam"], sort=True)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _grouped_bars(
    stats: pd.DataFrame, ylabel: str, title: str, out_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(stats["model"].unique())
    families = [f for f in SPLIT_FAMILIES if f in set(stats["split_fam"])]
    x = np.arange(len(families), dtype=float)
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(models):
        sub = stats[stats["model"] == model].set_index("split_fam")
        means = [sub["mean"].get(f, np.nan) for f in families]
        stds = [sub["std"].get(f, 0.0) for f in families]
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width, means, width,
            yerr=stds, capsize=3, label=model,
        )
    ax.set_xticks(x, families)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def plot_split_comparison(df: pd.DataFrame, out_path: Path) -> Path | None:
    """Grouped bars: macro-F1 per model across split families. None if no data."""
    if df.empty or "macro_f1" not in df:
        return None
    stats = _agg(df, "macro_f1")
    if stats.empty:
        return None
    return _grouped_bars(
        stats, "macro-F1",
        "Model comparison across evaluation splits (mean ± std over seeds/folds)",
        out_path,
    )


def plot_falling_recall(df: pd.DataFrame, out_path: Path) -> Path | None:
    """Grouped bars: falling recall (safety-critical class) per model/split."""
    if df.empty or "falling_recall" not in df:
        return None
    stats = _agg(df, "falling_recall")
    if stats.empty:
        return None
    return _grouped_bars(
        stats, "falling recall",
        "Falling recall across evaluation splits (mean ± std over seeds/folds)",
        out_path,
    )


def plot_ablation(df: pd.DataFrame, ablation: str, out_path: Path) -> Path | None:
    """macro-F1 vs the ablation's numeric axis, one line per split family.

    Expects rows whose ``notes`` parse to this ablation (written by
    ``csihar.experiments``). Returns None if no matching rows.
    """
    if ablation not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {ablation!r}; expected {ABLATION_NAMES}")
    if df.empty or "notes" not in df:
        return None
    parsed = df["notes"].map(parse_notes)
    mask = parsed.map(lambda d: d.get("ablation")) == ablation
    sub = df[mask].copy()
    if sub.empty:
        return None
    sub["x"] = [
        variant_value(ablation, d["variant"]) for d in parsed[mask]
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlabel = {
        "receivers": "number of receivers",
        "window": "window length (s)",
        "rate": "packet rate (Hz)",
    }[ablation]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family in SPLIT_FAMILIES:
        fam = sub[sub["split_fam"] == family]
        if fam.empty:
            continue
        stats = (
            fam.groupby("x", sort=True)["macro_f1"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            stats["x"], stats["mean"], yerr=stats["std"].fillna(0.0),
            marker="o", capsize=3, label=family,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Ablation: {xlabel} (mean ± std over seeds/folds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------- driver


def generate_all(
    results_dir: Path = RESULTS_DIR, out_dir: Path = FIGURES_DIR
) -> dict[str, Path | None]:
    """Regenerate every figure whose source CSVs exist; report the rest as None."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    outcomes: dict[str, Path | None] = {}

    model_df = load_results([results_dir / name for name in MODEL_RESULT_FILES])
    outcomes["split_comparison"] = plot_split_comparison(
        model_df, out_dir / "split_comparison.png"
    )
    outcomes["falling_recall"] = plot_falling_recall(
        model_df, out_dir / "falling_recall.png"
    )
    for ablation in ABLATION_NAMES:
        df = load_results([results_dir / f"ablation_{ablation}.csv"])
        outcomes[f"ablation_{ablation}"] = plot_ablation(
            df, ablation, out_dir / f"ablation_{ablation}.png"
        )
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m csihar.figures",
        description="Regenerate all report figures from experiments/results/ CSVs.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    outcomes = generate_all(args.results_dir, args.out)
    for name, path in sorted(outcomes.items()):
        status = str(path) if path is not None else "skipped (no source data)"
        print(f"{name:<20} {status}")
    if all(path is None for path in outcomes.values()):
        print(
            f"nothing generated — no results CSVs under {args.results_dir} yet "
            "(run training/ablations first)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
