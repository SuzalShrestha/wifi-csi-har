"""Phase 4 ablation runner: receiver count, window length, packet rate.

Defense-required ablations (IMPLEMENTATION_PLAN.md): how much does accuracy
degrade with 1 vs 2 vs 3 receivers, with shorter observation windows, and at
lower packet rates (100/50/25 Hz)? Every ablation is a PURE dataset
transform applied before training, so the discipline baked into
``train.train_model`` (train-only normalization, seeding, early stopping,
provenance CSV rows) is reused unchanged. The transform is recorded in the
results row's ``notes`` column, e.g. ``ablation=rate variant=50Hz``.

The window ablation can only SHORTEN windows (center-crop). Window lengths
longer than the assembled window require re-assembly from raw sessions with
a different ``PreprocessConfig(window_s=...)`` — out of scope here.

CLI:
    python -m csihar.experiments --data ds.npz --ablation receivers \
        --model cnn --split random --seeds 0 1 2 --epochs 30
    python -m csihar.experiments --config experiments/configs/ablation_rate.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from .dataset import HarDataset, iter_cross_subject, load_dataset
from .evaluate import MetricsReport
from .models import MODEL_NAMES
from .train import SPLIT_NAMES, TrainConfig, train_model

ABLATION_NAMES: tuple[str, ...] = ("receivers", "window", "rate", "all")
WINDOW_SECONDS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
RATE_FACTORS: tuple[int, ...] = (1, 2, 4)
BASE_RATE_HZ: float = 100.0
MIN_TIME_SAMPLES: int = 8

DEFAULT_CHECKPOINTS_DIR = "experiments/checkpoints/ablations"
DEFAULT_FIGURES_DIR = "docs/figures/ablations"


# ------------------------------------------------------ dataset transforms


def _with_x(ds: HarDataset, X: np.ndarray) -> HarDataset:
    """New HarDataset with a replaced X; metadata deep-copied, never shared."""
    return HarDataset(
        X=np.ascontiguousarray(X, dtype=np.float32),
        y=ds.y.copy(),
        subjects=ds.subjects.copy(),
        sessions=ds.sessions.copy(),
        environments=ds.environments.copy(),
        label_names=ds.label_names,
    )


def subset_receivers(ds: HarDataset, rx_indices: tuple[int, ...]) -> HarDataset:
    """New dataset keeping only the given receiver indices (axis 1 of X).

    Pure: the input dataset is never mutated. Order of ``rx_indices`` is
    preserved in the output.
    """
    n_rx = int(ds.X.shape[1])
    idx = tuple(int(i) for i in rx_indices)
    if not idx:
        raise ValueError("rx_indices must be non-empty")
    if len(set(idx)) != len(idx):
        raise ValueError(f"duplicate receiver indices: {idx}")
    bad = [i for i in idx if not 0 <= i < n_rx]
    if bad:
        raise ValueError(f"receiver indices out of range [0, {n_rx}): {bad}")
    return _with_x(ds, ds.X[:, list(idx), :, :])


def decimate_time(ds: HarDataset, factor: int) -> HarDataset:
    """New dataset with the time axis decimated by ``factor`` (X[:, :, ::factor]).

    Simulates a lower packet rate: with 100 Hz assembly, factor 1 -> 100 Hz,
    2 -> 50 Hz, 4 -> 25 Hz. Pure: the input dataset is never mutated.
    """
    if int(factor) != factor or factor < 1:
        raise ValueError(f"factor must be an integer >= 1, got {factor!r}")
    factor = int(factor)
    n_time = int(ds.X.shape[2])
    new_t = -(-n_time // factor)  # ceil division
    if new_t < MIN_TIME_SAMPLES:
        raise ValueError(
            f"factor {factor} leaves {new_t} time samples "
            f"(< minimum {MIN_TIME_SAMPLES}) from T={n_time}"
        )
    return _with_x(ds, ds.X[:, :, ::factor, :])


def crop_window(ds: HarDataset, seconds: float, fs: float = BASE_RATE_HZ) -> HarDataset:
    """New dataset with the time axis CENTER-cropped to ``round(seconds*fs)``.

    Simulates a shorter observation window. Pure: the input dataset is never
    mutated.

    NOTE: window lengths LONGER than the assembled window cannot be produced
    here — that requires re-assembly from raw sessions with a different
    ``PreprocessConfig(window_s=...)``, which is out of scope for this module.
    """
    if seconds <= 0.0:
        raise ValueError(f"seconds must be > 0, got {seconds!r}")
    if fs <= 0.0:
        raise ValueError(f"fs must be > 0, got {fs!r}")
    n = int(round(seconds * fs))
    n_time = int(ds.X.shape[2])
    if n < 1:
        raise ValueError(f"{seconds}s @ {fs}Hz rounds to zero samples")
    if n > n_time:
        raise ValueError(
            f"requested {n} samples ({seconds}s @ {fs}Hz) exceeds the assembled "
            f"window T={n_time}; re-assemble from raw sessions with "
            f"PreprocessConfig(window_s={seconds}) instead"
        )
    start = (n_time - n) // 2
    return _with_x(ds, ds.X[:, :, start:start + n, :])


# ----------------------------------------------------- variant enumeration


def receiver_variants(n_rx: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """(name, rx_indices) for every non-empty receiver subset (7 for n_rx=3)."""
    if n_rx < 1:
        raise ValueError(f"n_rx must be >= 1, got {n_rx}")
    out: list[tuple[str, tuple[int, ...]]] = []
    for size in range(1, n_rx + 1):
        for combo in itertools.combinations(range(n_rx), size):
            out.append(("+".join(f"rx{i}" for i in combo), combo))
    return tuple(out)


def window_variants(
    n_time: int, fs: float = BASE_RATE_HZ
) -> tuple[tuple[str, float], ...]:
    """(name, seconds) for each candidate window length that fits in n_time."""
    out = [
        (f"{seconds:g}s", seconds)
        for seconds in WINDOW_SECONDS
        if int(round(seconds * fs)) <= n_time
    ]
    if not out:
        raise ValueError(
            f"no candidate window length {WINDOW_SECONDS} fits in T={n_time} @ {fs}Hz"
        )
    return tuple(out)


def rate_variants() -> tuple[tuple[str, int], ...]:
    """(name, decimation factor) for the packet-rate ablation: 100/50/25 Hz."""
    return tuple((f"{BASE_RATE_HZ / f:g}Hz", f) for f in RATE_FACTORS)


_Transform = Callable[[HarDataset], HarDataset]


def _variant_specs(
    ds: HarDataset, ablation: str
) -> tuple[tuple[str, str, _Transform], ...]:
    """(ablation, variant_name, transform) for every variant of ``ablation``."""
    if ablation == "receivers":
        return tuple(
            ("receivers", name, partial(subset_receivers, rx_indices=combo))
            for name, combo in receiver_variants(int(ds.X.shape[1]))
        )
    if ablation == "window":
        return tuple(
            ("window", name, partial(crop_window, seconds=seconds))
            for name, seconds in window_variants(int(ds.X.shape[2]))
        )
    if ablation == "rate":
        return tuple(
            ("rate", name, partial(decimate_time, factor=factor))
            for name, factor in rate_variants()
        )
    if ablation == "all":
        return tuple(
            spec
            for sub in ("receivers", "window", "rate")
            for spec in _variant_specs(ds, sub)
        )
    raise ValueError(f"unknown ablation {ablation!r}; expected one of {ABLATION_NAMES}")


# ------------------------------------------------------------------ driver


class AblationResult(NamedTuple):
    """One trained variant run. First two fields are (variant_name, report)."""

    variant: str
    report: MetricsReport
    ablation: str
    split: str
    seed: int
    subject: str | None


def run_ablation(
    ds: HarDataset,
    *,
    ablation: str,
    model: str,
    split: str,
    seeds: tuple[int, ...],
    epochs: int,
    results_csv: str | Path | None = None,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    variants: tuple[str, ...] | None = None,
) -> list[AblationResult]:
    """Train every variant x seed of one ablation; return per-run reports.

    Each run appends a provenance row via ``train_model`` to ``results_csv``
    (default ``experiments/results/ablation_<name>.csv`` per ablation) with
    the variant encoded in the notes column. Checkpoints and figures land in
    per-variant subdirectories so variants never overwrite each other. For
    split "cross-subject" every variant loops LOSO over all subjects.

    ``variants`` optionally restricts runs to the named variants (useful for
    smoke tests); unknown names raise before any training starts.
    """
    if ablation not in ABLATION_NAMES:
        raise ValueError(
            f"unknown ablation {ablation!r}; expected one of {ABLATION_NAMES}"
        )
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    specs = _variant_specs(ds, ablation)
    if variants is not None:
        available = {name for _, name, _ in specs}
        unknown = set(variants) - available
        if unknown:
            raise ValueError(
                f"unknown variant names {sorted(unknown)}; "
                f"available: {sorted(available)}"
            )
        specs = tuple(spec for spec in specs if spec[1] in variants)

    results: list[AblationResult] = []
    for abl, name, transform in specs:
        ds_variant = transform(ds)
        csv_path = (
            Path(results_csv)
            if results_csv is not None
            else Path(f"experiments/results/ablation_{abl}.csv")
        )
        for seed in seeds:
            base_cfg = TrainConfig(
                model=model,
                split=split,
                epochs=epochs,
                seed=int(seed),
                results_csv=str(csv_path),
                checkpoints_dir=str(Path(checkpoints_dir) / abl / name),
                figures_dir=str(Path(figures_dir) / abl / name),
                notes=f"ablation={abl} variant={name}",
            )
            if split == "cross-subject":
                for subject, _, _ in iter_cross_subject(ds_variant):
                    cfg = replace(base_cfg, test_subject=subject)
                    report, _ = train_model(ds_variant, cfg)
                    results.append(
                        AblationResult(name, report, abl, split, int(seed), subject)
                    )
            else:
                report, _ = train_model(ds_variant, base_cfg)
                results.append(
                    AblationResult(name, report, abl, split, int(seed), None)
                )
    return results


def format_summary(results: list[AblationResult]) -> str:
    """Compact fixed-width summary table of a list of ablation runs."""
    header = (
        f"{'ablation':<10} {'variant':<14} {'split':<14} {'seed':>4} "
        f"{'subject':<8} {'accuracy':>8} {'macro_f1':>8} {'falling_recall':>14}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        falling = (
            "n/a" if r.report.falling_recall is None
            else f"{r.report.falling_recall:.4f}"
        )
        lines.append(
            f"{r.ablation:<10} {r.variant:<14} {r.split:<14} {r.seed:>4} "
            f"{(r.subject or '-'):<8} {r.report.accuracy:>8.4f} "
            f"{r.report.macro_f1:>8.4f} {falling:>14}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI


_CONFIG_KEYS = frozenset(
    {"data", "ablation", "model", "split", "splits", "seeds", "epochs",
     "results_csv", "checkpoints_dir", "figures_dir", "variants"}
)


def load_config_file(path: str | Path) -> dict:
    """Load and validate a JSON ablation config (see experiments/configs/)."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a JSON object")
    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"config {path} has unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_CONFIG_KEYS)}"
        )
    return raw


def resolve_settings(args: argparse.Namespace) -> dict:
    """Merge config file (if any) with CLI flags; CLI flags win. Validates."""
    cfg = load_config_file(args.config) if args.config else {}
    cfg_splits = cfg.get("splits") or ([cfg["split"]] if cfg.get("split") else None)
    settings = {
        "data": args.data or cfg.get("data"),
        "ablation": args.ablation or cfg.get("ablation"),
        "model": args.model or cfg.get("model") or "cnn",
        "splits": tuple([args.split] if args.split else (cfg_splits or ["random"])),
        "seeds": tuple(
            int(s) for s in (args.seeds if args.seeds else cfg.get("seeds", (0,)))
        ),
        "epochs": int(
            args.epochs if args.epochs is not None
            else cfg.get("epochs", TrainConfig.epochs)
        ),
        "results_csv": args.results_csv or cfg.get("results_csv"),
        "checkpoints_dir": (
            args.checkpoints_dir or cfg.get("checkpoints_dir")
            or DEFAULT_CHECKPOINTS_DIR
        ),
        "figures_dir": args.figures_dir or cfg.get("figures_dir")
        or DEFAULT_FIGURES_DIR,
        "variants": (
            tuple(args.variants) if args.variants
            else (tuple(cfg["variants"]) if cfg.get("variants") else None)
        ),
    }
    if not settings["data"]:
        raise ValueError("--data (or 'data' in --config) is required")
    if not settings["ablation"]:
        raise ValueError("--ablation (or 'ablation' in --config) is required")
    if settings["ablation"] not in ABLATION_NAMES:
        raise ValueError(
            f"unknown ablation {settings['ablation']!r}; expected {ABLATION_NAMES}"
        )
    if settings["model"] not in MODEL_NAMES:
        raise ValueError(
            f"unknown model {settings['model']!r}; expected {MODEL_NAMES}"
        )
    for split in settings["splits"]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected {SPLIT_NAMES}")
    if not settings["seeds"]:
        raise ValueError("seeds must be non-empty")
    if settings["epochs"] < 1:
        raise ValueError("epochs must be >= 1")
    return settings


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m csihar.experiments",
        description=(
            "Run defense-required ablations (receiver count, window length, "
            "packet rate) over an assembled dataset (.npz)."
        ),
    )
    parser.add_argument("--config", default=None,
                        help="JSON config file (CLI flags override its values)")
    parser.add_argument("--data", default=None, help="dataset .npz path")
    parser.add_argument("--ablation", choices=ABLATION_NAMES, default=None)
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--split", choices=SPLIT_NAMES, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--results-csv", default=None,
                        help="default: experiments/results/ablation_<name>.csv")
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="restrict to these variant names (e.g. rx0 50Hz)")
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = resolve_settings(_build_arg_parser().parse_args(argv))
    ds = load_dataset(Path(settings["data"]))
    results: list[AblationResult] = []
    for split in settings["splits"]:
        results.extend(
            run_ablation(
                ds,
                ablation=settings["ablation"],
                model=settings["model"],
                split=split,
                seeds=settings["seeds"],
                epochs=settings["epochs"],
                results_csv=settings["results_csv"],
                checkpoints_dir=settings["checkpoints_dir"],
                figures_dir=settings["figures_dir"],
                variants=settings["variants"],
            )
        )
    print()
    print(format_summary(results))


if __name__ == "__main__":
    main()
