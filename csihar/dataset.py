"""Session -> tensor assembly and the three mandatory evaluation splits.

This is the leakage-critical core of the project. Every published number
flows through here, so the rules are enforced in code, not convention:

- window labels come from ``labels.json`` segments (transition-trimmed) or,
  absent that, from the session's ``metadata.json`` label;
- ``split_cross_session`` guarantees no session straddles train/test;
- ``split_cross_subject`` guarantees no subject straddles train/test;
- ``split_random`` is provided ONLY for literature comparability and its
  docstring says so.

Scaler fitting stays in ``preprocessing.normalize`` — fit on train indices
only.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .preprocessing import PreprocessConfig, align_receivers, preprocess_stream
from .preprocessing.subcarriers import usable_lltf_indices
from .storage import load_amplitudes

# Fixed label order — index in this tuple IS the integer class id everywhere.
LABEL_NAMES: tuple[str, ...] = (
    "background", "standing", "sitting", "lying", "walking", "falling",
)
_LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}


@dataclass(frozen=True)
class Segment:
    """One labeled time span from labels.json (host-clock epoch seconds)."""

    label: str
    start_ts: float
    end_ts: float


@dataclass(frozen=True)
class SessionWindows:
    """Aligned windows of one session: X (n, n_rx, T, S), y (n,) class ids."""

    X: np.ndarray
    y: np.ndarray
    subject: str
    environment: str
    session: str


@dataclass(frozen=True)
class HarDataset:
    """Assembled dataset. X float32 (n, n_rx, T, S); y int64 class ids."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray      # (n,) str
    sessions: np.ndarray      # (n,) str
    environments: np.ndarray  # (n,) str
    label_names: tuple[str, ...] = field(default=LABEL_NAMES)

    def __post_init__(self) -> None:
        if self.X.ndim != 4:
            raise ValueError(f"X must be (n, n_rx, T, S), got shape {self.X.shape}")
        n = self.X.shape[0]
        for name, arr in (
            ("y", self.y), ("subjects", self.subjects),
            ("sessions", self.sessions), ("environments", self.environments),
        ):
            if arr.ndim != 1 or len(arr) != n:
                raise ValueError(f"{name} must be 1-D with length {n}")


# ---------------------------------------------------------------- assembly


def _load_segments(session_dir: Path) -> list[Segment] | None:
    """Parse labels.json if present; validate labels. None if absent."""
    path = session_dir / "labels.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    segments = []
    for seg in raw.get("segments", []):
        label = seg["label"]
        if label not in _LABEL_TO_ID:
            raise ValueError(
                f"session {session_dir.name}: unknown label {label!r} in labels.json"
            )
        segments.append(
            Segment(label=label, start_ts=float(seg["start_ts"]),
                    end_ts=float(seg["end_ts"]))
        )
    return segments


def _segment_label(
    segments: list[Segment], start_ts: float, window_s: float, margin_s: float
) -> str | None:
    """Label for a window fully inside a trimmed segment; None -> drop it."""
    end_ts = start_ts + window_s
    for seg in segments:
        if start_ts >= seg.start_ts + margin_s and end_ts <= seg.end_ts - margin_s:
            return seg.label
    return None


def assemble_session(session_dir: Path, cfg: PreprocessConfig) -> SessionWindows:
    """Load, preprocess, align, and label all receivers of one session.

    Window time is the FIRST (sorted) receiver's window start_ts. With
    labels.json, windows not fully inside any margin-trimmed segment are
    dropped (transition trimming); otherwise metadata.json's label applies
    to every window.
    """
    meta = json.loads((session_dir / "metadata.json").read_text())
    session_label = meta.get("label", "")
    segments = _load_segments(session_dir)
    if segments is None and session_label not in _LABEL_TO_ID:
        raise ValueError(
            f"session {session_dir.name}: unknown label {session_label!r}"
        )

    parquets = sorted(session_dir.glob("*.parquet"))
    if not parquets:
        raise ValueError(f"session {session_dir.name}: no receiver parquet files")
    streams = {}
    for pq in parquets:
        host_ts, amps = load_amplitudes(pq)
        streams[pq.stem] = preprocess_stream(host_ts, amps, cfg)

    rx_ids = sorted(streams)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for group in align_receivers(streams, tolerance_s=cfg.align_tolerance_s):
        start_ts = group[rx_ids[0]].start_ts
        if segments is None:
            label = session_label
        else:
            label = _segment_label(
                segments, start_ts, cfg.window_s, cfg.boundary_margin_s
            )
            if label is None:
                continue  # gap / straddles a boundary -> dropped
        xs.append(np.stack([group[rx].values for rx in rx_ids]))
        ys.append(_LABEL_TO_ID[label])

    if xs:
        X = np.stack(xs).astype(np.float32)
    else:
        n_samples = int(round(cfg.fs * cfg.window_s))
        X = np.empty(
            (0, len(rx_ids), n_samples, len(usable_lltf_indices())),
            dtype=np.float32,
        )
    return SessionWindows(
        X=X,
        y=np.asarray(ys, dtype=np.int64),
        subject=str(meta.get("subject", "")),
        environment=str(meta.get("environment", "")),
        session=session_dir.name,
    )


def is_excluded(session_dir: Path) -> bool:
    """True if the session's metadata marks it as not-for-training.

    Bring-up and rig-test captures sit in datasets/raw next to real sessions.
    Relying on whoever runs training to remember which is which is how an
    uncontrolled room ends up contributing windows to a reported result.
    """
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return False
    return bool(json.loads(meta_path.read_text()).get("exclude_from_dataset", False))


def assemble_dataset(raw_dir: Path, cfg: PreprocessConfig) -> HarDataset:
    """Assemble every session dir under raw_dir (those with metadata.json).

    Sessions marked ``exclude_from_dataset`` are left out. Sessions yielding
    zero windows are skipped and reported in one warning.
    """
    all_dirs = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )
    excluded = [d.name for d in all_dirs if is_excluded(d)]
    session_dirs = [d for d in all_dirs if not is_excluded(d)]
    if excluded:
        warnings.warn(f"excluded from dataset by metadata: {excluded}")
    if not session_dirs:
        raise ValueError(f"no session dirs with metadata.json under {raw_dir}")

    parts: list[SessionWindows] = []
    skipped: list[str] = []
    for session_dir in session_dirs:
        part = assemble_session(session_dir, cfg)
        if part.X.shape[0] == 0:
            skipped.append(part.session)
        else:
            parts.append(part)
    if skipped:
        warnings.warn(f"sessions yielded no windows and were skipped: {skipped}")
    if not parts:
        raise ValueError(f"no windows assembled from any session under {raw_dir}")

    def tag(value_of) -> np.ndarray:
        return np.concatenate(
            [np.full(len(p.y), value_of(p)) for p in parts]
        ).astype(str)

    return HarDataset(
        X=np.concatenate([p.X for p in parts]).astype(np.float32),
        y=np.concatenate([p.y for p in parts]).astype(np.int64),
        subjects=tag(lambda p: p.subject),
        sessions=tag(lambda p: p.session),
        environments=tag(lambda p: p.environment),
        label_names=LABEL_NAMES,
    )


# ------------------------------------------------------------- persistence


def save_dataset(ds: HarDataset, path: Path) -> Path:
    """Save as compressed npz (strings as fixed-width unicode, npz-safe)."""
    np.savez_compressed(
        path,
        X=ds.X,
        y=ds.y,
        subjects=np.asarray(ds.subjects, dtype=str),
        sessions=np.asarray(ds.sessions, dtype=str),
        environments=np.asarray(ds.environments, dtype=str),
        label_names=np.array(ds.label_names, dtype=str),
    )
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load_dataset(path: Path) -> HarDataset:
    with np.load(path) as z:
        return HarDataset(
            X=z["X"],
            y=z["y"].astype(np.int64),
            subjects=z["subjects"].astype(str),
            sessions=z["sessions"].astype(str),
            environments=z["environments"].astype(str),
            label_names=tuple(str(name) for name in z["label_names"]),
        )


# ------------------------------------------------------------------ splits


def split_random(
    ds: HarDataset, test_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random window split (per-class shuffle then slice).

    WARNING — this split contains temporal leakage BY DESIGN: overlapping
    windows from the same activity bout land on both sides, which inflates
    accuracy. It exists ONLY for comparability with the CSI-HAR literature.
    Never report it alone; cross-session and cross-subject are the honest
    numbers.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for cls in np.unique(ds.y):
        idx = rng.permutation(np.where(ds.y == cls)[0])
        n_test = int(round(len(idx) * test_fraction))
        n_test = min(max(n_test, 1 if len(idx) > 1 else 0), len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    test_idx = np.sort(np.concatenate(test_parts)).astype(np.int64)
    return train_idx, test_idx


def split_coverage_note(
    ds: HarDataset, train_idx: np.ndarray, test_idx: np.ndarray
) -> str:
    """Describe degenerate class coverage in a split, or "" if it is sound.

    A test set missing classes the training set contains scores the model on a
    different problem than it was fit for, and the resulting accuracy is
    uninterpretable rather than bad. With few sessions this is easy to hit —
    holding out whole sessions can remove every window of several classes —
    and a bare number in the results CSV then reads as catastrophic failure.
    """
    train_classes = set(np.unique(ds.y[train_idx]).tolist())
    test_classes = set(np.unique(ds.y[test_idx]).tolist())
    missing_from_test = train_classes - test_classes
    missing_from_train = test_classes - train_classes

    parts = []
    if missing_from_test:
        names = ", ".join(sorted(ds.label_names[i] for i in missing_from_test))
        parts.append(f"test missing {len(missing_from_test)} class(es): {names}")
    if missing_from_train:
        names = ", ".join(sorted(ds.label_names[i] for i in missing_from_train))
        parts.append(f"train missing {len(missing_from_train)} class(es): {names}")
    return "DEGENERATE SPLIT - " + "; ".join(parts) if parts else ""


def split_cross_session(
    ds: HarDataset, test_fraction: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole sessions (~test_fraction of windows), greedily chosen.

    Preferences: (a) every held-out session's subject also appears in train
    via other sessions where possible, (b) every class present in test is
    present in train (hard requirement). No session straddles the split.
    """
    unique_sessions = np.unique(ds.sessions)
    if len(unique_sessions) < 2:
        raise ValueError("cross-session split needs at least 2 sessions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_sessions)
    target = test_fraction * len(ds.y)

    def feasible(candidate: set[str], require_subject: bool) -> bool:
        test_mask = np.isin(ds.sessions, list(candidate))
        if test_mask.all():
            return False
        train_mask = ~test_mask
        if not set(ds.y[test_mask]) <= set(ds.y[train_mask]):
            return False
        if require_subject:
            return set(ds.subjects[test_mask]) <= set(ds.subjects[train_mask])
        return True

    test_sessions: set[str] = set()
    for require_subject in (True, False):
        for session in order:
            count = int(np.isin(ds.sessions, list(test_sessions)).sum())
            if test_sessions and count >= target:
                break
            candidate = test_sessions | {str(session)}
            if feasible(candidate, require_subject):
                test_sessions = candidate
        if test_sessions:
            break
    if not test_sessions:
        raise ValueError("no feasible held-out session set found")

    test_mask = np.isin(ds.sessions, list(test_sessions))
    train_idx = np.where(~test_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)
    assert not set(ds.sessions[train_idx]) & set(ds.sessions[test_idx])
    return train_idx, test_idx


def split_cross_subject(
    ds: HarDataset, test_subject: str
) -> tuple[np.ndarray, np.ndarray]:
    """All windows of test_subject vs everyone else (one LOSO fold)."""
    known = set(ds.subjects.tolist())
    if test_subject not in known:
        raise ValueError(f"unknown subject {test_subject!r}; have {sorted(known)}")
    test_mask = ds.subjects == test_subject
    if test_mask.all():
        raise ValueError(f"holding out {test_subject!r} would leave train empty")
    train_idx = np.where(~test_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)
    assert not set(ds.subjects[train_idx]) & set(ds.subjects[test_idx])
    return train_idx, test_idx


def split_cross_environment(
    ds: HarDataset, test_environment: str
) -> tuple[np.ndarray, np.ndarray]:
    """All windows of one environment vs the rest — the second-room transfer
    test the collection plan calls for. Same guarantees as the other group
    splits: no environment straddles the split."""
    known = set(ds.environments.tolist())
    if test_environment not in known:
        raise ValueError(
            f"unknown environment {test_environment!r}; have {sorted(known)}"
        )
    test_mask = ds.environments == test_environment
    if test_mask.all():
        raise ValueError(
            f"holding out {test_environment!r} would leave train empty"
        )
    train_idx = np.where(~test_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)
    assert not set(ds.environments[train_idx]) & set(ds.environments[test_idx])
    return train_idx, test_idx


def iter_cross_subject(
    ds: HarDataset,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (subject, train_idx, test_idx) for every subject — LOSO loop."""
    for subject in sorted(set(ds.subjects.tolist())):
        train_idx, test_idx = split_cross_subject(ds, subject)
        yield subject, train_idx, test_idx
