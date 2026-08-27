"""Dataset progress report: what has been collected, against the M3 targets.

Milestone M3 requires a frozen dataset with a datasheet — counts per class,
subject, session and environment. Collection runs over weeks across four
people, so the useful form of that datasheet is a *running* one that says what
is still missing while there is still time to record it.

Reads only labels.json / metadata.json and the Parquet timestamps, so it stays
fast enough to run after every session (it does not preprocess or window the
data). Durations are exact; window counts are the estimate implied by the
configured window/hop.

    python -m csihar.datasheet datasets/raw
    python -m csihar.datasheet datasets/raw --markdown docs/datasheet.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .dataset import LABEL_NAMES, is_excluded
from .preprocessing import PreprocessConfig

__all__ = ["SessionSummary", "summarize_session", "scan", "Targets", "main"]


@dataclass(frozen=True)
class Targets:
    """M3 acceptance targets from IMPLEMENTATION_PLAN Phase 3."""

    subjects: int = 8
    sessions_per_subject: int = 3
    seconds_per_activity_per_subject: float = 300.0
    total_windows: int = 25_000
    environments: int = 2


@dataclass(frozen=True)
class SessionSummary:
    """One session's contribution, in seconds of labelled data per class."""

    name: str
    subject: str
    environment: str
    date: str
    seconds_by_label: dict[str, float]

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds_by_label.values())


def summarize_session(session_dir: Path) -> SessionSummary | None:
    """Read one session's labelled duration per class. None if unreadable."""
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists() or is_excluded(session_dir):
        return None
    meta = json.loads(meta_path.read_text())

    labels_path = session_dir / "labels.json"
    seconds: dict[str, float] = defaultdict(float)
    if labels_path.exists():
        for seg in json.loads(labels_path.read_text())["segments"]:
            seconds[seg["label"]] += float(seg["end_ts"]) - float(seg["start_ts"])
    else:
        label = meta.get("label", "")
        if label not in LABEL_NAMES:
            return None  # e.g. label="scripted" with no labels.json -> unusable
        seconds[label] = _parquet_duration(session_dir)

    return SessionSummary(
        name=session_dir.name,
        subject=str(meta.get("subject", "")),
        environment=str(meta.get("environment", "")),
        date=session_dir.name[:8],
        seconds_by_label=dict(seconds),
    )


def _parquet_duration(session_dir: Path) -> float:
    """Longest receiver span in seconds; 0.0 if there are no parquet files."""
    spans = []
    for path in sorted(session_dir.glob("*.parquet")):
        ts = pd.read_parquet(path, columns=["host_ts"])["host_ts"]
        if len(ts) > 1:
            spans.append(float(ts.max() - ts.min()))
    return max(spans) if spans else 0.0


def scan(raw_dir: Path) -> list[SessionSummary]:
    """Summarize every session directory under `raw_dir`, oldest first."""
    summaries = [
        summarize_session(d) for d in sorted(raw_dir.iterdir()) if d.is_dir()
    ]
    return [s for s in summaries if s is not None]


def _estimated_windows(seconds: float, cfg: PreprocessConfig) -> int:
    """Windows a span of clean data yields at the configured window/hop."""
    usable = seconds - cfg.window_s
    return max(0, int(usable / cfg.hop_s) + 1) if usable >= 0 else 0


def format_report(
    summaries: list[SessionSummary],
    cfg: PreprocessConfig,
    targets: Targets,
) -> str:
    if not summaries:
        return "no readable sessions found"

    by_label: dict[str, float] = defaultdict(float)
    by_subject: dict[str, float] = defaultdict(float)
    subject_sessions: dict[str, set[str]] = defaultdict(set)
    subject_label: dict[tuple[str, str], float] = defaultdict(float)
    environments: set[str] = set()

    for s in summaries:
        environments.add(s.environment)
        subject_sessions[s.subject].add(s.date)
        for label, secs in s.seconds_by_label.items():
            by_label[label] += secs
            by_subject[s.subject] += secs
            subject_label[(s.subject, label)] += secs

    lines = [f"{len(summaries)} sessions, {len(subject_sessions)} subjects", ""]

    lines.append(f"{'session':<40}{'subject':<10}{'env':<10}{'labelled':>10}")
    lines.append("-" * 70)
    for s in summaries:
        lines.append(
            f"{s.name:<40}{s.subject:<10}{s.environment:<10}"
            f"{s.total_seconds:9.0f}s"
        )

    lines += ["", f"{'class':<14}{'seconds':>10}{'minutes':>10}{'~windows':>10}"]
    lines.append("-" * 44)
    total_windows = 0
    for label in LABEL_NAMES:
        secs = by_label.get(label, 0.0)
        windows = _estimated_windows(secs, cfg)
        total_windows += windows
        flag = "" if secs else "   <- MISSING"
        lines.append(
            f"{label:<14}{secs:10.0f}{secs / 60:10.1f}{windows:10d}{flag}"
        )
    lines.append(f"{'TOTAL':<14}{sum(by_label.values()):10.0f}"
                 f"{sum(by_label.values()) / 60:10.1f}{total_windows:10d}")

    lines += ["", "progress against M3 targets", "-" * 44]
    checks = [
        ("subjects", len(subject_sessions), targets.subjects),
        ("windows", total_windows, targets.total_windows),
        ("environments", len(environments), targets.environments),
    ]
    for name, have, want in checks:
        mark = "ok" if have >= want else "--"
        lines.append(f"  [{mark}] {name:<22}{have:>7} / {want}")

    for subject in sorted(subject_sessions):
        days = len(subject_sessions[subject])
        mark = "ok" if days >= targets.sessions_per_subject else "--"
        lines.append(
            f"  [{mark}] {subject + ' distinct days':<22}"
            f"{days:>7} / {targets.sessions_per_subject}"
        )

    short = [
        f"{subject}/{label}"
        for subject in sorted(subject_sessions)
        for label in LABEL_NAMES
        if subject_label[(subject, label)] < targets.seconds_per_activity_per_subject
    ]
    if short:
        lines += ["", f"under {targets.seconds_per_activity_per_subject:.0f}s "
                      f"per subject per class ({len(short)}):"]
        for i in range(0, len(short), 4):
            lines.append("  " + "  ".join(short[i:i + 4]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m csihar.datasheet",
        description="Report collected data against the M3 dataset targets.",
    )
    ap.add_argument("raw_dir", type=Path, nargs="?", default=Path("datasets/raw"))
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--hop-s", type=float, default=1.5)
    ap.add_argument("--out", type=Path, help="also write the report to a file")
    args = ap.parse_args(argv)

    if not args.raw_dir.is_dir():
        print(f"not a directory: {args.raw_dir}", file=sys.stderr)
        return 2

    cfg = PreprocessConfig(window_s=args.window_s, hop_s=args.hop_s)
    report = format_report(scan(args.raw_dir), cfg, Targets())
    print(report)
    if args.out:
        args.out.write_text(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
