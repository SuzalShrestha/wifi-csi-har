"""Tests for the collection-progress datasheet."""

import json
from pathlib import Path

import pytest

from csihar.datasheet import Targets, format_report, scan, summarize_session
from csihar.preprocessing import PreprocessConfig
from csihar.simulate import write_session

CFG = PreprocessConfig(window_s=3.0, hop_s=1.5)


def _scripted(tmp_path: Path, name: str, subject: str, segments, env="room_a") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({
        "label": "scripted", "subject": subject, "environment": env,
        "receivers": {}, "notes": "",
    }))
    t = 1_700_000_000.0
    out = []
    for label, secs in segments:
        out.append({"label": label, "start_ts": t, "end_ts": t + secs})
        t += secs + 1
    (d / "labels.json").write_text(json.dumps({"segments": out}))
    return d


def test_scripted_session_durations_come_from_labels():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = _scripted(Path(tmp), "20260827_120000_a_scripted", "alice",
                      [("walking", 300.0), ("sitting", 120.0)])
        s = summarize_session(d)
    assert s.subject == "alice"
    assert s.date == "20260827"
    assert s.seconds_by_label == {"walking": 300.0, "sitting": 120.0}
    assert s.total_seconds == 420.0


def test_single_label_session_duration_comes_from_parquet(tmp_path):
    d = write_session(tmp_path, label="background", subject="alice",
                      environment="room_a", duration_s=4.0, seed=1)
    s = summarize_session(d)
    assert set(s.seconds_by_label) == {"background"}
    assert 3.0 < s.seconds_by_label["background"] < 5.0


def test_excluded_sessions_are_not_counted(tmp_path):
    d = write_session(tmp_path, label="background", subject="rig",
                      environment="bench", duration_s=4.0, seed=2)
    meta = json.loads((d / "metadata.json").read_text())
    meta["exclude_from_dataset"] = True
    (d / "metadata.json").write_text(json.dumps(meta))

    assert summarize_session(d) is None
    assert scan(tmp_path) == []


def test_scripted_session_without_labels_json_is_unusable(tmp_path):
    d = tmp_path / "20260827_120000_a_scripted"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({
        "label": "scripted", "subject": "a", "environment": "room_a",
        "receivers": {}, "notes": "",
    }))
    # label="scripted" is not a real class, so without labels.json there is
    # nothing to attribute the time to.
    assert summarize_session(d) is None


def test_report_flags_missing_class_and_unmet_targets(tmp_path):
    _scripted(tmp_path, "20260827_120000_a_scripted", "alice",
              [("walking", 300.0), ("sitting", 300.0)])
    report = format_report(scan(tmp_path), CFG, Targets())

    assert "falling" in report and "MISSING" in report
    assert "[--] subjects" in report
    assert "alice/falling" in report


def test_report_marks_a_met_target(tmp_path):
    for i in range(2):
        _scripted(tmp_path, f"2026082{i}_120000_a_scripted", "alice",
                  [("walking", 300.0)], env=f"room_{i}")
    report = format_report(scan(tmp_path), CFG, Targets(environments=2))
    assert "[ok] environments" in report


def test_empty_directory_reports_cleanly(tmp_path):
    assert format_report(scan(tmp_path), CFG, Targets()) == "no readable sessions found"
