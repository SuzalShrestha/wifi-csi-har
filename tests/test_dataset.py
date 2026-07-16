import json

import numpy as np
import pandas as pd
import pytest

from csihar.dataset import (
    LABEL_NAMES,
    HarDataset,
    assemble_dataset,
    assemble_session,
    iter_cross_subject,
    load_dataset,
    save_dataset,
    split_cross_environment,
    split_cross_session,
    split_cross_subject,
    split_random,
)
from csihar.preprocessing import PreprocessConfig
from csihar.simulate import write_session
from csihar.storage import load_amplitudes

CFG = PreprocessConfig()
WALK = LABEL_NAMES.index("walking")
SIT = LABEL_NAMES.index("sitting")


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    """2 subjects x 2 labels = 4 short sessions, the shared fixture set."""
    root = tmp_path_factory.mktemp("raw")
    seed = 0
    for subject in ("s1", "s2"):
        for label in ("walking", "sitting"):
            write_session(
                root, label=label, subject=subject, duration_s=8.0, seed=seed
            )
            seed += 1
    return root


@pytest.fixture(scope="module")
def dataset(raw_dir):
    return assemble_dataset(raw_dir, CFG)


def first_window_start(session_dir) -> float:
    """Reference (first sorted receiver) stream start = first window start."""
    host_ts, _ = load_amplitudes(session_dir / "rx1.parquet")
    return float(host_ts.min())


def write_labels(session_dir, segments) -> None:
    (session_dir / "labels.json").write_text(json.dumps({"segments": segments}))


# ------------------------------------------------------------ write_session


def test_write_session_creates_parseable_session(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=6.0, seed=7)
    assert session.name == "sim_s1_walking_7"
    assert (session / "metadata.json").exists()
    parquets = sorted(session.glob("*.parquet"))
    assert [p.stem for p in parquets] == ["rx1", "rx2", "rx3"]
    for pq in parquets:
        host_ts, amps = load_amplitudes(pq)
        assert amps.shape[1] == 64
        assert len(host_ts) > 500  # ~600 packets minus drops
        assert host_ts.min() >= 999.9  # t0 default 1000 plus jitter
    meta = json.loads((session / "metadata.json").read_text())
    assert meta["label"] == "walking"
    assert meta["subject"] == "s1"
    # distinct per-receiver seeds -> streams are not identical
    df1 = pd.read_parquet(session / "rx1.parquet")
    df2 = pd.read_parquet(session / "rx2.parquet")
    assert not np.array_equal(
        np.stack(df1["csi_real"].head(50)), np.stack(df2["csi_real"].head(50))
    )


# --------------------------------------------------------- assemble_session


def test_assemble_single_label_session(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=8.0, seed=1)
    part = assemble_session(session, CFG)
    n = part.X.shape[0]
    assert n >= 3
    assert part.X.shape == (n, 3, 300, 52)
    assert part.X.dtype == np.float32
    assert part.y.dtype == np.int64
    assert (part.y == WALK).all()
    assert part.subject == "s1"
    assert part.environment == "sim_room"
    assert part.session == session.name


def test_assemble_unknown_label_raises(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=6.0, seed=2)
    meta_path = session / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "label": "juggling"}))
    with pytest.raises(ValueError, match=session.name):
        assemble_session(session, CFG)


def test_labels_json_segments_gap_and_trimming(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=12.0, seed=3)
    n_all = assemble_session(session, CFG).X.shape[0]
    t0 = first_window_start(session)
    # walking [t0-1, t0+5.3], gap, sitting [t0+6.6, t0+20]; margin 0.5,
    # window 3.0, hop 1.5 -> window starts t0 + 1.5k:
    #   k=0,1 walking; k=2,3,4 straddle gap/boundaries -> dropped; k=5 sitting
    write_labels(session, [
        {"label": "walking", "start_ts": t0 - 1.0, "end_ts": t0 + 5.3},
        {"label": "sitting", "start_ts": t0 + 6.6, "end_ts": t0 + 20.0},
    ])
    part = assemble_session(session, CFG)
    assert part.X.shape[0] < n_all  # gap + boundary windows dropped
    assert set(part.y.tolist()) == {WALK, SIT}
    assert (part.y == WALK).sum() == 2
    assert 1 <= (part.y == SIT).sum() <= 2


def test_boundary_margin_drops_window_at_exact_segment_start(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=8.0, seed=4)
    t0 = first_window_start(session)
    write_labels(session, [
        {"label": "walking", "start_ts": t0, "end_ts": t0 + 4.0},
    ])
    # margin 0: window starting exactly at segment start fits [t0, t0+3]
    from dataclasses import replace
    part = assemble_session(session, replace(CFG, boundary_margin_s=0.0))
    assert part.X.shape[0] == 1
    assert (part.y == WALK).all()
    # margin 0.5: trimmed segment is 3.0s -> no 3s window fits -> all dropped
    part = assemble_session(session, CFG)
    assert part.X.shape == (0, 3, 300, 52)
    assert part.y.shape == (0,)


def test_labels_json_unknown_label_raises(tmp_path):
    session = write_session(tmp_path, label="walking", subject="s1",
                            duration_s=6.0, seed=5)
    write_labels(session, [
        {"label": "moonwalk", "start_ts": 1000.0, "end_ts": 1006.0},
    ])
    with pytest.raises(ValueError, match=session.name):
        assemble_session(session, CFG)


# --------------------------------------------------------- assemble_dataset


def test_assemble_dataset_shapes_and_tags(dataset):
    n = dataset.X.shape[0]
    assert n >= 12  # 4 sessions x ~4 windows
    assert dataset.X.shape[1:] == (3, 300, 52)
    assert dataset.X.dtype == np.float32
    assert dataset.y.shape == (n,)
    assert set(dataset.y.tolist()) == {WALK, SIT}
    assert set(dataset.subjects.tolist()) == {"s1", "s2"}
    assert len(set(dataset.sessions.tolist())) == 4
    assert set(dataset.environments.tolist()) == {"sim_room"}
    assert dataset.label_names == LABEL_NAMES
    # every window's session name encodes its subject and label
    for i in range(n):
        assert dataset.subjects[i] in dataset.sessions[i]
        assert LABEL_NAMES[dataset.y[i]] in dataset.sessions[i]


def test_assemble_dataset_warns_and_skips_empty_sessions(tmp_path):
    write_session(tmp_path, label="walking", subject="s1", duration_s=8.0, seed=8)
    empty = write_session(tmp_path, label="sitting", subject="s2",
                          duration_s=8.0, seed=9)
    # labels.json with an out-of-range segment -> zero windows survive
    write_labels(empty, [
        {"label": "sitting", "start_ts": 1.0, "end_ts": 2.0},
    ])
    with pytest.warns(UserWarning, match=empty.name):
        ds = assemble_dataset(tmp_path, CFG)
    assert set(ds.sessions.tolist()) == {"sim_s1_walking_8"}


def test_hardataset_validates_shapes():
    with pytest.raises(ValueError):
        HarDataset(
            X=np.zeros((3, 2, 10, 5), np.float32),
            y=np.zeros(2, np.int64),  # length mismatch
            subjects=np.array(["a", "a", "a"]),
            sessions=np.array(["s", "s", "s"]),
            environments=np.array(["e", "e", "e"]),
        )


# -------------------------------------------------------------- persistence


def test_save_load_roundtrip(dataset, tmp_path):
    path = save_dataset(dataset, tmp_path / "ds.npz")
    loaded = load_dataset(path)
    assert np.array_equal(loaded.X, dataset.X)
    assert np.array_equal(loaded.y, dataset.y)
    assert loaded.y.dtype == np.int64
    assert loaded.subjects.tolist() == dataset.subjects.tolist()
    assert loaded.sessions.tolist() == dataset.sessions.tolist()
    assert loaded.environments.tolist() == dataset.environments.tolist()
    assert loaded.label_names == dataset.label_names


# ------------------------------------------------------------------- splits


def assert_partition(ds, train_idx, test_idx):
    assert train_idx.dtype == np.int64 and test_idx.dtype == np.int64
    assert len(set(train_idx) & set(test_idx)) == 0
    assert sorted(set(train_idx) | set(test_idx)) == list(range(len(ds.y)))
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_split_random_stratified_and_deterministic(dataset):
    train_idx, test_idx = split_random(dataset, test_fraction=0.2, seed=0)
    assert_partition(dataset, train_idx, test_idx)
    for cls in np.unique(dataset.y):
        n_cls = (dataset.y == cls).sum()
        n_test = (dataset.y[test_idx] == cls).sum()
        assert 1 <= n_test < n_cls  # approximate stratification
        assert abs(n_test / n_cls - 0.2) < 0.15
    again = split_random(dataset, test_fraction=0.2, seed=0)
    assert np.array_equal(again[0], train_idx)
    assert np.array_equal(again[1], test_idx)
    other = split_random(dataset, test_fraction=0.2, seed=1)
    assert not np.array_equal(other[1], test_idx)


def test_split_cross_session_invariants(dataset):
    train_idx, test_idx = split_cross_session(dataset, seed=0)
    assert_partition(dataset, train_idx, test_idx)
    train_sessions = set(dataset.sessions[train_idx])
    test_sessions = set(dataset.sessions[test_idx])
    assert train_sessions & test_sessions == set()  # no session straddles
    assert set(dataset.y[test_idx]) <= set(dataset.y[train_idx])
    again = split_cross_session(dataset, seed=0)
    assert np.array_equal(again[0], train_idx)
    assert np.array_equal(again[1], test_idx)


def test_split_cross_session_needs_two_sessions(tmp_path):
    write_session(tmp_path, label="walking", subject="s1", duration_s=8.0, seed=6)
    ds = assemble_dataset(tmp_path, CFG)
    with pytest.raises(ValueError):
        split_cross_session(ds)


def test_split_cross_subject_invariants(dataset):
    train_idx, test_idx = split_cross_subject(dataset, "s1")
    assert_partition(dataset, train_idx, test_idx)
    assert set(dataset.subjects[test_idx]) == {"s1"}
    assert set(dataset.subjects[train_idx]) == {"s2"}
    with pytest.raises(ValueError, match="unknown subject"):
        split_cross_subject(dataset, "nobody")


def test_split_cross_environment_invariants():
    n = 10
    ds = HarDataset(
        X=np.zeros((n, 1, 8, 4), np.float32),
        y=(np.arange(n) % 2).astype(np.int64),
        subjects=np.array(["s1"] * n),
        sessions=np.array(["sess1"] * 6 + ["sess2"] * 4),
        environments=np.array(["room_a"] * 6 + ["room_b"] * 4),
    )
    train_idx, test_idx = split_cross_environment(ds, "room_b")
    assert not set(ds.environments[train_idx]) & set(ds.environments[test_idx])
    assert len(train_idx) == 6 and len(test_idx) == 4
    with pytest.raises(ValueError, match="unknown environment"):
        split_cross_environment(ds, "nope")
    only = HarDataset(
        X=ds.X, y=ds.y,
        subjects=ds.subjects, sessions=ds.sessions,
        environments=np.array(["room_a"] * n),
    )
    with pytest.raises(ValueError, match="train empty"):
        split_cross_environment(only, "room_a")


def test_iter_cross_subject_covers_all_subjects(dataset):
    folds = list(iter_cross_subject(dataset))
    assert [subject for subject, _, _ in folds] == ["s1", "s2"]
    for subject, train_idx, test_idx in folds:
        assert_partition(dataset, train_idx, test_idx)
        assert set(dataset.subjects[test_idx]) == {subject}
        assert subject not in set(dataset.subjects[train_idx])


def test_splits_do_not_mutate_dataset(dataset):
    before = {
        "X": dataset.X.copy(), "y": dataset.y.copy(),
        "subjects": dataset.subjects.copy(),
        "sessions": dataset.sessions.copy(),
        "environments": dataset.environments.copy(),
    }
    split_random(dataset, seed=0)
    split_cross_session(dataset, seed=0)
    split_cross_subject(dataset, "s1")
    list(iter_cross_subject(dataset))
    assert np.array_equal(dataset.X, before["X"])
    assert np.array_equal(dataset.y, before["y"])
    assert dataset.subjects.tolist() == before["subjects"].tolist()
    assert dataset.sessions.tolist() == before["sessions"].tolist()
    assert dataset.environments.tolist() == before["environments"].tolist()
