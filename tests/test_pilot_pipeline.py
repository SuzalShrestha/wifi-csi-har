"""End-to-end contract test: a scripted session must assemble into a dataset.

session_script writes a session directory (parquets + metadata.json +
labels.json) and dataset.assemble_dataset reads one. Both sides of that
contract were unit-tested in isolation, but nothing pinned that they agree —
and the pilot dataset depends entirely on them agreeing. This walks the
whole Phase 2 path on synthetic frames: record -> QA -> assemble -> split.
"""

from pathlib import Path

import numpy as np
import pytest

from csihar import session_script
from csihar.dataset import LABEL_NAMES, assemble_dataset, split_random
from csihar.parser import parse_line
from csihar.preprocessing import PreprocessConfig
from csihar.preprocessing.subcarriers import N_USABLE
from csihar.qa import check_session
from csihar.session_script import parse_script, run_scripted_session
from csihar.simulate import generate_lines

T0 = 1_700_000_000.0  # a realistic host-clock epoch
SEGMENT_S = 8.0
STREAM_S = 30.0  # frames must span the whole scripted run


class _FakeClock:
    """Coarse steps so multi-second segments finish without a real wait."""

    def __init__(self, start: float = T0, step: float = 1.0):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _stub_read_receiver(state, stop) -> None:
    """Fill a receiver with ~100 Hz frames spanning T0..T0+STREAM_S.

    Seeded off the receiver id so the three streams are decorrelated, exactly
    as simulate.write_session does.
    """
    seed = 1000 * (int(state.receiver_id.removeprefix("rx")) - 1)
    lines = generate_lines("walking", duration_s=STREAM_S, seed=seed)
    for ts, line in lines:
        frame = parse_line(line, host_ts=T0 + ts)
        if frame is not None:
            state.frames.append(frame)


# Recording a session costs a few seconds of polled waiting and assembling it
# re-runs the whole preprocessing chain, so both are built once per module and
# only ever read by the tests below.
@pytest.fixture(scope="module")
def scripted_session(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("pilot")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session_script, "read_receiver", _stub_read_receiver)
        return run_scripted_session(
            ports={"/dev/f1": "rx1", "/dev/f2": "rx2", "/dev/f3": "rx3"},
            out_dir=out_dir,
            script=parse_script(f"background:{SEGMENT_S},walking:{SEGMENT_S}"),
            subject="pilot1",
            environment="room_a",
            prompt_fn=lambda _: "",
            clock=_FakeClock(),
        )


@pytest.fixture(scope="module")
def dataset(scripted_session):
    return assemble_dataset(scripted_session.parent, PreprocessConfig())


def test_scripted_session_has_the_files_assembly_expects(scripted_session):
    names = sorted(p.name for p in scripted_session.iterdir())
    assert names == [
        "labels.json", "metadata.json", "rx1.parquet", "rx2.parquet", "rx3.parquet",
    ]


def test_scripted_session_passes_qa(scripted_session):
    reports = check_session(scripted_session)
    assert len(reports) == 3
    for report in reports:
        assert report.mean_rate_hz > 90.0, report
        assert not report.radio_config_drift, report
        assert not report.null_subcarrier_mismatch, report


def test_scripted_session_assembles_into_a_trainable_dataset(
    dataset, scripted_session
):
    ds = dataset

    # (n, n_rx, T, S) with T = fs * window_s and the 52 usable subcarriers.
    assert ds.X.ndim == 4
    assert ds.X.shape[1] == 3
    assert ds.X.shape[2] == 300
    assert ds.X.shape[3] == N_USABLE
    assert ds.X.dtype == np.float32
    assert len(ds.y) == ds.X.shape[0] > 0
    assert np.isfinite(ds.X).all()

    assert set(ds.subjects) == {"pilot1"}
    assert set(ds.environments) == {"room_a"}
    assert set(ds.sessions) == {scripted_session.name}


def test_both_scripted_labels_survive_into_the_dataset(dataset):
    ds = dataset
    present = {LABEL_NAMES[i] for i in np.unique(ds.y)}
    assert present == {"background", "walking"}


def test_assembled_dataset_splits_without_losing_a_class(dataset):
    ds = dataset
    train_idx, test_idx = split_random(ds, test_fraction=0.25, seed=0)

    assert len(train_idx) + len(test_idx) == len(ds.y)
    assert not set(train_idx) & set(test_idx)
    assert set(np.unique(ds.y[train_idx])) == set(np.unique(ds.y))
