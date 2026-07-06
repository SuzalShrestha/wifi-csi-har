import pandas as pd
import pytest

from csihar.baseline import run_baseline
from csihar.dataset import assemble_dataset
from csihar.preprocessing import PreprocessConfig
from csihar.simulate import write_session

CFG = PreprocessConfig()


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    """2 subjects x 2 labels (background, walking) x 1 session each."""
    root = tmp_path_factory.mktemp("raw")
    seed = 0
    for subject in ("s1", "s2"):
        for label in ("background", "walking"):
            write_session(
                root, label=label, subject=subject, duration_s=8.0, seed=seed
            )
            seed += 1
    return root


@pytest.fixture(scope="module")
def dataset(raw_dir):
    return assemble_dataset(raw_dir, CFG)


def test_dataset_fixture_yields_expected_window_count(dataset):
    assert 10 <= dataset.X.shape[0] <= 20


def test_run_baseline_random_split_svm_accurate(dataset, tmp_path):
    results_csv = tmp_path / "results.csv"
    figures_dir = tmp_path / "figures"

    rows = run_baseline(
        dataset, "random", seed=0,
        results_csv=results_csv, figures_dir=figures_dir,
    )

    model_names = {name for name, _, _ in rows}
    assert model_names == {"svm_rbf", "random_forest"}

    svm_report = next(r for name, _, r in rows if name == "svm_rbf")
    assert svm_report.accuracy > 0.8

    assert results_csv.exists()
    df = pd.read_csv(results_csv)
    assert len(df) == 2
    assert set(df["model"]) == {"svm_rbf", "random_forest"}
    assert (df["split"] == "random").all()

    assert (figures_dir / "cm_svm_rbf_random.png").exists()
    assert (figures_dir / "cm_random_forest_random.png").exists()


def test_run_baseline_cross_subject_loso(dataset, tmp_path):
    results_csv = tmp_path / "results.csv"
    figures_dir = tmp_path / "figures"

    rows = run_baseline(
        dataset, "cross-subject", seed=0,
        results_csv=results_csv, figures_dir=figures_dir,
    )

    # 2 models x (2 subject folds + 1 pooled) = 6 rows
    assert len(rows) == 6

    df = pd.read_csv(results_csv)
    assert len(df) == 6
    assert (df["split"] == "cross-subject").all()

    pooled_notes = df[df["notes"] == "LOSO pooled"]
    assert len(pooled_notes) == 2  # one per model

    fold_notes = df[df["notes"].str.startswith("fold=")]
    assert len(fold_notes) == 4  # 2 models x 2 subjects

    assert (figures_dir / "cm_svm_rbf_cross-subject.png").exists()
    assert (figures_dir / "cm_random_forest_cross-subject.png").exists()
