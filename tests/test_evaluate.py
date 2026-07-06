import csv

import numpy as np
import pytest

from csihar.evaluate import (
    MetricsReport,
    append_result,
    compute_metrics,
    save_confusion_matrix,
)

LABELS = ("background", "walking", "falling")


def test_compute_metrics_hand_computable():
    # 3 classes, 6 samples. Class ids: 0=background, 1=walking, 2=falling.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    report = compute_metrics(y_true, y_pred, LABELS)

    # accuracy: correct = indices 0,2,3,4 -> 4/6
    assert report.accuracy == pytest.approx(4 / 6)

    # per-class precision/recall/f1/support hand-computed:
    # background: TP=1 (idx0), FN=1 (idx1 predicted walking),
    #   FP=1 (idx5 true falling predicted background)
    #   precision = 1/2, recall = 1/2, f1 = 1/2, support=2
    bg = report.per_class["background"]
    assert bg.support == 2
    assert bg.precision == pytest.approx(0.5)
    assert bg.recall == pytest.approx(0.5)
    assert bg.f1 == pytest.approx(0.5)

    # walking: true idx2,3 both predicted 1 -> TP=2, FN=0
    #   FP: idx1 (true background predicted walking) -> 1
    #   precision = 2/3, recall = 1.0, support=2
    walk = report.per_class["walking"]
    assert walk.support == 2
    assert walk.precision == pytest.approx(2 / 3)
    assert walk.recall == pytest.approx(1.0)

    # falling: true idx4,5 -> pred 2,0 -> TP=1, FN=1, FP=0
    #   precision = 1.0, recall = 0.5, support=2
    fall = report.per_class["falling"]
    assert fall.support == 2
    assert fall.precision == pytest.approx(1.0)
    assert fall.recall == pytest.approx(0.5)

    expected_macro_f1 = (bg.f1 + walk.f1 + fall.f1) / 3
    assert report.macro_f1 == pytest.approx(expected_macro_f1)

    # falling present in y_true -> falling_recall == falling's recall
    assert report.falling_recall == pytest.approx(0.5)


def test_falling_recall_none_when_absent():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    report = compute_metrics(y_true, y_pred, LABELS)
    assert report.falling_recall is None


def test_falling_recall_none_when_falling_not_in_label_names():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    report = compute_metrics(y_true, y_pred, ("background", "walking"))
    assert report.falling_recall is None


def _dummy_report() -> MetricsReport:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    return compute_metrics(y_true, y_pred, LABELS)


def test_append_result_creates_header_once_and_appends(tmp_path):
    csv_path = tmp_path / "results.csv"
    report = _dummy_report()

    append_result(
        csv_path, model="svm_rbf", split="random", seed=0, config="cfg-a",
        report=report, n_train=10, n_test=6, notes="first",
    )
    append_result(
        csv_path, model="random_forest", split="random", seed=0, config="cfg-a",
        report=report, n_train=10, n_test=6, notes="second",
    )

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert rows[0]["model"] == "svm_rbf"
    assert rows[0]["notes"] == "first"
    assert rows[1]["model"] == "random_forest"
    assert rows[1]["notes"] == "second"
    assert rows[0]["split"] == "random"
    assert rows[0]["seed"] == "0"
    assert rows[0]["config"] == "cfg-a"
    assert rows[0]["n_train"] == "10"
    assert rows[0]["n_test"] == "6"
    assert float(rows[0]["accuracy"]) == pytest.approx(report.accuracy, abs=1e-4)
    assert float(rows[0]["macro_f1"]) == pytest.approx(report.macro_f1, abs=1e-4)

    # header appears exactly once
    text = csv_path.read_text()
    assert text.count("accuracy") == 1


def test_save_confusion_matrix_writes_png(tmp_path):
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    out_path = tmp_path / "figs" / "cm.png"
    result = save_confusion_matrix(y_true, y_pred, LABELS, out_path, title="test")
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
