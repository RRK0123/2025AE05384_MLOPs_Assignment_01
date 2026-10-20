"""Adversarial CV evidence changes must fail before a release can be approved."""

import json

import pandas as pd
import pytest
from cv_fixture import write_cv_fixture

from heart_disease.cv_validation import reconcile_cv


@pytest.fixture
def evidence(tmp_path):
    metadata, _, _, _ = write_cv_fixture(tmp_path)
    return tmp_path, metadata, tmp_path / metadata["report_directory"] / "tables"


def change_csv(path, mutate):
    frame = pd.read_csv(path)
    updated = mutate(frame)
    (frame if updated is None else updated).to_csv(path, index=False)


def test_valid_evidence_and_arbitrary_csv_row_order_are_accepted(evidence):
    root, metadata, tables = evidence
    result = reconcile_cv(root, metadata)
    assert len(result.folds) == 15 and len(result.summary) == 3
    assert result.summary.set_index("family").loc["logistic_regression", "accuracy_mean"] == 0.75
    for name in ["out_of_fold_predictions.csv", "outer_fold_metrics.csv", "model_comparison.csv"]:
        change_csv(tables / name, lambda frame: frame.sample(frac=1, random_state=7))
    assert reconcile_cv(root, metadata).summary.equals(result.summary)


@pytest.mark.parametrize(
    "metric",
    [
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ],
)
def test_changed_fold_metrics_are_rejected(evidence, metric):
    root, metadata, tables = evidence
    change_csv(
        tables / "outer_fold_metrics.csv",
        lambda frame: frame.__setitem__(metric, frame[metric] + 0.01),
    )
    with pytest.raises(ValueError, match="CV fold report differs"):
        reconcile_cv(root, metadata)


@pytest.mark.parametrize("metric", ["roc_auc_mean", "accuracy_mean", "recall_std", "folds"])
def test_changed_summary_statistics_are_rejected(evidence, metric):
    root, metadata, tables = evidence
    change_csv(
        tables / "model_comparison.csv",
        lambda frame: frame.__setitem__(metric, frame[metric] + 0.01),
    )
    with pytest.raises(ValueError, match="CV comparison report differs"):
        reconcile_cv(root, metadata)


@pytest.mark.parametrize(
    "problem",
    [
        "label",
        "probability",
        "target",
        "fold",
        "duplicate",
        "missing",
        "family",
        "invalid_label",
        "invalid_fold",
    ],
)
def test_changed_prediction_contract_is_rejected(evidence, problem):
    root, metadata, tables = evidence
    path = tables / "out_of_fold_predictions.csv"
    frame = pd.read_csv(path)
    index = frame.index[frame.family == "logistic_regression"][0]
    if problem == "label":
        frame.loc[index, "prediction"] = 1 - frame.loc[index, "prediction"]
    elif problem == "probability":
        frame.loc[index, "probability_disease"] += 0.01  # Same rank/label; Brier must still change.
    elif problem == "target":
        frame.loc[index, "target"] = 1 - frame.loc[index, "target"]
    elif problem in ["fold", "invalid_fold"]:
        frame.loc[index, "outer_fold"] = 2 if problem == "fold" else 99
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "missing":
        frame = frame.drop(index=index)
    elif problem == "family":
        frame.loc[index, "family"] = "unexpected"
    else:
        frame.loc[index, "prediction"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="CV"):
        reconcile_cv(root, metadata)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_invalid_probabilities_are_rejected(evidence, value):
    root, metadata, tables = evidence
    change_csv(
        tables / "out_of_fold_predictions.csv",
        lambda frame: frame.__setitem__("probability_disease", value),
    )
    with pytest.raises(ValueError, match="Invalid CV probabilities"):
        reconcile_cv(root, metadata)


@pytest.mark.parametrize(
    "table, key", [("outer_fold_metrics.csv", "outer_fold"), ("model_comparison.csv", "family")]
)
def test_duplicate_report_rows_are_rejected(evidence, table, key):
    root, metadata, tables = evidence
    change_csv(tables / table, lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="Duplicate CV"):
        reconcile_cv(root, metadata)


@pytest.mark.parametrize("boundary", ["outer_splits", "inner_splits", "final_tuning_splits"])
def test_changed_split_definitions_are_rejected(evidence, boundary):
    root, metadata, tables = evidence
    path = tables.parent / "cv_splits.json"
    definitions = json.loads(path.read_text())
    if boundary == "inner_splits":
        row = definitions["outer_splits"][0]["inner_splits"][0]
    else:
        row = definitions[boundary][0]
    row["validation_source_positions"][0] = 49  # A held-out source position.
    path.write_text(json.dumps(definitions))
    with pytest.raises(ValueError, match="CV .*fold definitions differ"):
        reconcile_cv(root, metadata)


def test_selected_family_is_derived_from_predictions(evidence):
    root, metadata, _ = evidence
    metadata["selected_family"] = "random_forest"
    with pytest.raises(ValueError, match="Selected family differs from prediction-derived"):
        reconcile_cv(root, metadata)


def test_changed_source_training_labels_fail_integrity_check(evidence):
    root, metadata, _ = evidence
    change_csv(
        root / "data/processed/train.csv",
        lambda frame: frame.__setitem__("target", 1 - frame.target),
    )
    with pytest.raises(ValueError, match="CV training evidence changed"):
        reconcile_cv(root, metadata)


def test_original_adversarial_combination_no_longer_passes(evidence):
    root, metadata, tables = evidence
    predictions = pd.read_csv(tables / "out_of_fold_predictions.csv")
    predictions["prediction"], predictions["probability_disease"] = 0, 0.5
    predictions.to_csv(tables / "out_of_fold_predictions.csv", index=False)
    comparison = pd.read_csv(tables / "model_comparison.csv")
    comparison.loc[comparison.family == "logistic_regression", "roc_auc_mean"] = 0.99
    comparison.to_csv(tables / "model_comparison.csv", index=False)
    with pytest.raises(ValueError, match="CV fold report differs from predictions"):
        reconcile_cv(root, metadata)
