"""Reject contradictory search records without fitting or altering real models."""

import json

import pandas as pd
import pytest
from cv_fixture import write_cv_fixture
from tuning_fixture import write_tuning_fixture

from heart_disease.modeling import build_model
from heart_disease.tuning_validation import reconcile_tuning, validate_fitted_parameters


@pytest.fixture
def tuning(tmp_path):
    metadata, *_ = write_cv_fixture(tmp_path)
    write_tuning_fixture(tmp_path, metadata)
    return tmp_path, metadata


def test_complete_tuning_tables_agree(tuning):
    root, metadata = tuning
    evidence = reconcile_tuning(root, metadata)
    assert len(evidence) == 12
    assert evidence["logistic_regression", None].best_parameters == metadata["selected_parameters"]
    assert evidence["logistic_regression", 1].folds == 3
    assert evidence["random_forest", None].folds == 5


@pytest.mark.parametrize(
    "change",
    [
        "selected_parameters",
        "final_parameters",
        "final_auc",
        "candidate_count",
        "grid",
        "outer_parameters",
        "outer_auc",
        "candidate_json",
        "parameter_column",
        "mean",
        "std",
        "rank",
        "split_score",
        "fold_count",
        "train_mean",
        "missing_candidate",
        "duplicate_candidate",
        "nonfinite",
        "outside_range",
    ],
)
def test_inconsistent_tuning_evidence_is_rejected(tuning, change):
    root, metadata = tuning
    tables = root / metadata["report_directory"] / "tables"
    path = tables / "logistic_regression_final_search.csv"
    rows = pd.read_csv(path)
    if change == "selected_parameters":
        metadata["selected_parameters"] = {"classifier__C": 999.0}
    elif change == "final_parameters":
        metadata["final_searches"]["logistic_regression"]["parameters"] = {"classifier__C": 999.0}
    elif change == "final_auc":
        metadata["final_searches"]["logistic_regression"]["tuning_roc_auc"] = 0.99999
    elif change == "candidate_count":
        metadata["final_searches"]["logistic_regression"]["candidate_count"] = 1
    elif change == "grid":
        metadata["parameter_grids"] = {}
    elif change.startswith("outer_"):
        path = tables / "outer_fold_metrics.csv"
        rows = pd.read_csv(path)
        mask = (rows.family == "logistic_regression") & (rows.outer_fold == 1)
        if change == "outer_parameters":
            rows.loc[mask, "best_parameters"] = json.dumps({"classifier__C": 999})
        else:
            rows.loc[mask, "inner_best_roc_auc"] = 0.99999
    elif change == "candidate_json":
        rows.loc[0, "params"] = json.dumps({"classifier__C": 999})
    elif change == "parameter_column":
        rows.loc[0, "param_classifier__C"] = 999
    elif change in ["mean", "std", "rank"]:
        column = {
            "mean": "mean_test_roc_auc",
            "std": "std_test_roc_auc",
            "rank": "rank_test_roc_auc",
        }[change]
        rows.loc[0, column] = 999 if change == "rank" else 0.99999
    elif change == "split_score":
        rows.loc[0, "split0_test_roc_auc"] = 0.99999
    elif change == "fold_count":
        rows["split5_test_roc_auc"] = 0.75
    elif change == "train_mean":
        rows.loc[0, "mean_train_roc_auc"] = 0.99999
    elif change == "missing_candidate":
        rows = rows.iloc[1:]
    elif change == "duplicate_candidate":
        rows.iloc[1] = rows.iloc[0]
    elif change == "nonfinite":
        rows.loc[0, "split0_test_roc_auc"] = float("nan")
    else:
        rows.loc[0, "split0_test_roc_auc"] = 1.1
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError):
        reconcile_tuning(root, metadata)


@pytest.mark.parametrize("family", ["logistic_regression", "random_forest"])
def test_fitted_classifier_parameters_must_match_declared_search(tuning, family):
    _, metadata = tuning
    parameters = metadata["final_searches"][family]["parameters"]
    model = build_model(family).set_params(**parameters)
    validate_fitted_parameters(model, family, parameters)
    model.set_params(classifier__random_state=999)
    with pytest.raises(ValueError, match="Fitted classifier parameters"):
        validate_fitted_parameters(model, family, parameters)
