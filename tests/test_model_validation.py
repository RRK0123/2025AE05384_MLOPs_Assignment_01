"""Regressions for reconciling the selected artifact with held-out evidence."""

import json
import shutil

import joblib
import pytest

from heart_disease.config import ROOT
from heart_disease.releases import resolve_model_directory
from heart_disease.validate_model import checksum, validate_model

pytestmark = pytest.mark.integration


@pytest.fixture
def model_workspace(tmp_path):
    selected = resolve_model_directory()
    metadata = json.loads((selected / "metadata.json").read_text())
    for name in [
        "data",
        metadata["model_directory"],
        metadata["report_directory"],
    ]:
        shutil.copytree(ROOT / name, tmp_path / name)
    shutil.copytree(selected, tmp_path / "models/selected")
    (tmp_path / "docs").mkdir()
    shutil.copyfile(ROOT / "docs/sample_request.json", tmp_path / "docs/sample_request.json")
    return tmp_path, metadata


def test_current_saved_model_reproduces_test_evidence(model_workspace):
    root, _ = model_workspace
    result = validate_model(root)
    assert result["model_test_predictions_verified"]
    assert result["reported_test_metrics_reproduced_from_saved_model"]
    assert result["outer_cv_predictions_labels_and_folds_verified"]
    assert result["outer_cv_fold_metrics_recomputed"]
    assert result["outer_cv_summary_and_selection_recomputed"]
    assert result["tuning_tables_and_selected_parameters_reconciled"]
    assert result["fitted_classifier_parameters_match_tuning"]


def test_resealed_incorrect_classifier_is_rejected(model_workspace):
    root, metadata = model_workspace
    artifact = root / "models/selected/model.joblib"
    model = joblib.load(artifact)
    model.named_steps["classifier"].coef_[:] = 0
    model.named_steps["classifier"].intercept_[:] = 0
    joblib.dump(model, artifact, compress=3)
    shutil.copyfile(artifact, root / metadata["model_directory"] / "logistic_regression.joblib")
    metadata["model_sha256"] = checksum(artifact)
    metadata["final_searches"]["logistic_regression"]["artifact_sha256"] = checksum(artifact)
    (root / "models/selected/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match="Saved model class predictions differ"):
        validate_model(root)


def test_changed_prediction_csv_is_rejected(model_workspace):
    import pandas as pd

    root, metadata = model_workspace
    path = root / metadata["report_directory"] / "tables/selected_test_predictions.csv"
    rows = pd.read_csv(path)
    rows.loc[0, "probability_disease"] += 0.01
    rows.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="Saved model probabilities differ"):
        validate_model(root)


@pytest.mark.parametrize("change", ["predictions", "summary"])
def test_saved_model_validation_rejects_changed_cv_evidence(model_workspace, change):
    import pandas as pd

    root, metadata = model_workspace
    tables = root / metadata["report_directory"] / "tables"
    if change == "predictions":
        path = tables / "out_of_fold_predictions.csv"
        rows = pd.read_csv(path)
        rows["prediction"], rows["probability_disease"] = 0, 0.5
    else:
        path = tables / "model_comparison.csv"
        rows = pd.read_csv(path)
        rows.loc[rows.family == "logistic_regression", "roc_auc_mean"] = 0.99
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="CV .*report differs from predictions"):
        validate_model(root)


@pytest.mark.parametrize("change", ["selected_parameters", "tuning_score", "fitted_parameters"])
def test_saved_model_validation_rejects_changed_tuning_evidence(model_workspace, change):
    root, metadata = model_workspace
    if change == "selected_parameters":
        metadata["selected_parameters"] = {"classifier__C": 999.0}
    elif change == "tuning_score":
        metadata["final_searches"]["logistic_regression"]["tuning_roc_auc"] = 0.99999
    else:
        artifact = root / "models/selected/model.joblib"
        model = joblib.load(artifact)
        model.set_params(classifier__random_state=999)
        joblib.dump(model, artifact, compress=3)
        shutil.copyfile(artifact, root / metadata["model_directory"] / "logistic_regression.joblib")
        digest = checksum(artifact)
        metadata["model_sha256"] = digest
        metadata["final_searches"]["logistic_regression"]["artifact_sha256"] = digest
    (root / "models/selected/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="[Tt]uning|Fitted classifier parameters"):
        validate_model(root)
