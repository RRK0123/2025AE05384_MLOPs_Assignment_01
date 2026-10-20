"""Verify metrics, selection rules, and serialization of the full pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest

from heart_disease.modeling import build_model, evaluate_predictions, select_family
from heart_disease.predict import feature_frame
from heart_disease.schema import CATEGORIES, FEATURES, NUMERIC


def training_example():
    frame = pd.DataFrame({c: [float(next(iter(levels)))] * 10 for c, levels in CATEGORIES.items()})
    for column in NUMERIC:
        frame[column] = np.arange(10, 20, dtype=float)
    frame.loc[0, "ca"] = np.nan
    return frame[FEATURES], pd.Series([0] * 5 + [1] * 5)


def test_metrics_use_disease_as_positive_class():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.8, 0.7, 0.4])
    assert result["accuracy"] == result["precision"] == result["recall"] == result["roc_auc"] == 0.5
    assert [
        result[key]
        for key in ["true_negative", "false_positive", "false_negative", "true_positive"]
    ] == [1, 1, 1, 1]


def test_family_selection_uses_only_cv_auc_and_ignores_baseline():
    summary = pd.DataFrame(
        {
            "family": ["dummy_baseline", "logistic_regression", "random_forest"],
            "roc_auc_mean": [1.0, 0.8, 0.9],
        }
    )
    assert select_family(summary) == "random_forest"
    summary.loc[1, "roc_auc_mean"] = 0.9
    assert select_family(summary) == "logistic_regression"


def test_pipeline_serialization_preserves_missing_input_predictions(tmp_path):
    features, target = training_example()
    model = build_model("logistic_regression").fit(features, target)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
    np.testing.assert_allclose(
        restored.predict_proba(features), model.predict_proba(features), rtol=0, atol=0
    )
    assert list(model.feature_names_in_) == FEATURES


def test_forest_does_not_scale_numeric_measurements():
    features, _ = training_example()
    preprocessor = build_model("random_forest").named_steps["preprocessor"]
    transformed = preprocessor.fit_transform(features)
    np.testing.assert_allclose(transformed[:, :5], features[NUMERIC].to_numpy())


@pytest.mark.parametrize("bad_value", ["54", True, float("inf"), -1])
def test_prediction_input_rejects_invalid_measurements(bad_value):
    features, _ = training_example()
    record = features.iloc[1].to_dict()
    record["age"] = bad_value
    with pytest.raises(ValueError):
        feature_frame({"instances": [record]})


def test_prediction_input_rejects_target_leakage():
    features, _ = training_example()
    record = features.iloc[1].to_dict()
    record["target"] = 1
    with pytest.raises(ValueError, match="13 documented"):
        feature_frame({"instances": [record]})


def test_prediction_input_accepts_null_for_fitted_imputation():
    features, _ = training_example()
    record = features.iloc[1].to_dict()
    record["ca"] = None
    assert feature_frame({"instances": [record]})["ca"].isna().all()
