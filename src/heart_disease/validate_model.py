"""Check saved models, fold boundaries, and reported evaluation evidence."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .cv_validation import reconcile_cv
from .modeling import (
    FAMILIES,
    disease_probabilities,
    evaluate_predictions,
)
from .predict import predict_payload
from .releases import MODEL_FILES, resolve_model_directory
from .schema import CATEGORIES, FEATURES, NUMERIC, TARGET
from .tuning_validation import reconcile_tuning, validate_fitted_parameters
from .validate import require


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model(root: Path = ROOT, model_directory: Path | None = None):
    root = Path(root).resolve()
    selected = resolve_model_directory(model_directory, root)
    metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
    reports, models = root / metadata["report_directory"], root / metadata["model_directory"]
    processed = root / "data/processed"
    split = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    require(
        checksum(selected / "model.joblib") == metadata["model_sha256"],
        "Selected artifact checksum mismatch.",
    )
    for path, key in [
        (processed / "split_manifest.json", "split_manifest_sha256"),
        (processed / "train.csv", "training_csv_sha256"),
        (processed / "test.csv", "test_csv_sha256"),
    ]:
        require(checksum(path) == metadata[key], f"Training evidence changed: {path.name}.")
    require(
        checksum(root / "data/raw/processed.cleveland.data") == metadata["source_sha256"],
        "Raw dataset changed since training.",
    )
    reconcile_cv(root, metadata)
    searches = reconcile_tuning(root, metadata)
    for family in FAMILIES:
        artifact = models / f"{family}.joblib"
        require(
            checksum(artifact) == metadata["final_searches"][family]["artifact_sha256"],
            "Candidate artifact checksum mismatch.",
        )
        validate_fitted_parameters(
            joblib.load(artifact), family, searches[family, None].best_parameters
        )
    train = pd.read_csv(processed / "train.csv")
    model = joblib.load(selected / "model.joblib")
    require(
        metadata["model_sha256"]
        == metadata["final_searches"][metadata["selected_family"]]["artifact_sha256"],
        "Selected model differs from the final tuning artifact",
    )
    validate_fitted_parameters(model, metadata["selected_family"], metadata["selected_parameters"])
    require(list(model.feature_names_in_) == FEATURES, "Unexpected model input features.")
    require(model.classes_.tolist() == [0, 1], "Unexpected output classes.")
    transformed = model.named_steps["preprocessor"].transform(train[FEATURES].astype(float))
    require(
        transformed.shape == (len(train), 28) and np.isfinite(transformed).all(),
        "Invalid transformed training data.",
    )
    preprocessor = model.named_steps["preprocessor"]
    np.testing.assert_allclose(
        preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_,
        train[NUMERIC].median(),
    )
    expected_modes = [float(train[c].mode().iloc[0]) for c in CATEGORIES]
    np.testing.assert_allclose(
        preprocessor.named_transformers_["categorical"].named_steps["imputer"].statistics_,
        expected_modes,
    )
    if metadata["numeric_scaling"]:
        np.testing.assert_allclose(
            preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_,
            train[NUMERIC].fillna(train[NUMERIC].median()).mean(),
        )
    stored_test = pd.read_csv(reports / "tables/selected_test_predictions.csv")
    test = pd.read_csv(processed / "test.csv")
    require(
        stored_test["source_position"].tolist() == split["test_positions"],
        "Test prediction alignment mismatch.",
    )
    np.testing.assert_array_equal(stored_test[TARGET], test[TARGET])
    test_features = test[FEATURES].astype(float)
    actual_prediction = model.predict(test_features)
    actual_probability = disease_probabilities(model, test_features)
    require(
        np.isfinite(actual_probability).all()
        and ((actual_probability >= 0) & (actual_probability <= 1)).all(),
        "Saved model produced invalid test probabilities.",
    )
    np.testing.assert_array_equal(
        actual_prediction,
        stored_test["prediction"],
        err_msg="Saved model class predictions differ from recorded test predictions.",
    )
    np.testing.assert_allclose(
        actual_probability,
        stored_test["probability_disease"],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Saved model probabilities differ from recorded test predictions.",
    )
    metrics = evaluate_predictions(test[TARGET], actual_prediction, actual_probability)
    for key, value in metrics.items():
        np.testing.assert_allclose(value, metadata["test_metrics"][key], rtol=1e-12, atol=1e-12)
    sample_path = selected / "sample_request.json"
    if not sample_path.exists():
        sample_path = root / "docs/sample_request.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    prediction = predict_payload(sample, selected)
    require(0 <= prediction["predictions"][0]["confidence"] <= 1, "Invalid inference confidence.")
    result = {
        "run_id": metadata["run_id"],
        "model_files_sha256": {name: checksum(selected / name) for name in MODEL_FILES},
        "model_checksums_verified": True,
        "no_test_rows_in_nested_or_final_cv": True,
        "fold_coverage_verified": True,
        "all_candidate_search_metrics_finite": True,
        "tuning_tables_and_selected_parameters_reconciled": True,
        "fitted_classifier_parameters_match_tuning": True,
        "selected_by_outer_cv_only": True,
        "outer_cv_predictions_labels_and_folds_verified": True,
        "outer_cv_fold_metrics_recomputed": True,
        "outer_cv_summary_and_selection_recomputed": True,
        "training_fitted_preprocessing_verified": True,
        "reported_test_metrics_reproduced_from_saved_predictions": True,
        "model_test_predictions_verified": True,
        "reported_test_metrics_reproduced_from_saved_model": True,
        "inference_smoke_check_passed": True,
    }
    (reports / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    print(json.dumps(validate_model(), indent=2))


if __name__ == "__main__":
    main()
