"""Check bundle tampering and reusable inference contracts."""

import importlib.metadata
import json

import joblib
import pytest
from test_modeling import training_example

from heart_disease.modeling import build_model
from heart_disease.predict import load_predictor
from heart_disease.schema import CATEGORIES, FEATURES, NUMERIC
from heart_disease.verify_bundle import checksum, verify_files


def saved_model(directory):
    features, target = training_example()
    model = build_model("logistic_regression").fit(features, target)
    artifact = directory / "model.joblib"
    joblib.dump(model, artifact)
    metadata = {
        "run_id": "test-model",
        "model_sha256": checksum(artifact),
        "input_features": FEATURES,
        "encoded_feature_count": 28,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    schema = {
        "fields": FEATURES,
        "numeric_fields": NUMERIC,
        "categorical_codes": {name: list(levels) for name, levels in CATEGORIES.items()},
    }
    (directory / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return features, metadata


def test_predictor_reuses_model_and_fitted_statistics(tmp_path, monkeypatch):
    features, _ = saved_model(tmp_path)
    predictor = load_predictor(tmp_path)
    medians = (
        predictor.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_.copy()
    )

    def forbid_loading(*args, **kwargs):
        raise AssertionError("An already loaded predictor must not reload the model")

    monkeypatch.setattr(joblib, "load", forbid_loading)
    record = features.iloc[1].to_dict()
    record["chol"], record["thal"] = None, None
    payload = {"instances": [record]}
    assert predictor.predict(payload) == predictor.predict(payload)
    assert (
        medians
        == predictor.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_
    ).all()


@pytest.mark.parametrize("problem", ["model", "schema", "environment"])
def test_loader_rejects_incompatible_artifact_before_deserializing(tmp_path, monkeypatch, problem):
    _, metadata = saved_model(tmp_path)
    if problem == "model":
        (tmp_path / "model.joblib").write_bytes(b"tampered")
    elif problem == "schema":
        schema_path = tmp_path / "feature_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["fields"] = list(reversed(FEATURES))
        schema_path.write_text(json.dumps(schema))
    else:
        metadata["package_versions"]["scikit-learn"] = "0.0.0"
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Deserialized before validation"))
    with pytest.raises(ValueError):
        load_predictor(tmp_path)


@pytest.mark.parametrize("problem", ["modified", "missing", "extra", "path_escape"])
def test_bundle_integrity_rejects_bad_payload(tmp_path, problem):
    payload = tmp_path / "payload.txt"
    payload.write_text("original", encoding="utf-8")
    manifest = {"format_version": 1, "files": {"payload.txt": checksum(payload)}}
    if problem == "modified":
        payload.write_text("changed", encoding="utf-8")
    elif problem == "missing":
        payload.unlink()
    elif problem == "extra":
        (tmp_path / "extra.txt").write_text("unexpected")
    else:
        manifest["files"] = {"../payload.txt": checksum(payload)}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_files(tmp_path)
