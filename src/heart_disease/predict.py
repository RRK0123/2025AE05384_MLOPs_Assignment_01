"""Validate JSON records and run a saved pipeline without plotting dependencies."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .modeling import disease_probabilities
from .releases import resolve_model_directory
from .schema import CATEGORIES, FEATURES, NUMERIC


def feature_frame(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or set(payload) != {"instances"}:
        raise ValueError("Input must contain only an 'instances' list.")
    records = payload["instances"]
    if not isinstance(records, list) or not records:
        raise ValueError("instances must be a non-empty list.")
    for number, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(FEATURES):
            raise ValueError(f"Instance {number} must have exactly the 13 documented input fields.")
        for field, value in record.items():
            if value is None:
                continue
            try:
                finite_number = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                )
            except OverflowError:
                finite_number = False
            if not finite_number:
                raise ValueError(f"Instance {number}, {field}: expected a finite number or null.")
            if field in CATEGORIES and value not in CATEGORIES[field]:
                raise ValueError(f"Instance {number}, {field}: unsupported category code {value}.")
            if field in NUMERIC and (value < 0 or (field != "oldpeak" and value == 0)):
                raise ValueError(f"Instance {number}, {field}: invalid measurement.")
    return pd.DataFrame(records, columns=FEATURES).astype(float)


@dataclass
class Predictor:
    """Load once, then reuse the fitted pipeline for many inference requests."""

    pipeline: Pipeline
    metadata: dict

    def predict(self, payload: dict) -> dict:
        features = feature_frame(payload)
        prediction = self.pipeline.predict(features)
        probability = disease_probabilities(self.pipeline, features)
        return {
            "model_run_id": self.metadata["run_id"],
            "predictions": [
                {
                    "prediction": int(label),
                    "probability_disease": float(score),
                    "confidence": float(score if label == 1 else 1 - score),
                }
                for label, score in zip(prediction, probability)
            ],
        }


def load_predictor(model_directory: Path | None = None) -> Predictor:
    model_directory = resolve_model_directory(model_directory)
    metadata = json.loads((model_directory / "metadata.json").read_text(encoding="utf-8"))
    artifact = model_directory / "model.joblib"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch.")
    if metadata["input_features"] != FEATURES:
        raise ValueError("Model feature schema does not match the application.")
    schema = json.loads((model_directory / "feature_schema.json").read_text(encoding="utf-8"))
    if (
        schema["fields"] != FEATURES
        or schema["numeric_fields"] != NUMERIC
        or schema["categorical_codes"]
        != {name: list(levels) for name, levels in CATEGORIES.items()}
    ):
        raise ValueError("Packaged feature schema does not match the application.")
    for package in ["numpy", "pandas", "scikit-learn", "joblib"]:
        if importlib.metadata.version(package) != metadata["package_versions"][package]:
            raise ValueError(
                f"Incompatible {package} version; install the pinned runtime requirements."
            )
    model = joblib.load(artifact)
    if (
        not isinstance(model, Pipeline)
        or list(model.named_steps) != ["preprocessor", "classifier"]
        or list(model.feature_names_in_) != FEATURES
        or model.classes_.tolist() != [0, 1]
        or len(model.named_steps["preprocessor"].get_feature_names_out())
        != metadata["encoded_feature_count"]
    ):
        raise ValueError("Saved pipeline structure does not match the model metadata.")
    return Predictor(model, metadata)


def predict_payload(payload: dict, model_directory: Path | None = None) -> dict:
    return load_predictor(model_directory).predict(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON file with an instances list."
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(predict_payload(payload, args.model_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
