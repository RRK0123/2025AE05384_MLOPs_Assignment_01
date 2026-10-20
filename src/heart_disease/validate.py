"""Validate generated data and preprocessing artifacts from the command line."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, SEED, TEST_SIZE
from .data import clean_data, read_raw
from .preprocessing import build_preprocessor
from .schema import CATEGORIES, FEATURES, NUMERIC, TARGET


def require(condition: bool, message: str):
    """Check invariants even when Python is invoked with optimization enabled."""
    if not condition:
        raise ValueError(message)


def validate_artifacts(root: Path = ROOT):
    raw_path = root / "data/raw/processed.cleveland.data"
    processed = root / "data/processed"
    reports = root / "reports"
    raw = read_raw(raw_path)
    expected = clean_data(raw)
    source = json.loads((root / "data/raw/source.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    require(source_hash == source["sha256"], "Raw source checksum mismatch.")
    saved_clean = pd.read_csv(processed / "heart_clean.csv")
    pd.testing.assert_frame_equal(saved_clean, expected.reset_index(drop=True), check_dtype=False)
    manifest = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    require(manifest["raw_sha256"] == source_hash, "Split refers to a different raw source.")
    require(
        manifest["seed"] == SEED and manifest["test_size"] == TEST_SIZE,
        "Saved evaluation settings differ from configuration.",
    )
    require(manifest["stratify"] == TARGET, "Split target mismatch.")
    train_positions, test_positions = manifest["train_positions"], manifest["test_positions"]
    require(set(train_positions).isdisjoint(test_positions), "Train/test overlap.")
    require(len(train_positions) + len(test_positions) == len(expected), "Incorrect split lengths.")
    require(
        set(train_positions) | set(test_positions) == set(range(len(expected))),
        "Split does not cover every source row exactly once.",
    )
    expected_train, expected_test = train_test_split(
        np.arange(len(expected)),
        random_state=SEED,
        test_size=TEST_SIZE,
        stratify=expected[TARGET],
    )
    require(
        train_positions == sorted(expected_train.tolist())
        and test_positions == sorted(expected_test.tolist()),
        "Split is not reproducible.",
    )
    train = pd.read_csv(processed / "train.csv")
    test = pd.read_csv(processed / "test.csv")
    pd.testing.assert_frame_equal(
        train, expected.iloc[train_positions].reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        test, expected.iloc[test_positions].reset_index(drop=True), check_dtype=False
    )
    transformer = build_preprocessor()
    transformed_train = transformer.fit_transform(train[FEATURES].astype(float))
    transformed_test = transformer.transform(test[FEATURES].astype(float))
    require(list(transformer.feature_names_in_) == FEATURES, "Unexpected input features.")
    names = transformer.get_feature_names_out().tolist()
    require(
        np.isfinite(transformed_train).all() and np.isfinite(transformed_test).all(),
        "Transformed features contain missing or infinite values.",
    )
    for label, values in [("train", transformed_train), ("test", transformed_test)]:
        saved = pd.read_csv(processed / f"{label}_preprocessed_preview.csv")
        require(saved.columns.tolist() == names, f"{label} encoded schema mismatch.")
        np.testing.assert_allclose(saved.to_numpy(), values, rtol=1e-10, atol=1e-10)
    numeric = transformer.named_transformers_["numeric"]
    medians = numeric.named_steps["imputer"].statistics_
    np.testing.assert_allclose(medians, train[NUMERIC].median().to_numpy())
    audit = json.loads((reports / "preprocessing_audit.json").read_text(encoding="utf-8"))
    require(audit["encoded_columns"] == names, "Preprocessing audit schema mismatch.")
    require(audit["output_features"] == len(names), "Preprocessing audit feature-count mismatch.")
    np.testing.assert_allclose([audit["numerical_imputation_medians"][c] for c in NUMERIC], medians)
    modes = transformer.named_transformers_["categorical"].named_steps["imputer"].statistics_
    np.testing.assert_allclose(
        [audit["categorical_imputation_modes"][c] for c in CATEGORIES], modes
    )
    quality = json.loads((reports / "data_quality.json").read_text(encoding="utf-8"))
    require(
        quality["class_counts"]
        == {str(k): int(v) for k, v in expected[TARGET].value_counts().sort_index().items()},
        "Quality report class counts mismatch.",
    )
    require(
        quality["missing_by_column"] == {c: int(n) for c, n in raw.isna().sum().items()},
        "Quality report missingness mismatch.",
    )
    figures = [
        "01_missing_values.png",
        "02_class_balance.png",
        "03_numeric_histograms.png",
        "04_numeric_correlation.png",
        "05_categorical_disease_rates.png",
        "06_numeric_boxplots.png",
        "07_outlier_audit.png",
    ]
    for name in figures:
        content = (reports / "figures" / name).read_bytes()
        require(
            content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) > 1024,
            f"Invalid figure: {name}.",
        )
    require((reports / "eda_findings.md").is_file(), "EDA findings report is missing.")
    result = {
        "python": platform.python_version(),
        "validation_entrypoint": "python -m heart_disease.validate",
        "source_sha256": source_hash,
        "split_covers_source_without_overlap": True,
        "saved_splits_match_source_rows": True,
        "split_reproduced_from_seed": True,
        "preprocessing_fitted_to_training_statistics": True,
        "transformed_shape_train": list(transformed_train.shape),
        "transformed_shape_test": list(transformed_test.shape),
        "target_and_diagnosis_excluded": True,
        "required_png_figures_present": len(figures),
    }
    (reports / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    print(json.dumps(validate_artifacts(), indent=2))


if __name__ == "__main__":
    main()
