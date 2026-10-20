"""Run reproducible EDA from the command line."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, SEED, TEST_SIZE
from .data import clean_data, read_raw
from .download import download
from .plotting import generate_figures
from .preprocessing import build_preprocessor
from .reporting import write_findings
from .schema import CATEGORIES, FEATURES, NUMERIC, TARGET


def run_eda():
    path = download()
    raw = read_raw(path)
    cleaned = clean_data(raw)
    train_positions, test_positions = train_test_split(
        np.arange(len(cleaned)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=cleaned[TARGET],
    )
    train_positions, test_positions = np.sort(train_positions), np.sort(test_positions)
    if set(train_positions) & set(test_positions):
        raise AssertionError("Training and test row positions overlap.")
    if set(train_positions) | set(test_positions) != set(range(len(cleaned))):
        raise AssertionError("Split does not cover the source rows.")
    # Exact source duplicates could otherwise cross the evaluation boundary.
    if raw.duplicated().any():
        raise ValueError(
            "Duplicate source rows require a grouped split decision before proceeding."
        )
    train, test = cleaned.iloc[train_positions].copy(), cleaned.iloc[test_positions].copy()
    processed, reports = ROOT / "data" / "processed", ROOT / "reports"
    figures, tables = reports / "figures", reports / "tables"
    for directory in [processed, figures, tables]:
        directory.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed / "heart_clean.csv", index=False)
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)
    split = {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "stratify": TARGET,
        "index_convention": "zero-based positions in the original headerless source file",
        "train_positions": train_positions.tolist(),
        "test_positions": test_positions.tolist(),
    }
    (processed / "split_manifest.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    quality = {
        "rows": len(raw),
        "predictors": len(FEATURES),
        "missing_cells": int(raw.isna().sum().sum()),
        "rows_with_missing": int(raw.isna().any(axis=1).sum()),
        "missing_by_column": {c: int(n) for c, n in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "original_diagnosis_counts": {
            str(k): int(v) for k, v in raw["num"].value_counts().sort_index().items()
        },
        "class_counts": {
            str(k): int(v) for k, v in cleaned[TARGET].value_counts().sort_index().items()
        },
        "train_class_counts": {
            str(k): int(v) for k, v in train[TARGET].value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in test[TARGET].value_counts().sort_index().items()
        },
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (reports / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    train[NUMERIC].describe().T.to_csv(tables / "numeric_summary_train.csv", index_label="feature")
    train.groupby(TARGET)[NUMERIC].median().to_csv(tables / "numeric_medians_by_target_train.csv")
    train[list(CATEGORIES)].apply(lambda series: series.value_counts()).to_csv(
        tables / "category_counts_train.csv", index_label="code"
    )

    transformer = build_preprocessor()
    train_features = transformer.fit_transform(train[FEATURES].astype(float))
    test_features = transformer.transform(test[FEATURES].astype(float))
    names = transformer.get_feature_names_out()
    if list(transformer.feature_names_in_) != FEATURES:
        raise AssertionError("Diagnosis or target leaked into features.")
    if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
        raise AssertionError("Preprocessing left non-finite values.")
    pd.DataFrame(train_features, columns=names).to_csv(
        processed / "train_preprocessed_preview.csv", index=False
    )
    pd.DataFrame(test_features, columns=names).to_csv(
        processed / "test_preprocessed_preview.csv", index=False
    )
    audit = {
        "scope": "EDA demonstration only; refit inside each model CV fold later",
        "fitted_on": "train.csv only",
        "encoded_columns": names.tolist(),
        "output_features": len(names),
        "non_finite_values": 0,
        "numerical_imputation_medians": dict(
            zip(
                NUMERIC,
                transformer.named_transformers_["numeric"]
                .named_steps["imputer"]
                .statistics_.tolist(),
            )
        ),
        "categorical_imputation_modes": dict(
            zip(
                CATEGORIES,
                transformer.named_transformers_["categorical"]
                .named_steps["imputer"]
                .statistics_.tolist(),
            )
        ),
        "scaled_training_means": transformer.named_transformers_["numeric"]
        .named_steps["scaler"]
        .mean_.tolist(),
    }
    (reports / "preprocessing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    generate_figures(raw, cleaned, train, figures, tables)
    write_findings(raw, cleaned, train, test, quality, reports)
    print(json.dumps(quality, indent=2))
    print(f"EDA complete: {figures}\nFindings: {reports / 'eda_findings.md'}")
    return quality


def main():
    run_eda()


if __name__ == "__main__":
    main()
