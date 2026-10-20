"""Small independent CV evidence shared by validator and tracking regressions."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from heart_disease.modeling import FAMILIES, evaluate_predictions, select_family, summarize_folds


def write_cv_fixture(root):
    metadata = {
        "seed": 42,
        "outer_folds": 5,
        "inner_folds": 3,
        "train_rows": 40,
        "test_rows": 10,
        "report_directory": "reports/modeling/fixture-session",
    }
    processed = root / "data/processed"
    tables = root / metadata["report_directory"] / "tables"
    processed.mkdir(parents=True)
    tables.mkdir(parents=True)
    positions, target = np.arange(40), np.tile([0, 1], 20)
    pd.DataFrame({"target": target}).to_csv(processed / "train.csv", index=False)
    split = {"train_positions": positions.tolist(), "test_positions": list(range(40, 50))}
    (processed / "split_manifest.json").write_text(json.dumps(split))

    def boundary(rows, fitting, validation):
        return {
            "fit_source_positions": rows[fitting].tolist(),
            "validation_source_positions": rows[validation].tolist(),
        }

    definitions, final, records, predictions = [], [], [], []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for number, (fitting, validation) in enumerate(outer.split(positions, target), 1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + number)
        definition = {
            "fold": number,
            **boundary(positions, fitting, validation),
            "inner_splits": [
                boundary(positions[fitting], a, b)
                for a, b in inner.split(positions[fitting], target[fitting])
            ],
        }
        definitions.append(definition)
        final.append(boundary(positions, fitting, validation))
        for family in ["dummy_baseline", *FAMILIES]:
            label = target[validation].copy()
            if family == "dummy_baseline":
                label[:] = 0
            else:
                label[:: 7 if family == "logistic_regression" else 3] ^= 1
            probability = np.where(label == 1, 0.8, 0.2)
            records.append(
                {
                    "family": family,
                    "outer_fold": number,
                    "fit_rows": len(fitting),
                    "validation_rows": len(validation),
                    "inner_best_roc_auc": None if family == "dummy_baseline" else 0.75,
                    "best_parameters": "{}",
                    **evaluate_predictions(target[validation], label, probability),
                }
            )
            predictions.extend(
                {
                    "family": family,
                    "outer_fold": number,
                    "source_position": int(position),
                    "target": int(truth),
                    "prediction": int(prediction),
                    "probability_disease": float(score),
                }
                for position, truth, prediction, score in zip(
                    validation, target[validation], label, probability
                )
            )
    definitions = {"outer_splits": definitions, "final_tuning_splits": final}
    (tables.parent / "cv_splits.json").write_text(json.dumps(definitions))
    folds = pd.DataFrame(records)
    summary = summarize_folds(folds)
    folds.to_csv(tables / "outer_fold_metrics.csv", index=False)
    summary.to_csv(tables / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(tables / "out_of_fold_predictions.csv", index=False)
    metadata.update(
        selected_family=select_family(summary),
        **{
            key: hashlib.sha256((processed / name).read_bytes()).hexdigest()
            for name, key in [
                ("train.csv", "training_csv_sha256"),
                ("split_manifest.json", "split_manifest_sha256"),
            ]
        },
    )
    return metadata, folds, summary, definitions
