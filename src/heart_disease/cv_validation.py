"""Reconcile recorded outer-CV predictions, fold reports, and selection scores."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .modeling import FAMILIES, evaluate_predictions, select_family, summarize_folds
from .schema import TARGET
from .validate import require


@dataclass
class CVEvidence:
    folds: pd.DataFrame
    summary: pd.DataFrame
    definitions: list[dict]


def require_close(actual, expected, message):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, rtol=1e-12, atol=1e-12),
        message,
    )


def _boundary(positions, fitting, validation):
    return {
        "fit_source_positions": positions[fitting].tolist(),
        "validation_source_positions": positions[validation].tolist(),
    }


def reconcile_cv(root, metadata):
    """Recompute evidence without refitting models or using held-out predictions."""
    root = Path(root)
    processed, reports = root / "data/processed", root / metadata["report_directory"]
    for name, key in [
        ("train.csv", "training_csv_sha256"),
        ("split_manifest.json", "split_manifest_sha256"),
    ]:
        require(
            hashlib.sha256((processed / name).read_bytes()).hexdigest() == metadata[key],
            f"CV training evidence changed: {name}",
        )
    train = pd.read_csv(processed / "train.csv")
    split = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    train_positions, test_positions = split["train_positions"], split["test_positions"]
    require(
        len(train_positions) == len(set(train_positions)) == len(train) == metadata["train_rows"],
        "Invalid CV training positions",
    )
    require(
        len(test_positions) == len(set(test_positions)) == metadata["test_rows"]
        and set(train_positions).isdisjoint(test_positions),
        "Invalid CV test boundary",
    )
    require(
        set(train_positions) | set(test_positions)
        == set(range(len(train_positions) + len(test_positions))),
        "CV split does not cover source positions",
    )
    require(
        train[TARGET].isin([0, 1]).all() and train[TARGET].nunique() == 2,
        "Invalid CV source labels",
    )
    positions = np.asarray(train_positions)
    labels = train[TARGET].to_numpy()
    definitions = json.loads((reports / "cv_splits.json").read_text(encoding="utf-8"))
    outer = StratifiedKFold(
        n_splits=metadata["outer_folds"], shuffle=True, random_state=metadata["seed"]
    )
    expected_outer, expected_final = [], []
    for number, (fitting, validation) in enumerate(outer.split(positions, labels), start=1):
        inner = StratifiedKFold(
            n_splits=metadata["inner_folds"], shuffle=True, random_state=metadata["seed"] + number
        )
        expected_outer.append(
            {
                "fold": number,
                **_boundary(positions, fitting, validation),
                "inner_splits": [
                    _boundary(positions[fitting], a, b)
                    for a, b in inner.split(positions[fitting], labels[fitting])
                ],
            }
        )
        expected_final.append(_boundary(positions, fitting, validation))
    require(
        definitions["outer_splits"] == expected_outer,
        "CV outer/inner fold definitions differ from the declared seeded split",
    )
    require(
        definitions["final_tuning_splits"] == expected_final,
        "CV final-tuning fold definitions differ from the declared seeded split",
    )

    tables = reports / "tables"
    oof = pd.read_csv(tables / "out_of_fold_predictions.csv", float_precision="round_trip")
    require(
        set(oof.columns)
        == {"family", "outer_fold", "source_position", TARGET, "prediction", "probability_disease"},
        "Unexpected CV prediction schema",
    )
    families = ["dummy_baseline", *FAMILIES]
    require(set(oof["family"]) == set(families), "Unexpected CV prediction families")
    require(not oof.duplicated(["family", "source_position"]).any(), "Duplicate CV prediction row")
    require(
        oof[TARGET].isin([0, 1]).all() and oof["prediction"].isin([0, 1]).all(),
        "Invalid CV prediction or target labels",
    )
    probability = oof["probability_disease"].to_numpy(dtype=float)
    require(
        np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
        "Invalid CV probabilities",
    )
    truth = dict(zip(train_positions, labels))
    require(
        np.array_equal(oof[TARGET].to_numpy(), oof["source_position"].map(truth).to_numpy()),
        "CV prediction targets differ from source labels",
    )
    records = []
    for family in families:
        rows = oof.loc[oof["family"] == family]
        require(
            len(rows) == len(train) and set(rows["source_position"]) == set(train_positions),
            f"Incomplete CV predictions for {family}",
        )
        require(
            rows["outer_fold"].isin(range(1, metadata["outer_folds"] + 1)).all(),
            f"Invalid CV fold identifier for {family}",
        )
        for definition in expected_outer:
            fold = rows.loc[rows["outer_fold"] == definition["fold"]]
            require(
                set(fold["source_position"]) == set(definition["validation_source_positions"]),
                f"CV prediction fold membership differs for {family}, fold {definition['fold']}",
            )
            records.append(
                {
                    "family": family,
                    "outer_fold": definition["fold"],
                    "fit_rows": len(definition["fit_source_positions"]),
                    "validation_rows": len(fold),
                    **evaluate_predictions(
                        fold[TARGET], fold["prediction"], fold["probability_disease"]
                    ),
                }
            )
    folds = pd.DataFrame(records)
    stored_folds = pd.read_csv(tables / "outer_fold_metrics.csv", float_precision="round_trip")
    require(
        set(stored_folds.columns) == set(folds.columns) | {"inner_best_roc_auc", "best_parameters"},
        "Unexpected CV fold report schema",
    )
    require(
        not stored_folds.duplicated(["family", "outer_fold"]).any(), "Duplicate CV fold report row"
    )
    expected = folds.set_index(["family", "outer_fold"]).sort_index()
    actual = stored_folds.set_index(["family", "outer_fold"]).sort_index()
    require(actual.index.equals(expected.index), "Missing or unexpected CV fold report row")
    require_close(actual[expected.columns], expected, "CV fold report differs from predictions")
    summary = summarize_folds(folds)
    stored_summary = pd.read_csv(tables / "model_comparison.csv", float_precision="round_trip")
    require(set(stored_summary.columns) == set(summary.columns), "Unexpected CV comparison schema")
    require(not stored_summary["family"].duplicated().any(), "Duplicate CV comparison family")
    expected = summary.set_index("family").sort_index()
    actual = stored_summary.set_index("family").sort_index()
    require(actual.index.equals(expected.index), "Missing or unexpected CV comparison family")
    require_close(
        actual[expected.columns], expected, "CV comparison report differs from predictions"
    )
    require(
        select_family(summary) == metadata["selected_family"],
        "Selected family differs from prediction-derived CV scores",
    )
    return CVEvidence(folds, summary, expected_outer)
