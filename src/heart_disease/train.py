"""Tune, compare, select, and save classification pipelines reproducibly."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from .config import ROOT, SEED
from .model_report import model_figures, write_model_report
from .modeling import (
    FAMILIES,
    GRIDS,
    build_model,
    build_search,
    disease_probabilities,
    evaluate_predictions,
    select_family,
    summarize_folds,
)
from .releases import promote_release
from .schema import CATEGORIES, FEATURES, NUMERIC, TARGET
from .tracking import ExperimentTracker
from .validate import validate_artifacts

OUTER_FOLDS = 5
INNER_FOLDS = 3


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_table(search, path):
    table = pd.DataFrame(search.cv_results_)
    table["params"] = table["params"].map(lambda value: json.dumps(value, sort_keys=True))
    table.to_csv(path, index=False)


def original_positions(train_positions, local_positions):
    return [int(train_positions[i]) for i in local_positions]


def fold_definition(train_positions, fitting, validation):
    return {
        "fit_source_positions": original_positions(train_positions, fitting),
        "validation_source_positions": original_positions(train_positions, validation),
    }


def nested_comparison(features, target, train_positions, tables, jobs, tracker):
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    splits = list(outer.split(features, target))
    results, predictions, definitions = [], [], []
    for number, (fitting, validation) in enumerate(splits, start=1):
        fold = {"fold": number, **fold_definition(train_positions, fitting, validation)}
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + number)
        fold["inner_splits"] = [
            fold_definition(train_positions, fitting[a], fitting[b])
            for a, b in inner.split(features.iloc[fitting], target.iloc[fitting])
        ]
        definitions.append(fold)
        for family in ["dummy_baseline", *FAMILIES]:
            print(f"Outer fold {number}/{OUTER_FOLDS}: {family}", flush=True)
            search = None
            search_path = tables / f"{family}_outer{number}_search.csv"
            if family == "dummy_baseline":
                model = build_model(family).fit(features.iloc[fitting], target.iloc[fitting])
                parameters, tuning_auc = {}, None
            else:
                search = build_search(family, INNER_FOLDS, SEED + number, jobs)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    search.fit(features.iloc[fitting], target.iloc[fitting])
                model = search.best_estimator_
                parameters, tuning_auc = search.best_params_, float(search.best_score_)
                search_table(search, search_path)
            probability = disease_probabilities(model, features.iloc[validation])
            prediction = model.predict(features.iloc[validation])
            metrics = evaluate_predictions(target.iloc[validation], prediction, probability)
            tracker.outer_fold(
                family, number, model, metrics, fold, search, search_path, len(fitting)
            )
            results.append(
                {
                    "family": family,
                    "outer_fold": number,
                    "fit_rows": len(fitting),
                    "validation_rows": len(validation),
                    "inner_best_roc_auc": tuning_auc,
                    "best_parameters": json.dumps(parameters, sort_keys=True),
                    **metrics,
                }
            )
            for local, truth, label, score in zip(
                validation, target.iloc[validation], prediction, probability
            ):
                predictions.append(
                    {
                        "family": family,
                        "outer_fold": number,
                        "source_position": int(train_positions[local]),
                        "target": int(truth),
                        "prediction": int(label),
                        "probability_disease": float(score),
                    }
                )
            print(
                f"  Validation ROC-AUC={metrics['roc_auc']:.4f}, recall={metrics['recall']:.4f}",
                flush=True,
            )
    folds = pd.DataFrame(results)
    folds.to_csv(tables / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(tables / "out_of_fold_predictions.csv", index=False)
    summary = summarize_folds(folds)
    summary.to_csv(tables / "model_comparison.csv", index=False)
    return summary, definitions


def train_models(jobs: int = 2):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tracker = ExperimentTracker(run_id)
    with tracker.session_run():
        metadata = _train_models(jobs, run_id, tracker)
    from .package_model import build_bundle
    from .validate_model import validate_model
    from .validate_tracking import validate_tracking

    try:
        candidate = ROOT / metadata["serving_directory"]
        model_validation = validate_model(ROOT, candidate)
        validation = validate_tracking(metadata)
        reports = ROOT / metadata["report_directory"]
        for name in ["validation.json", "tracking_validation.json", "tracking_runs.csv"]:
            tracker.client.log_artifact(tracker.parent_id, str(reports / name), "validation")
        bundle = build_bundle(model_directory=candidate)
        tracker.client.log_artifact(tracker.parent_id, str(bundle), "release")
        release = promote_release(candidate, model_validation, validation, bundle, root=ROOT)
    except BaseException:
        tracker.client.set_terminated(tracker.parent_id, "FAILED")
        raise
    print(
        f"MLflow: {validation['run_count']} verified runs; selected release {release}", flush=True
    )
    return metadata


def _train_models(jobs, run_id, tracker):
    if jobs < 1:
        raise ValueError("jobs must be a positive integer.")
    validate_artifacts()
    processed = ROOT / "data/processed"
    train = pd.read_csv(processed / "train.csv")
    split = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    features, target = train[FEATURES].astype(float), train[TARGET]
    reports = ROOT / "reports/modeling" / run_id
    models = ROOT / "models/runs" / run_id
    tables, figures = reports / "tables", reports / "figures"
    for directory in [tables, figures, models]:
        directory.mkdir(parents=True, exist_ok=False)
    print(
        f"Run {run_id}: {len(features)} training records; nested CV only for selection.", flush=True
    )
    summary, outer_definitions = nested_comparison(
        features, target, split["train_positions"], tables, jobs, tracker
    )
    selected = select_family(summary)
    print(f"Selected before test evaluation: {selected}", flush=True)
    final_searches, fitted = {}, {}
    final_folds = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    final_definitions = [
        fold_definition(split["train_positions"], a, b)
        for a, b in final_folds.split(features, target)
    ]
    for family in FAMILIES:
        print(f"Final development-set tuning: {family}", flush=True)
        search = build_search(family, OUTER_FOLDS, SEED, jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(features, target)
        fitted[family] = search.best_estimator_
        search_table(search, tables / f"{family}_final_search.csv")
        final_searches[family] = {
            "parameters": search.best_params_,
            "tuning_roc_auc": float(search.best_score_),
            "candidate_count": len(ParameterGrid(GRIDS[family])),
        }
        artifact = models / f"{family}.joblib"
        joblib.dump(search.best_estimator_, artifact, compress=3)
        reloaded = joblib.load(artifact)
        np.testing.assert_allclose(
            reloaded.predict_proba(features),
            search.best_estimator_.predict_proba(features),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            reloaded.predict(features), search.best_estimator_.predict(features)
        )
        final_searches[family]["artifact_sha256"] = checksum(artifact)
        tracker.final_model(
            family, search, artifact, tables / f"{family}_final_search.csv", features
        )
    # The model family and all settings are now frozen; only now score test labels.
    test = pd.read_csv(processed / "test.csv")
    test_features, test_target = test[FEATURES].astype(float), test[TARGET]
    model = fitted[selected]
    probability = disease_probabilities(model, test_features)
    prediction = model.predict(test_features)
    test_metrics = evaluate_predictions(test_target, prediction, probability)
    pd.DataFrame(
        {
            "source_position": split["test_positions"],
            "target": test_target,
            "prediction": prediction,
            "probability_disease": probability,
        }
    ).to_csv(tables / "selected_test_predictions.csv", index=False)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_family": selected,
        "selection_rule": "Maximum mean outer-fold ROC-AUC; exact ties prefer Logistic Regression",
        "selection_data": "train.csv only; test metrics do not enter model or parameter selection",
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "input_features": FEATURES,
        "encoded_feature_count": len(model.named_steps["preprocessor"].get_feature_names_out()),
        "numeric_scaling": selected == "logistic_regression",
        "target_mapping": "num == 0 -> 0; num > 0 -> 1",
        "positive_class": 1,
        "prediction_policy": "classifier.predict; default 0.5 probability boundary",
        "probabilities_calibrated": False,
        "source_sha256": split["raw_sha256"],
        "split_manifest_sha256": checksum(processed / "split_manifest.json"),
        "training_csv_sha256": checksum(processed / "train.csv"),
        "test_csv_sha256": checksum(processed / "test.csv"),
        "code_sha256": {
            p.name: checksum(p) for p in sorted((ROOT / "src/heart_disease").glob("*.py"))
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["pandas", "numpy", "scikit-learn", "joblib", "matplotlib", "seaborn"]
        },
        "parameter_grids": GRIDS,
        "final_searches": final_searches,
        "selected_parameters": final_searches[selected]["parameters"],
        "test_metrics": test_metrics,
        "model_sha256": final_searches[selected]["artifact_sha256"],
        "serialization_roundtrip_verified": True,
        "report_directory": reports.relative_to(ROOT).as_posix(),
        "model_directory": models.relative_to(ROOT).as_posix(),
        "serving_directory": (models / "serving").relative_to(ROOT).as_posix(),
        "experiment_tracking": tracker.reference(),
    }
    write_json(
        reports / "cv_splits.json",
        {"outer_splits": outer_definitions, "final_tuning_splits": final_definitions},
    )
    write_json(models / "metadata.json", metadata)
    write_json(reports / "metrics.json", metadata)
    model_figures(summary, test_target, prediction, probability, selected, figures)
    write_model_report(summary, metadata, reports)
    selected_directory = models / "serving"
    selected_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(models / f"{selected}.joblib", selected_directory / "model.joblib")
    write_json(selected_directory / "metadata.json", metadata)
    write_json(
        selected_directory / "feature_schema.json",
        {
            "fields": FEATURES,
            "numeric_fields": NUMERIC,
            "categorical_codes": {c: list(levels) for c, levels in CATEGORIES.items()},
            "missing_values": "JSON null; imputed by the fitted model pipeline",
            "excluded": ["num", "target"],
        },
    )
    sample = {
        name: None if pd.isna(train.iloc[0][name]) else float(train.iloc[0][name])
        for name in FEATURES
    }
    write_json(selected_directory / "sample_request.json", {"instances": [sample]})
    tracker.complete(metadata, summary, reports, models)
    print(
        json.dumps(
            {
                "selected_family": selected,
                "test_metrics": test_metrics,
                "candidate_model": str(selected_directory / "model.joblib"),
                "report": str(reports / "model_development.md"),
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", type=int, default=2, help="Concurrent search workers; positive integer."
    )
    train_models(parser.parse_args().jobs)


if __name__ == "__main__":
    main()
