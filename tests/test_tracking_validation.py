"""Validation guards must remain active under Python optimization."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from cv_fixture import write_cv_fixture
from tuning_fixture import write_tuning_fixture

from heart_disease.config import ROOT
from heart_disease.modeling import (
    FAMILIES,
    METRICS,
    build_model,
    evaluate_predictions,
    summarize_folds,
)
from heart_disease.tracking import ExperimentTracker, local_uri


@pytest.fixture(scope="session")
def tracking_fixture(tmp_path_factory):
    """Synthetic complete lineage; no checkout database, models or artifact URIs."""
    root = tmp_path_factory.mktemp("tracking-fixture")
    metadata, folds_table, summary, definitions = write_cv_fixture(root)
    searches = write_tuning_fixture(root, metadata)
    fold_scores = folds_table.set_index(["family", "outer_fold"])
    tracker = ExperimentTracker("fixture-session", root=root)
    client = tracker.client
    reference = {
        "experiment_id": tracker.experiment_id,
        "experiment_name": "heart-disease-classification",
        "final_model_run_ids": {},
        "model_uris": {},
    }
    with tracker.session_run():
        reference["parent_run_id"] = tracker.parent_id
        tracker.parameters(
            tracker.parent_id,
            {key: metadata[key] for key in ["selected_family", "selected_parameters"]},
        )
        tracker.scores(tracker.parent_id, {"selected_test_accuracy": 0.75})
        for row in summary.to_dict("records"):
            tracker.scores(
                tracker.parent_id,
                {
                    f"{row['family']}_outer_{metric}_{statistic}": row[f"{metric}_{statistic}"]
                    for metric in METRICS
                    for statistic in ["mean", "std"]
                },
            )
        for name in [
            "reports/figures/01_nested_cv_comparison.png",
            "reports/figures/02_selected_test_curves.png",
            "reports/figures/03_selected_test_confusion.png",
            "metadata.json",
            "environment/requirements-lock.txt",
            "data/split_manifest.json",
        ]:
            client.log_text(tracker.parent_id, "synthetic test artifact", name)
        for phase, folds in [("outer_fold", range(1, 6)), ("final_model", [None])]:
            for fold in folds:
                families = ["dummy_baseline", *FAMILIES] if fold is not None else FAMILIES
                for family in families:
                    with tracker.run(
                        f"{phase}-{family}-{fold}", phase, family, tracker.parent_id, fold
                    ) as run:
                        model = build_model(family)
                        if family in FAMILIES:
                            model.set_params(**searches[family, fold].cv_results_["params"][0])
                        tracker.parameters(
                            run, {"fixture": True, **model.named_steps["classifier"].get_params()}
                        )
                        if phase != "outer_fold":
                            search = searches[family, None]
                            tracker.parameters(
                                run,
                                {
                                    "cv_folds": search.n_splits_,
                                    "training_rows": metadata["train_rows"],
                                    "refit_metric": "roc_auc",
                                },
                            )
                            tracker.scores(
                                run,
                                {
                                    f"final_cv_{metric}": float(
                                        search.cv_results_[f"mean_test_{metric}"][0]
                                    )
                                    for metric in METRICS
                                },
                            )
                        tracker.plot(run, {"accuracy": 0.75}, "Synthetic scores")
                        if phase == "outer_fold":
                            scores = fold_scores.loc[(family, fold)]
                            tracker.parameters(
                                run,
                                {key: int(scores[key]) for key in ["fit_rows", "validation_rows"]},
                            )
                            tracker.scores(
                                run,
                                {
                                    f"outer_{key}": float(value)
                                    for key, value in scores.items()
                                    if key
                                    not in [
                                        "fit_rows",
                                        "validation_rows",
                                        "inner_best_roc_auc",
                                        "best_parameters",
                                    ]
                                },
                            )
                            client.log_dict(
                                run, definitions["outer_splits"][fold - 1], "fold_definition.json"
                            )
                        else:
                            client.set_tag(run, "model_roundtrip_verified", "true")
                            client.log_text(run, "synthetic model", f"pipeline/{family}.joblib")
                            model = client.create_logged_model(
                                tracker.experiment_id, name=family, source_run_id=run
                            )
                            client.finalize_logged_model(model.model_id, "READY")
                            reference["final_model_run_ids"][family] = run
                            reference["model_uris"][family] = f"models:/{model.model_id}"
                        if family == "dummy_baseline":
                            continue
                        tracker.search_candidates(
                            searches[family, fold],
                            family,
                            "inner_cv" if fold is not None else "final_cv",
                            run,
                            fold,
                        )
    metadata.update(
        {
            "run_id": tracker.session,
            "outer_folds": 5,
            "report_directory": "reports/modeling/fixture-session",
            "test_metrics": {"accuracy": 0.75},
            "experiment_tracking": reference,
        }
    )
    return root, metadata


@pytest.fixture
def tracking_workspace(tmp_path, tracking_fixture):
    source_root, metadata = tracking_fixture
    (tmp_path / "tracking").mkdir()
    (tmp_path / "models/selected").mkdir(parents=True)
    (tmp_path / "models/selected/metadata.json").write_text(json.dumps(metadata))
    shutil.copytree(
        source_root / metadata["report_directory"], tmp_path / metadata["report_directory"]
    )
    shutil.copytree(source_root / "data", tmp_path / "data")
    with sqlite3.connect(
        (source_root / "tracking/mlflow.db").as_uri() + "?mode=ro", uri=True
    ) as source:
        with sqlite3.connect(tmp_path / "tracking/mlflow.db") as destination:
            source.backup(destination)
    return tmp_path, metadata


def validate_in_subprocess(root, optimized):
    env = {**os.environ, "HEART_DISEASE_PROJECT_ROOT": str(root)}
    env.pop("PYTHONOPTIMIZE", None)
    return subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-m", "heart_disease.validate_tracking"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_complete_tracking_passes_with_and_without_optimization(tracking_workspace, optimized):
    root, _ = tracking_workspace
    result = validate_in_subprocess(root, optimized)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run_count"] == 102


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("failure", ["missing_runs", "failed_run"])
def test_bad_tracking_rejected_even_with_optimization(tracking_workspace, optimized, failure):
    root, metadata = tracking_workspace
    with sqlite3.connect(root / "tracking/mlflow.db") as store:
        baseline = [
            row[0]
            for row in store.execute(
                "SELECT a.run_uuid FROM tags a JOIN tags b ON a.run_uuid=b.run_uuid "
                "WHERE a.key='training_session' AND a.value=? "
                "AND b.key='family' AND b.value='dummy_baseline'",
                [metadata["run_id"]],
            )
        ]
        assert len(baseline) == 5
        if failure == "missing_runs":
            store.executemany(
                "UPDATE runs SET lifecycle_stage='deleted' WHERE run_uuid=?",
                [(run,) for run in baseline],
            )
        else:
            store.execute("UPDATE runs SET status='FAILED' WHERE run_uuid=?", [baseline[0]])
    result = validate_in_subprocess(root, optimized)
    assert result.returncode != 0
    if failure == "missing_runs":
        assert "Incomplete tracking lineage" in result.stderr
    else:
        assert "is FAILED" in result.stderr


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("failure", ["outer_metric", "parent_summary", "fold_tag", "row_count"])
def test_tracked_cv_mismatches_are_rejected(tracking_workspace, optimized, failure):
    from mlflow import MlflowClient

    root, metadata = tracking_workspace
    client = MlflowClient(tracking_uri=local_uri(root))
    outer = client.search_runs(
        [metadata["experiment_tracking"]["experiment_id"]],
        "tags.phase = 'outer_fold' AND tags.family = 'logistic_regression'",
    )[0]
    if failure == "outer_metric":
        client.log_metric(outer.info.run_id, "outer_roc_auc", 0.99)
    elif failure == "parent_summary":
        client.log_metric(
            metadata["experiment_tracking"]["parent_run_id"],
            "logistic_regression_outer_roc_auc_mean",
            0.99,
        )
    elif failure == "fold_tag":
        client.set_tag(outer.info.run_id, "outer_fold", "99")
    else:
        # MLflow parameters are immutable through its API; alter only the copied store.
        with sqlite3.connect(root / "tracking/mlflow.db") as store:
            store.execute(
                "UPDATE params SET value='999' WHERE run_uuid=? AND key='validation_rows'",
                [outer.info.run_id],
            )
    result = validate_in_subprocess(root, optimized)
    assert result.returncode != 0
    assert "Tracked CV" in result.stderr or "tracked CV" in result.stderr


@pytest.mark.parametrize("optimized", [False, True])
def test_coordinated_csv_changes_are_rejected_against_tracking(tracking_workspace, optimized):
    import pandas as pd

    root, metadata = tracking_workspace
    tables = root / metadata["report_directory"] / "tables"
    predictions = pd.read_csv(tables / "out_of_fold_predictions.csv")
    predictions["prediction"], predictions["probability_disease"] = 0, 0.5
    predictions.to_csv(tables / "out_of_fold_predictions.csv", index=False)
    folds = pd.read_csv(tables / "outer_fold_metrics.csv")
    for index, row in folds.iterrows():
        rows = predictions.loc[
            (predictions.family == row.family) & (predictions.outer_fold == row.outer_fold)
        ]
        for metric, value in evaluate_predictions(
            rows.target, rows.prediction, rows.probability_disease
        ).items():
            folds.loc[index, metric] = value
    folds.to_csv(tables / "outer_fold_metrics.csv", index=False)
    summarize_folds(folds).to_csv(tables / "model_comparison.csv", index=False)
    result = validate_in_subprocess(root, optimized)
    assert result.returncode != 0
    assert "Tracked CV comparison differs from prediction-derived scores" in result.stderr


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "parent_parameters",
        "final_parameters",
        "candidate_std",
        "candidate_history",
        "best_tag",
        "candidate_json_parameters",
        "candidate_json_summary",
        "coordinated_table",
    ],
)
def test_tuning_tracking_mismatches_are_rejected(tracking_workspace, optimized, failure):
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    import pandas as pd
    from mlflow import MlflowClient

    root, metadata = tracking_workspace
    client = MlflowClient(tracking_uri=local_uri(root))
    final_id = metadata["experiment_tracking"]["final_model_run_ids"]["logistic_regression"]
    candidates = client.search_runs(
        [metadata["experiment_tracking"]["experiment_id"]],
        f"tags.mlflow.parentRunId = '{final_id}'",
    )
    best = next(run for run in candidates if run.data.tags["best_candidate"] == "True")
    if failure in ["parent_parameters", "final_parameters"]:
        run_id = (
            metadata["experiment_tracking"]["parent_run_id"]
            if failure == "parent_parameters"
            else final_id
        )
        key = "selected_parameters" if failure == "parent_parameters" else "C"
        value = json.dumps({"classifier__C": 999.0}) if failure == "parent_parameters" else "999.0"
        with sqlite3.connect(root / "tracking/mlflow.db") as store:
            store.execute(
                "UPDATE params SET value=? WHERE run_uuid=? AND key=?", [value, run_id, key]
            )
    elif failure == "candidate_std":
        client.log_metric(best.info.run_id, "cv_roc_auc_std", 0.5)
    elif failure == "candidate_history":
        with sqlite3.connect(root / "tracking/mlflow.db") as store:
            store.execute(
                "UPDATE metrics SET value=value+CASE step WHEN 1 THEN 0.01 ELSE -0.01 END "
                "WHERE run_uuid=? AND key='cv_roc_auc' AND step IN (1,2)",
                [best.info.run_id],
            )
    elif failure == "best_tag":
        other = next(run for run in candidates if run.info.run_id != best.info.run_id)
        client.set_tag(best.info.run_id, "best_candidate", "False")
        client.set_tag(other.info.run_id, "best_candidate", "True")
    elif failure.startswith("candidate_json"):
        # Copy only this candidate's artifacts before changing its database URI.
        # The session fixture's shared original artifacts remain untouched.
        original = Path(url2pathname(urlparse(best.info.artifact_uri).path))
        private = root / "private-candidate-artifacts"
        shutil.copytree(original, private)
        with sqlite3.connect(root / "tracking/mlflow.db") as store:
            store.execute(
                "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                [private.as_uri(), best.info.run_id],
            )
        path = private / "cross_validation.json"
        content = json.loads(path.read_text())
        if failure == "candidate_json_parameters":
            content["parameters"]["classifier__C"] = 999.0
        else:
            content["summary"]["roc_auc"]["mean"] = 0.99999
        path.write_text(json.dumps(content))
    else:
        path = root / metadata["report_directory"] / "tables/logistic_regression_final_search.csv"
        table = pd.read_csv(path)
        for fold in range(metadata["outer_folds"]):
            table.loc[0, f"split{fold}_test_roc_auc"] = 0.99999
        table.loc[0, "mean_test_roc_auc"], table.loc[0, "std_test_roc_auc"] = 0.99999, 0.0
        table["rank_test_roc_auc"] = table.mean_test_roc_auc.rank(method="min", ascending=False)
        table.to_csv(path, index=False)
        metadata = json.loads(json.dumps(metadata))
        metadata["final_searches"]["logistic_regression"]["tuning_roc_auc"] = 0.99999
        (root / "models/selected/metadata.json").write_text(json.dumps(metadata))
    result = validate_in_subprocess(root, optimized)
    assert result.returncode != 0
    assert "Tracked tuning" in result.stderr, result.stderr
