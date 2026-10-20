"""Verify MLflow lineage and export readable experiment evidence."""

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from sklearn.model_selection import ParameterGrid

from .config import ROOT
from .cv_validation import reconcile_cv, require_close
from .modeling import FAMILIES, GRIDS, METRICS
from .releases import resolve_model_directory
from .tracking import local_uri
from .tuning_validation import reconcile_tracked_searches, reconcile_tuning
from .validate import require


def artifact_paths(client, run_id, path="", repository=None):
    # Use the explicit artifact URI: MLflow 3.17's virtual runs:/ model listing
    # performs a global/default-store lookup even with a client tracking URI.
    repository = repository or get_artifact_repository(
        client.get_run(run_id).info.artifact_uri, tracking_uri=client.tracking_uri
    )
    paths = set()
    for item in repository.list_artifacts(path):
        if item.is_dir:
            paths.update(artifact_paths(client, run_id, item.path, repository))
        else:
            paths.add(item.path)
    return paths


def artifact_json(client, run, name):
    repository = get_artifact_repository(run.info.artifact_uri, tracking_uri=client.tracking_uri)
    with tempfile.TemporaryDirectory(prefix="heart-cv-artifact-") as temporary:
        path = repository.download_artifacts(name, dst_path=temporary)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_tracking(metadata=None, root=ROOT):
    root = Path(root).resolve()
    if metadata is None:
        metadata = json.loads(
            (resolve_model_directory(root=root) / "metadata.json").read_text(encoding="utf-8")
        )
    reference = metadata["experiment_tracking"]
    cv = reconcile_cv(root, metadata)
    searches = reconcile_tuning(root, metadata)
    cv_folds = cv.folds.set_index(["family", "outer_fold"])
    client = MlflowClient(tracking_uri=local_uri(root))
    runs = client.search_runs(
        [reference["experiment_id"]],
        filter_string=f"tags.training_session = '{metadata['run_id']}'",
        max_results=10000,
    )
    by_id = {run.info.run_id: run for run in runs}
    phases = Counter(run.data.tags["phase"] for run in runs)
    candidates = sum(len(ParameterGrid(GRIDS[family])) for family in FAMILIES)
    expected = {
        "training_session": 1,
        "outer_fold": metadata["outer_folds"] * (len(FAMILIES) + 1),
        "final_model": len(FAMILIES),
        "search_candidate": candidates * (metadata["outer_folds"] + 1),
    }
    require(dict(phases) == expected, f"Incomplete tracking lineage: {dict(phases)} != {expected}")
    parent = by_id[reference["parent_run_id"]]
    require(
        parent.data.tags["phase"] == "training_session",
        "Tracking validation failed: parent.data.tags['phase'] == 'training_session'",
    )
    for key, value in metadata["test_metrics"].items():
        np.testing.assert_allclose(
            parent.data.metrics[f"selected_test_{key}"], value, rtol=0, atol=0
        )
    for row in cv.summary.to_dict("records"):
        for metric in METRICS:
            for statistic in ["mean", "std"]:
                require_close(
                    parent.data.metrics.get(f"{row['family']}_outer_{metric}_{statistic}", np.nan),
                    row[f"{metric}_{statistic}"],
                    "Tracked CV comparison differs from prediction-derived scores",
                )
    rows = []
    observed_outer = set()
    for run in runs:
        require(run.info.status == "FINISHED", f"Run {run.info.run_id} is {run.info.status}")
    reconcile_tracked_searches(client, runs, metadata, searches, artifact_json)
    for run in runs:
        run_id, phase = run.info.run_id, run.data.tags["phase"]
        require(run.data.params, f"Missing parameters for {run_id}")
        require(run.data.metrics, f"Missing metrics for {run_id}")
        repository = get_artifact_repository(
            run.info.artifact_uri, tracking_uri=client.tracking_uri
        )
        paths = artifact_paths(client, run_id, repository=repository)
        require(paths, f"No artifacts for {run_id}")
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if phase != "training_session":
            require(parent_id in by_id, "Tracking validation failed: parent_id in by_id")
            require(
                "plots/metric_scores.svg" in paths,
                "Tracking validation failed: 'plots/metric_scores.svg' in paths",
            )
        if phase == "search_candidate":
            search_parent = by_id[parent_id]
            require(
                search_parent.data.tags["phase"] in ["outer_fold", "final_model"],
                "Tracking validation failed: search_parent.data.tags['phase'] in ['outer_fold', 'final_model']",
            )
            require(
                run.data.tags["family"] == search_parent.data.tags["family"],
                "Tracking validation failed: run.data.tags['family'] == search_parent.data.tags['family']",
            )
            require(
                "cross_validation.json" in paths,
                "Tracking validation failed: 'cross_validation.json' in paths",
            )
        elif phase == "outer_fold":
            require(
                parent_id == parent.info.run_id,
                "Tracking validation failed: parent_id == parent.info.run_id",
            )
            require(
                "fold_definition.json" in paths,
                "Tracking validation failed: 'fold_definition.json' in paths",
            )
            fold_token = run.data.tags.get("outer_fold")
            require(
                fold_token in {str(i) for i in range(1, metadata["outer_folds"] + 1)},
                "Invalid tracked CV fold identifier",
            )
            key = (run.data.tags.get("family"), int(fold_token))
            require(
                key in cv_folds.index and key not in observed_outer,
                "Duplicate or unexpected tracked CV fold",
            )
            observed_outer.add(key)
            expected_fold = cv_folds.loc[key]
            for metric in expected_fold.index.difference(["fit_rows", "validation_rows"]):
                require_close(
                    run.data.metrics.get(f"outer_{metric}", np.nan),
                    expected_fold[metric],
                    "Tracked CV fold metrics differ from prediction-derived scores",
                )
            for parameter in ["fit_rows", "validation_rows"]:
                require_close(
                    run.data.params.get(parameter, np.nan),
                    expected_fold[parameter],
                    "Tracked CV fold row counts differ from source split",
                )
            require(
                artifact_json(client, run, "fold_definition.json") == cv.definitions[key[1] - 1],
                "Tracked CV fold definition differs from source split",
            )
        elif phase == "final_model":
            require(
                parent_id == parent.info.run_id,
                "Tracking validation failed: parent_id == parent.info.run_id",
            )
            require(
                run.data.tags["model_roundtrip_verified"] == "true",
                "Tracking validation failed: run.data.tags['model_roundtrip_verified'] == 'true'",
            )
            family = run.data.tags["family"]
            require(
                f"pipeline/{family}.joblib" in paths,
                "Tracking validation failed: f'pipeline/{family}.joblib' in paths",
            )
            require(
                reference["final_model_run_ids"][family] == run_id,
                "Tracking validation failed: reference['final_model_run_ids'][family] == run_id",
            )
            logged_model = client.get_logged_model(reference["model_uris"][family].split("/")[-1])
            require(
                logged_model.source_run_id == run_id,
                "Tracking validation failed: logged_model.source_run_id == run_id",
            )
            require(
                logged_model.status.name == "READY",
                "Tracking validation failed: logged_model.status.name == 'READY'",
            )
        rows.append(
            {
                "run_id": run_id,
                "parent_run_id": parent_id,
                "name": run.data.tags["mlflow.runName"],
                "phase": phase,
                "family": run.data.tags["family"],
                "outer_fold": run.data.tags.get("outer_fold"),
                "search_phase": run.data.tags.get("search_phase"),
                "status": run.info.status,
                "parameters": json.dumps(run.data.params, sort_keys=True),
                "metrics": json.dumps(run.data.metrics, sort_keys=True),
                "artifact_paths": json.dumps(sorted(paths)),
            }
        )
    require(observed_outer == set(cv_folds.index), "Incomplete tracked CV fold coverage")
    parent_paths = artifact_paths(client, parent.info.run_id)
    require(
        "reports/figures/01_nested_cv_comparison.png" in parent_paths,
        "Tracking validation failed: 'reports/figures/01_nested_cv_comparison.png' in parent_paths",
    )
    require(
        "reports/figures/02_selected_test_curves.png" in parent_paths,
        "Tracking validation failed: 'reports/figures/02_selected_test_curves.png' in parent_paths",
    )
    require(
        "reports/figures/03_selected_test_confusion.png" in parent_paths,
        "Tracking validation failed: 'reports/figures/03_selected_test_confusion.png' in parent_paths",
    )
    require(
        "metadata.json" in parent_paths,
        "Tracking validation failed: 'metadata.json' in parent_paths",
    )
    require(
        "environment/requirements-lock.txt" in parent_paths,
        "Tracking validation failed: 'environment/requirements-lock.txt' in parent_paths",
    )
    require(
        "data/split_manifest.json" in parent_paths,
        "Tracking validation failed: 'data/split_manifest.json' in parent_paths",
    )
    report = root / metadata["report_directory"]
    pd.DataFrame(rows).sort_values(["phase", "family", "name"]).to_csv(
        report / "tracking_runs.csv", index=False
    )
    result = {
        "training_session": metadata["run_id"],
        "parent_run_id": parent.info.run_id,
        "experiment_name": reference["experiment_name"],
        "run_count": len(runs),
        "phase_counts": dict(phases),
        "all_finished": True,
        "all_runs_have_parameters_metrics_artifacts_plots": True,
        "candidate_fold_histories_verified": True,
        "tuning_tables_parameters_histories_and_artifacts_reconciled": True,
        "outer_cv_predictions_and_reports_reconciled": True,
        "outer_cv_tracking_metrics_reconciled": True,
        "parent_test_metrics_match_report": True,
        "both_final_models_logged_and_roundtrip_verified": True,
    }
    (report / "tracking_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    print(json.dumps(validate_tracking(), indent=2))


if __name__ == "__main__":
    main()
