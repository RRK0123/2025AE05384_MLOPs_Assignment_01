"""Exercise persistence, candidate lineage and failure status in an isolated store."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from heart_disease.modeling import build_search
from heart_disease.schema import CATEGORIES, FEATURES, NUMERIC
from heart_disease.tracking import ExperimentTracker
from heart_disease.validate_tracking import artifact_paths


def test_failed_session_and_child_are_recorded_without_active_run(tmp_path):
    tracker = ExperimentTracker("failure-check", root=tmp_path)
    with pytest.raises(RuntimeError, match="deliberate"):
        with tracker.session_run():
            with tracker.run("broken-fit", "outer_fold", parent=tracker.parent_id) as child:
                raise RuntimeError("deliberate fit error")
    assert tracker.client.get_run(tracker.parent_id).info.status == "FAILED"
    assert tracker.client.get_run(child).info.status == "FAILED"
    assert tracker.client.get_run(child).data.tags["failure_type"] == "RuntimeError"


def test_search_candidates_preserve_params_scores_history_and_artifacts(tmp_path):
    features = pd.DataFrame(
        {c: [float(next(iter(levels)))] * 20 for c, levels in CATEGORIES.items()}
    )
    for column in NUMERIC:
        features[column] = np.arange(20, 40, dtype=float)
    target = pd.Series([0, 1] * 10)
    search = build_search("logistic_regression", folds=2, seed=42, jobs=1)
    search.fit(features[FEATURES], target)
    tracker = ExperimentTracker("candidate-check", root=tmp_path)
    with tracker.session_run():
        with tracker.run("search", "final_model", parent=tracker.parent_id) as search_run:
            tracker.search_candidates(search, "logistic_regression", "final_cv", search_run)
    runs = tracker.client.search_runs([tracker.experiment_id], "tags.phase = 'search_candidate'")
    assert len(runs) == 6
    assert sum(run.data.tags["best_candidate"] == "True" for run in runs) == 1
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.parentRunId"] == search_run
        assert float(run.data.params["C"]) in [0.1, 1, 10]
        history = tracker.client.get_metric_history(run.info.run_id, "cv_roc_auc")
        assert sorted(point.step for point in history) == [1, 2]
        np.testing.assert_allclose(
            np.mean([point.value for point in history]), run.data.metrics["cv_roc_auc_mean"]
        )
        artifact = tracker.client.download_artifacts(run.info.run_id, "cross_validation.json")
        assert (
            len(json.loads(Path(artifact).read_text(encoding="utf-8"))["fold_scores"]["recall"])
            == 2
        )
        plot = tracker.client.download_artifacts(run.info.run_id, "plots/metric_scores.svg")
        assert "Candidate cross-validation means" in Path(plot).read_text(encoding="utf-8")


@pytest.mark.parametrize("family", ["logistic_regression", "random_forest"])
def test_native_mlflow_model_retains_probabilities_and_pinned_serializer(tmp_path, family):
    features = pd.DataFrame(
        {c: [float(next(iter(levels)))] * 20 for c, levels in CATEGORIES.items()}
    )
    for column in NUMERIC:
        features[column] = np.arange(20, 40, dtype=float)
    features.loc[0, "ca"] = np.nan
    features = features[FEATURES]
    search = build_search(family, folds=2, seed=42, jobs=1)
    search.param_grid = {"classifier__class_weight": [None]}
    search.fit(features, pd.Series([0, 1] * 10))
    artifact, table = tmp_path / f"{family}.joblib", tmp_path / "search.csv"
    joblib.dump(search.best_estimator_, artifact)
    pd.DataFrame(search.cv_results_).to_csv(table, index=False)
    previous_uri = mlflow.get_tracking_uri()
    tracker = ExperimentTracker("native-model-check", root=tmp_path)
    with tracker.session_run():
        tracker.final_model(family, search, artifact, table, features)
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() == previous_uri
    assert f"pipeline/{family}.joblib" in artifact_paths(tracker.client, tracker.final_runs[family])
    try:
        mlflow.set_tracking_uri(tracker.uri)
        uri = tracker.model_uris[family]
        restored = mlflow.sklearn.load_model(uri)
        np.testing.assert_allclose(
            restored.predict_proba(features),
            search.best_estimator_.predict_proba(features),
            rtol=0,
            atol=0,
        )
        directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
        assert "skops==" in (directory / "requirements.txt").read_text()
        model_config = (directory / "MLmodel").read_text()
        assert "numpy.dtype" in model_config
        assert "signature:" in model_config
        assert "serialization_format: skops" in model_config
    finally:
        mlflow.set_tracking_uri(previous_uri)
