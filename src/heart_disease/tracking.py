"""Local MLflow lineage for training sessions, folds, searches and models."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from html import escape
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.models import infer_signature

from .config import ROOT, SEED
from .modeling import METRICS

EXPERIMENT = "heart-disease-classification"
MODEL_PACKAGES = ["scikit-learn", "numpy", "pandas", "joblib", "cloudpickle", "skops"]
# OneHotEncoder's documented category arrays retain NumPy dtype objects.
# The forest's fitted trees are produced here by sklearn, not accepted as user input.
# Keep the skops allowlist narrow and explicit for these locally fitted pipelines.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def local_uri(root=ROOT):
    return "sqlite:///" + (Path(root).resolve() / "tracking/mlflow.db").as_posix()


def score_plot(metrics, title):
    """A small vector plot stored with every run, including search candidates."""
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="660" height="260" viewBox="0 0 660 260">',
        '<rect width="660" height="260" fill="white"/>',
        f'<text x="20" y="30" font-size="17">{escape(title)}</text>',
        '<text x="170" y="55" font-size="12">Score (bar range 0 to 1)</text>',
    ]
    for index, (name, value) in enumerate(metrics.items()):
        y = 75 + index * 40
        elements.extend(
            [
                f'<text x="20" y="{y + 17}" font-size="13">{escape(name)}</text>',
                f'<rect x="170" y="{y}" width="400" height="23" fill="#e9eef0"/>',
                f'<rect x="170" y="{y}" width="{400 * float(value):.2f}" '
                'height="23" fill="#247a89"/>',
                f'<text x="580" y="{y + 17}" font-size="13">{value:.4f}</text>',
            ]
        )
    return "\n".join([*elements, "</svg>"])


class ExperimentTracker:
    def __init__(self, session, root=ROOT):
        self.root = Path(root).resolve()
        (self.root / "tracking/artifacts").mkdir(parents=True, exist_ok=True)
        self.uri = local_uri(self.root)
        self.client = MlflowClient(tracking_uri=self.uri)
        experiment = self.client.get_experiment_by_name(EXPERIMENT)
        self.experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment(
                EXPERIMENT, artifact_location=(self.root / "tracking/artifacts").as_uri()
            )
        )
        self.session = session
        self.parent_id = None
        self.final_runs = {}
        self.model_uris = {}

    @contextmanager
    def run(self, name, phase, family="all", parent=None, fold=None):
        tags = {
            "mlflow.runName": name,
            "training_session": self.session,
            "phase": phase,
            "family": family,
        }
        if parent:
            tags["mlflow.parentRunId"] = parent
        if fold is not None:
            tags["outer_fold"] = str(fold)
        run_id = self.client.create_run(self.experiment_id, tags=tags).info.run_id
        try:
            yield run_id
        except BaseException as error:
            self.client.set_tag(run_id, "failure_type", type(error).__name__)
            self.client.set_terminated(run_id, "FAILED")
            raise
        else:
            self.client.set_terminated(run_id, "FINISHED")

    @contextmanager
    def session_run(self):
        with self.run(f"training-{self.session}", "training_session") as run_id:
            self.parent_id = run_id
            self.parameters(run_id, {"seed": SEED, "training_session": self.session})
            yield self

    def parameters(self, run_id, parameters):
        values = []
        for name, value in parameters.items():
            encoded = (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            )
            values.append(Param(name, str(encoded)))
        self.client.log_batch(run_id, params=values)

    def scores(self, run_id, metrics, prefix="", step=0):
        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            run_id,
            metrics=[
                Metric(prefix + name, float(value), timestamp, step)
                for name, value in metrics.items()
            ],
        )

    def plot(self, run_id, metrics, title):
        self.client.log_text(run_id, score_plot(metrics, title), "plots/metric_scores.svg")

    def search_candidates(self, search, family, phase, parent, fold=None):
        results = search.cv_results_
        for index, parameters in enumerate(results["params"]):
            with self.run(
                f"{family}-{phase}-candidate-{index + 1}",
                "search_candidate",
                family,
                parent,
                fold,
            ) as run_id:
                self.client.set_tag(run_id, "search_phase", phase)
                self.client.set_tag(run_id, "best_candidate", str(index == search.best_index_))
                fixed = search.estimator.named_steps["classifier"].get_params(deep=False)
                fixed.update(
                    {name.removeprefix("classifier__"): v for name, v in parameters.items()}
                )
                self.parameters(
                    run_id,
                    {**fixed, "cv_folds": search.n_splits_, "refit_metric": "roc_auc"},
                )
                means, points = {}, []
                timestamp = int(time.time() * 1000)
                evidence = {"parameters": parameters, "fold_scores": {}, "summary": {}}
                for metric in METRICS:
                    mean = float(results[f"mean_test_{metric}"][index])
                    std = float(results[f"std_test_{metric}"][index])
                    means[metric] = mean
                    points.extend(
                        [
                            Metric(f"cv_{metric}_mean", mean, timestamp, 0),
                            Metric(f"cv_{metric}_std", std, timestamp, 0),
                        ]
                    )
                    evidence["summary"][metric] = {"mean": mean, "std": std}
                    evidence["fold_scores"][metric] = []
                    for step in range(search.n_splits_):
                        value = float(results[f"split{step}_test_{metric}"][index])
                        points.append(Metric(f"cv_{metric}", value, timestamp, step + 1))
                        evidence["fold_scores"][metric].append(value)
                self.client.log_batch(run_id, metrics=points)
                self.client.log_dict(run_id, evidence, "cross_validation.json")
                self.plot(run_id, means, "Candidate cross-validation means")

    def outer_fold(self, family, fold, model, metrics, definition, search, table, fit_rows):
        with self.run(
            f"{family}-outer-{fold}", "outer_fold", family, self.parent_id, fold
        ) as run_id:
            self.parameters(
                run_id,
                {
                    **model.named_steps["classifier"].get_params(deep=False),
                    "seed": SEED,
                    "fit_rows": fit_rows,
                    "validation_rows": len(definition["validation_source_positions"]),
                    "numeric_scaling": family == "logistic_regression",
                    "numeric_imputation": "median",
                    "categorical_imputation": "most_frequent",
                    "categorical_encoding": "one_hot_documented_levels",
                },
            )
            self.scores(run_id, metrics, "outer_")
            self.client.log_dict(run_id, definition, "fold_definition.json")
            self.plot(run_id, {key: metrics[key] for key in METRICS}, "Outer validation scores")
            if search is not None:
                self.client.log_artifact(run_id, str(table), "search")
                self.search_candidates(search, family, "inner_cv", run_id, fold)

    def final_model(self, family, search, artifact, table, features):
        with self.run(f"{family}-final", "final_model", family, self.parent_id) as run_id:
            self.final_runs[family] = run_id
            self.parameters(
                run_id,
                {
                    **search.best_estimator_.named_steps["classifier"].get_params(deep=False),
                    "training_rows": len(features),
                    "cv_folds": search.n_splits_,
                    "numeric_scaling": family == "logistic_regression",
                    "numeric_imputation": "median",
                    "categorical_imputation": "most_frequent",
                    "categorical_encoding": "one_hot_documented_levels",
                    "refit_metric": "roc_auc",
                },
            )
            metrics = {
                key: float(search.cv_results_[f"mean_test_{key}"][search.best_index_])
                for key in METRICS
            }
            self.scores(run_id, metrics, "final_cv_")
            self.plot(run_id, metrics, "Final tuning CV means (selection scores)")
            self.client.log_artifact(run_id, str(table), "search")
            self.client.log_artifact(run_id, str(artifact), "pipeline")
            self.search_candidates(search, family, "final_cv", run_id)
            # Explicit client-created runs keep nesting independent of fluent global state.
            previous_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(self.uri)
            try:
                with mlflow.start_run(run_id=run_id):
                    example = features.head(3)
                    info = mlflow.sklearn.log_model(
                        search.best_estimator_,
                        name="pipeline",
                        signature=infer_signature(
                            features, search.best_estimator_.predict(features)
                        ),
                        input_example=example,
                        skops_trusted_types=SKOPS_TRUSTED_TYPES,
                        pip_requirements=[
                            f"{name}=={importlib.metadata.version(name)}" for name in MODEL_PACKAGES
                        ],
                    )
                    restored = mlflow.sklearn.load_model(info.model_uri)
                    np.testing.assert_array_equal(
                        restored.predict(features), search.best_estimator_.predict(features)
                    )
                    np.testing.assert_allclose(
                        restored.predict_proba(features),
                        search.best_estimator_.predict_proba(features),
                        rtol=0,
                        atol=0,
                    )
                    self.model_uris[family] = info.model_uri
                    self.client.set_tag(run_id, "model_roundtrip_verified", "true")
            finally:
                mlflow.set_tracking_uri(previous_uri)

    def reference(self):
        return {
            "tool": "MLflow",
            "version": importlib.metadata.version("mlflow"),
            "experiment_name": EXPERIMENT,
            "experiment_id": self.experiment_id,
            "parent_run_id": self.parent_id,
            "backend_relative_path": "tracking/mlflow.db",
            "artifacts_relative_path": "tracking/artifacts",
            "final_model_run_ids": self.final_runs,
            "model_uris": self.model_uris,
        }

    def complete(self, metadata, summary, reports, models):
        parent = self.parent_id
        self.parameters(
            parent,
            {
                key: metadata[key]
                for key in [
                    "seed",
                    "train_rows",
                    "test_rows",
                    "outer_folds",
                    "inner_folds",
                    "selected_family",
                    "selected_parameters",
                    "source_sha256",
                    "split_manifest_sha256",
                    "training_csv_sha256",
                    "test_csv_sha256",
                    "selection_rule",
                    "encoded_feature_count",
                ]
            },
        )
        for _, row in summary.iterrows():
            for metric in METRICS:
                self.scores(
                    parent,
                    {
                        f"{row['family']}_outer_{metric}_{stat}": row[f"{metric}_{stat}"]
                        for stat in ["mean", "std"]
                    },
                )
        self.scores(parent, metadata["test_metrics"], "selected_test_")
        for family, run_id in self.final_runs.items():
            self.client.set_tag(run_id, "selected", str(family == metadata["selected_family"]))
        self.client.log_dict(parent, metadata, "metadata.json")
        self.client.log_artifacts(parent, str(reports), "reports")
        self.client.log_artifacts(parent, str(models), "models")
        for source in sorted((self.root / "src/heart_disease").glob("*.py")):
            self.client.log_artifact(parent, str(source), "source")
        for name in [
            "pyproject.toml",
            "requirements.txt",
            "requirements-train.txt",
            "requirements-lock.txt",
        ]:
            self.client.log_artifact(parent, str(self.root / name), "environment")
        self.client.log_artifact(parent, str(self.root / "data/raw/source.json"), "data")
        self.client.log_artifact(
            parent, str(self.root / "data/processed/split_manifest.json"), "data"
        )


def main():
    parser = argparse.ArgumentParser(description="Open the local MLflow tracking UI.")
    parser.add_argument("--port", type=int, default=5000)
    arguments = parser.parse_args()
    if not (ROOT / "tracking/mlflow.db").exists():
        parser.error("Run heart_disease.train first to create experiment tracking records.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            local_uri(),
            "--host",
            "127.0.0.1",
            "--port",
            str(arguments.port),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
