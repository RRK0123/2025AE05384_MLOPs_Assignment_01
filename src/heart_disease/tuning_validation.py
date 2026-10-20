"""Reconcile search tables, selected parameters and tracked tuning evidence."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .cv_validation import require_close
from .modeling import FAMILIES, GRIDS, METRICS, build_model
from .validate import require


@dataclass
class SearchEvidence:
    family: str
    fold: int | None
    folds: int
    table: pd.DataFrame
    parameters: list[dict]
    best_index: int

    @property
    def best_parameters(self):
        return self.parameters[self.best_index]

    @property
    def best_score(self):
        return float(self.table.iloc[self.best_index]["mean_test_roc_auc"])


def classifier_parameters(family, parameters):
    return build_model(family).set_params(**parameters).named_steps["classifier"].get_params()


def validate_fitted_parameters(pipeline, family, parameters):
    expected = build_model(family).set_params(**parameters).named_steps["classifier"]
    actual = pipeline.named_steps["classifier"]
    require(
        type(actual) is type(expected) and actual.get_params() == expected.get_params(),
        f"Fitted classifier parameters differ from tuning evidence: {family}",
    )


def require_logged_parameters(recorded, expected, description):
    # MLflow stores classifier settings as strings. Match their logged encoding.
    for name, value in expected.items():
        encoded = (
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        )
        require(
            recorded.get(name) == encoded,
            f"Tracked tuning parameters differ: {description}, {name}",
        )


def read_search(path, family, fold, folds):
    table = pd.read_csv(path, float_precision="round_trip")
    parameters = [json.loads(value) for value in table["params"]]
    grid = list(ParameterGrid(GRIDS[family]))
    require(parameters == grid, f"Tuning candidate grid/order differs: {family}, {fold}")
    require(
        {name for name in table if name.startswith("param_")}
        == {f"param_{name}" for name in GRIDS[family]},
        "Tuning parameter columns differ from the declared grid",
    )
    for index, candidate in enumerate(parameters):
        for name, value in candidate.items():
            observed = table.iloc[index][f"param_{name}"]
            require(
                pd.isna(observed) if value is None else observed == value,
                "Tuning parameter columns differ from candidate JSON",
            )
    for partition in ["test", "train"]:
        for metric in METRICS:
            columns = [f"split{i}_{partition}_{metric}" for i in range(folds)]
            require(
                {
                    name
                    for name in table
                    if name.startswith("split") and name.endswith(f"_{partition}_{metric}")
                }
                == set(columns),
                "Tuning split-score columns differ from the declared fold count",
            )
            values = table[columns].to_numpy(dtype=float)
            require(
                np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                "Invalid tuning fold scores",
            )
            require_close(
                table[f"mean_{partition}_{metric}"],
                values.mean(axis=1),
                "Tuning means differ from split scores",
            )
            require_close(
                table[f"std_{partition}_{metric}"],
                values.std(axis=1, ddof=0),
                "Tuning standard deviations differ from split scores",
            )
            if partition == "test":
                ranks = table[f"mean_test_{metric}"].rank(method="min", ascending=False)
                require_close(
                    table[f"rank_test_{metric}"], ranks, "Tuning ranks differ from scores"
                )
    return SearchEvidence(
        family,
        fold,
        folds,
        table,
        parameters,
        int(np.argmax(table["mean_test_roc_auc"].to_numpy())),
    )


def reconcile_tuning(root, metadata):
    root = Path(root)
    require(metadata["parameter_grids"] == GRIDS, "Tuning metadata grid differs from configuration")
    require(set(metadata["final_searches"]) == set(FAMILIES), "Unexpected final tuning families")
    tables = root / metadata["report_directory"] / "tables"
    outer = pd.read_csv(tables / "outer_fold_metrics.csv", float_precision="round_trip")
    searches = {}
    for family in FAMILIES:
        for fold in [*range(1, metadata["outer_folds"] + 1), None]:
            label = f"outer{fold}" if fold is not None else "final"
            search = read_search(
                tables / f"{family}_{label}_search.csv",
                family,
                fold,
                metadata["inner_folds"] if fold is not None else metadata["outer_folds"],
            )
            searches[family, fold] = search
            if fold is not None:
                row = outer.loc[(outer.family == family) & (outer.outer_fold == fold)]
                require(len(row) == 1, "Missing or duplicate tuning outer-fold selection")
                require(
                    json.loads(row.iloc[0]["best_parameters"]) == search.best_parameters,
                    "Tuning outer-fold best parameters differ from search",
                )
                require_close(
                    row.iloc[0]["inner_best_roc_auc"],
                    search.best_score,
                    "Tuning outer-fold best score differs from search",
                )
            else:
                final = metadata["final_searches"][family]
                require(
                    final["parameters"] == search.best_parameters,
                    "Tuning final best parameters differ from search",
                )
                require(
                    final["candidate_count"] == len(search.parameters),
                    "Tuning candidate count differs",
                )
                require_close(
                    final["tuning_roc_auc"],
                    search.best_score,
                    "Tuning final best score differs from search",
                )
    require(
        metadata["selected_parameters"]
        == searches[metadata["selected_family"], None].best_parameters,
        "Selected tuning parameters differ from the final search",
    )
    return searches


def reconcile_tracked_searches(client, runs, metadata, searches, artifact_json):
    """Bind every local candidate to its logged parameters, history and JSON artifact."""
    by_id = {run.info.run_id: run for run in runs}
    parent = by_id[metadata["experiment_tracking"]["parent_run_id"]]
    require(
        parent.data.params.get("selected_family") == metadata["selected_family"],
        "Tracked tuning selected family differs",
    )
    require(
        json.loads(parent.data.params.get("selected_parameters", "null"))
        == metadata["selected_parameters"],
        "Tracked tuning selected parameters differ",
    )
    observed = set()
    ordered = sorted(
        runs,
        key=lambda run: (
            run.data.tags["family"],
            run.data.tags["phase"] != "final_model",
            run.data.tags.get("outer_fold", ""),
        ),
    )
    for run in ordered:
        phase = run.data.tags["phase"]
        family = run.data.tags["family"]
        if phase not in ["outer_fold", "final_model"] or family not in FAMILIES:
            continue
        token = run.data.tags.get("outer_fold")
        require(
            phase != "outer_fold"
            or token in {str(i) for i in range(1, metadata["outer_folds"] + 1)},
            "Invalid tracked CV tuning outer fold",
        )
        key = family, int(token) if phase == "outer_fold" else None
        require(
            key in searches and key not in observed, "Duplicate or unexpected tracked tuning search"
        )
        observed.add(key)
        search = searches[key]
        require_logged_parameters(
            run.data.params,
            classifier_parameters(family, search.best_parameters),
            f"{family}, {search.fold}",
        )
        if phase == "final_model":
            require_logged_parameters(
                run.data.params,
                {
                    "cv_folds": search.folds,
                    "training_rows": metadata["train_rows"],
                    "refit_metric": "roc_auc",
                },
                "final search",
            )
            for metric in METRICS:
                require_close(
                    run.data.metrics.get(f"final_cv_{metric}", np.nan),
                    search.table.iloc[search.best_index][f"mean_test_{metric}"],
                    "Tracked tuning final score differs from search",
                )
        children = [
            child for child in runs if child.data.tags.get("mlflow.parentRunId") == run.info.run_id
        ]
        require(len(children) == len(search.parameters), "Tracked tuning candidate count differs")
        seen = set()
        for child in children:
            require(
                child.data.tags.get("phase") == "search_candidate"
                and child.data.tags.get("family") == family
                and child.data.tags.get("outer_fold") == token
                and child.data.tags.get("search_phase")
                == ("inner_cv" if search.fold is not None else "final_cv"),
                "Tracked tuning candidate identity differs",
            )
            evidence = artifact_json(client, child, "cross_validation.json")
            parameters = evidence["parameters"]
            require(
                parameters in search.parameters,
                "Tracked tuning candidate parameters are outside the grid",
            )
            index = search.parameters.index(parameters)
            require(index not in seen, "Duplicate tracked tuning candidate")
            seen.add(index)
            require_logged_parameters(
                child.data.params, classifier_parameters(family, parameters), "candidate"
            )
            require_logged_parameters(
                child.data.params,
                {"cv_folds": search.folds, "refit_metric": "roc_auc"},
                "candidate",
            )
            require(
                child.data.tags.get("best_candidate") == str(index == search.best_index),
                "Tracked tuning best-candidate tag differs from search",
            )
            row = search.table.iloc[index]
            for metric in METRICS:
                values = [float(row[f"split{i}_test_{metric}"]) for i in range(search.folds)]
                history = sorted(
                    client.get_metric_history(child.info.run_id, f"cv_{metric}"),
                    key=lambda point: point.step,
                )
                require(
                    [point.step for point in history] == list(range(1, search.folds + 1)),
                    "Tracked tuning fold history steps differ",
                )
                require_close(
                    [point.value for point in history],
                    values,
                    "Tracked tuning fold history differs from search",
                )
                require_close(
                    evidence["fold_scores"][metric],
                    values,
                    "Tracked tuning JSON fold scores differ from search",
                )
                for statistic in ["mean", "std"]:
                    expected = float(row[f"{statistic}_test_{metric}"])
                    require_close(
                        child.data.metrics.get(f"cv_{metric}_{statistic}", np.nan),
                        expected,
                        "Tracked tuning candidate summary differs from search",
                    )
                    require_close(
                        evidence["summary"][metric][statistic],
                        expected,
                        "Tracked tuning JSON summary differs from search",
                    )
        require(seen == set(range(len(search.parameters))), "Incomplete tracked tuning grid")
    require(observed == set(searches), "Incomplete tracked tuning search coverage")
