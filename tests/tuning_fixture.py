"""Deterministic, fully reconciled tuning tables without expensive model fitting."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from heart_disease.modeling import FAMILIES, GRIDS, METRICS, build_model


def write_tuning_fixture(root, metadata):
    tables = root / metadata["report_directory"] / "tables"
    outer_path = tables / "outer_fold_metrics.csv"
    outer = pd.read_csv(outer_path)
    searches, final = {}, {}
    for family in FAMILIES:
        parameters = list(ParameterGrid(GRIDS[family]))
        for fold in [*range(1, metadata["outer_folds"] + 1), None]:
            folds = metadata["inner_folds"] if fold is not None else metadata["outer_folds"]
            columns = {"params": parameters}
            for parameter in GRIDS[family]:
                columns[f"param_{parameter}"] = [candidate[parameter] for candidate in parameters]
            for partition in ["test", "train"]:
                for number, metric in enumerate(METRICS):
                    scores = np.array(
                        [
                            [
                                0.70
                                + number * 0.01
                                + step * 0.001
                                - index * 0.005
                                + (0.02 if partition == "train" else 0)
                                for step in range(folds)
                            ]
                            for index in range(len(parameters))
                        ]
                    )
                    for step in range(folds):
                        columns[f"split{step}_{partition}_{metric}"] = scores[:, step]
                    columns[f"mean_{partition}_{metric}"] = scores.mean(axis=1)
                    columns[f"std_{partition}_{metric}"] = scores.std(axis=1, ddof=0)
                    if partition == "test":
                        columns[f"rank_test_{metric}"] = np.arange(1, len(parameters) + 1)
            table = pd.DataFrame(columns)
            label = f"outer{fold}" if fold is not None else "final"
            saved = table.copy()
            saved["params"] = saved.params.map(lambda value: json.dumps(value, sort_keys=True))
            saved.to_csv(tables / f"{family}_{label}_search.csv", index=False)
            search = SimpleNamespace(
                estimator=build_model(family), cv_results_=columns, best_index_=0, n_splits_=folds
            )
            searches[family, fold] = search
            score = float(table.iloc[0]["mean_test_roc_auc"])
            if fold is None:
                final[family] = {
                    "parameters": parameters[0],
                    "tuning_roc_auc": score,
                    "candidate_count": len(parameters),
                }
            else:
                mask = (outer.family == family) & (outer.outer_fold == fold)
                outer.loc[mask, "best_parameters"] = json.dumps(parameters[0], sort_keys=True)
                outer.loc[mask, "inner_best_roc_auc"] = score
    outer.to_csv(outer_path, index=False)
    metadata.update(
        parameter_grids=GRIDS,
        final_searches=final,
        selected_parameters=final[metadata["selected_family"]]["parameters"],
    )
    return searches
