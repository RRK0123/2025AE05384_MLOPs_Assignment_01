"""Labeled HTTP evaluation and explicit, educational monitoring decisions."""

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from heart_disease.schema import CATEGORIES, FEATURES, NUMERIC
from scripts.smoke_api import request_json


def metrics(labels, predictions, probabilities):
    labels, predictions, probabilities = map(np.asarray, (labels, predictions, probabilities))
    if not (len(labels) == len(predictions) == len(probabilities)) or not len(labels):
        raise ValueError("Labels and predictions must have equal nonzero lengths")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected binary labels and predictions")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    both = len(np.unique(labels)) == 2
    return {
        "records": len(labels),
        "class_counts": {str(i): int(sum(labels == i)) for i in [0, 1]},
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities))
        if both
        else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def confidence_intervals(labels, predictions, probabilities, *, samples=500, seed=42):
    labels, predictions, probabilities = map(np.asarray, (labels, predictions, probabilities))
    groups = [np.flatnonzero(labels == value) for value in [0, 1]]
    if not all(len(group) for group in groups):
        return {"status": "insufficient_classes"}
    rng = np.random.default_rng(seed)
    names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc_average_precision",
        "brier_score",
    ]
    values = {name: [] for name in names}
    for _ in range(samples):
        positions = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        result = metrics(labels[positions], predictions[positions], probabilities[positions])
        for name in names:
            values[name].append(result[name])
    return {
        "method": "95% percentile stratified bootstrap; fixed observed class counts",
        "samples": samples,
        "seed": seed,
        "intervals": {
            name: np.quantile(scores, [0.025, 0.975]).tolist() for name, scores in values.items()
        },
    }


def drift(reference, observed, policy):
    signals = []
    for name in NUMERIC:
        a, b = reference[name].dropna(), observed[name].dropna()
        statistic = float(ks_2samp(a, b).statistic) if len(a) and len(b) else None
        signals.append(
            {
                "feature": name,
                "method": "two-sample KS effect size",
                "value": statistic,
                "threshold": policy["numeric_ks_threshold"],
                "alert": statistic is None or statistic > policy["numeric_ks_threshold"],
            }
        )
    for name in CATEGORIES:
        a = reference[name].fillna("missing").value_counts(normalize=True)
        b = observed[name].fillna("missing").value_counts(normalize=True)
        categories = a.index.union(b.index)
        distance = float(
            abs(a.reindex(categories, fill_value=0) - b.reindex(categories, fill_value=0)).sum() / 2
        )
        signals.append(
            {
                "feature": name,
                "method": "total variation distance",
                "value": distance,
                "threshold": policy["categorical_total_variation_threshold"],
                "alert": distance > policy["categorical_total_variation_threshold"],
            }
        )
    return {
        "drift_alert": any(item["alert"] for item in signals),
        "features": signals,
        "interpretation": "Distribution changes require investigation; they do not alone prove predictive degradation.",
    }


def monitoring_decision(reference, observed, policy, *, scenario, model_run_id):
    enough = observed["records"] >= policy["minimum_labeled_records"] and all(
        observed["class_counts"][str(i)] >= policy["minimum_per_class"] for i in [0, 1]
    )
    triggers = []
    if enough:
        for metric, setting in [("recall", "recall_drop"), ("roc_auc", "roc_auc_drop")]:
            if reference[metric] is not None and observed[metric] is not None:
                drop = reference[metric] - observed[metric]
                if drop >= policy[setting]:
                    triggers.append(
                        {
                            "metric": metric,
                            "baseline": reference[metric],
                            "observed": observed[metric],
                            "drop": drop,
                            "threshold": policy[setting],
                        }
                    )
    return {
        "scenario": scenario,
        "model_run_id": model_run_id,
        "decision": "request_retraining"
        if triggers
        else "healthy"
        if enough
        else "insufficient_evidence",
        "request_retraining": bool(triggers),
        "triggers": triggers,
        "sample_count": observed["records"],
        "class_counts": observed["class_counts"],
        "criteria": policy,
        "scope": "Single labeled replay window for educational demonstration. Reused rows are not independent production observations.",
        "automatic_retraining_performed": False,
    }


def payload_for(frame):
    return {"instances": json.loads(frame[FEATURES].to_json(orient="records"))}


def endpoint_predictions(url, frame, expected_run):
    predictions = []
    for start in range(0, len(frame), 256):
        batch = frame.iloc[start : start + 256]
        status, result, _ = request_json(url + "/predict", payload_for(batch))
        if status != 200 or result.get("model_run_id") != expected_run:
            raise ValueError(f"Endpoint prediction or model identity failed: HTTP {status}")
        if len(result["predictions"]) != len(batch):
            raise ValueError("Endpoint prediction count differs from input")
        predictions.extend(result["predictions"])
    return predictions


def degraded_replay(test, *, seed):
    """Destroy input/label association, preserving labels and valid field schemas."""
    rng = np.random.default_rng(seed)
    replay = test.copy()
    # Independent column permutation models corrupted patient measurements. No
    # target is changed, and this must never be presented as new clinical data.
    for name in FEATURES:
        replay[name] = rng.permutation(test[name].to_numpy())
    return replay


def load_check(url, example, *, requests=80, concurrency=4):
    def one(_):
        started = time.perf_counter()
        try:
            status, _, _ = request_json(url + "/predict", copy.deepcopy(example))
        except OSError:
            status = 0
        return {"status": status, "milliseconds": (time.perf_counter() - started) * 1000}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        observations = list(pool.map(one, range(requests)))
    duration = time.perf_counter() - started
    durations = [item["milliseconds"] for item in observations]
    return {
        "requests": requests,
        "concurrency": concurrency,
        "duration_seconds": duration,
        "throughput_requests_per_second": requests / duration,
        "latency_ms": {
            name: float(np.quantile(durations, quantile))
            for name, quantile in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]
        },
        "error_rate": sum(item["status"] != 200 for item in observations) / requests,
        "observations": observations,
    }


def statistical_data(workspace):
    workspace = Path(workspace)
    clean = pd.read_csv(workspace / "data/processed/heart_clean.csv")
    pointer = json.loads((workspace / "reports/modeling/latest.json").read_text())
    report = workspace / Path(pointer["report"]).parent
    return {
        "data": {
            "records": len(clean),
            "predictors": FEATURES,
            "input_feature_count": len(FEATURES),
            "missing_by_feature": clean[FEATURES].isna().sum().to_dict(),
            "class_counts": clean["target"].value_counts().sort_index().to_dict(),
        },
        "metadata": json.loads((report / "metrics.json").read_text()),
        "tables": {
            p.name: json.loads(pd.read_csv(p).to_json(orient="records"))
            for p in (report / "tables").glob("*.csv")
        },
        "tracking": json.loads((report / "tracking_validation.json").read_text()),
        "data_quality": json.loads((workspace / "reports/data_quality.json").read_text()),
        "preprocessing": json.loads((workspace / "reports/preprocessing_audit.json").read_text()),
        "eda_tables": {
            p.name: json.loads(pd.read_csv(p).to_json(orient="records"))
            for p in (workspace / "reports/tables").glob("*.csv")
        },
        "provenance": "Fresh acceptance workspace; full-source quality counts, training-only EDA, reserved-test evaluation.",
    }
