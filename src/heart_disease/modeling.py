"""Model definitions, parameter searches, and evaluation rules."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import SEED
from .preprocessing import build_preprocessor

METRICS = ["accuracy", "precision", "recall", "roc_auc"]
FAMILIES = ["logistic_regression", "random_forest"]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",
}
GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "classifier__max_depth": [3, None],
        "classifier__min_samples_leaf": [1, 4],
        "classifier__class_weight": [None, "balanced"],
    },
}


def build_model(family: str) -> Pipeline:
    if family == "logistic_regression":
        classifier = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)
    elif family == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", n_jobs=1, random_state=SEED
        )
    elif family == "dummy_baseline":
        classifier = DummyClassifier(strategy="most_frequent", random_state=SEED)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=family == "logistic_regression")),
            ("classifier", classifier),
        ]
    )


def build_search(family: str, folds: int, seed: int, jobs: int) -> GridSearchCV:
    return GridSearchCV(
        build_model(family),
        GRIDS[family],
        scoring=SCORING,
        refit="roc_auc",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=jobs,
        pre_dispatch=jobs,
        error_score="raise",
        return_train_score=True,
    )


def disease_probabilities(model, features) -> np.ndarray:
    classes = model.classes_.tolist()
    if classes != [0, 1]:
        raise ValueError(f"Expected fitted binary classes [0, 1]; received {classes}.")
    return model.predict_proba(features)[:, classes.index(1)]


def evaluate_predictions(target, prediction, probability) -> dict:
    """Binary class 1 is disease present; probabilities measure that class."""
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def summarize_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for family in ["dummy_baseline", *FAMILIES]:
        folds = fold_results.loc[fold_results["family"] == family]
        row = {"family": family, "folds": len(folds)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(folds[metric].mean())
            row[f"{metric}_std"] = float(folds[metric].std(ddof=1))
        records.append(row)
    return pd.DataFrame(records)


def select_family(summary: pd.DataFrame) -> str:
    """Choose by outer-fold ROC-AUC only; exact ties favor Logistic Regression."""
    candidates = summary.loc[summary["family"].isin(FAMILIES)].copy()
    if set(candidates["family"]) != set(FAMILIES) or len(candidates) != len(FAMILIES):
        raise ValueError("Both classifier families must be evaluated exactly once.")
    if not np.isfinite(candidates["roc_auc_mean"]).all():
        raise ValueError("Model selection requires finite ROC-AUC scores.")
    candidates["tie_priority"] = candidates["family"].map(
        {name: i for i, name in enumerate(FAMILIES)}
    )
    return candidates.sort_values(["roc_auc_mean", "tie_priority"], ascending=[False, True]).iloc[
        0
    ]["family"]
