"""Write model comparison figures and a report from completed experiments."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .modeling import METRICS
from .plotting import save_figure

DISPLAY_FAMILIES = {
    "dummy_baseline": "Majority baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def model_figures(summary, target, prediction, probability, selected_family, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = [DISPLAY_FAMILIES[x] for x in summary["family"]]
    positions = np.arange(len(names))
    for ax, metric in zip(axes.flat, METRICS):
        ax.bar(
            positions,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            capsize=5,
            color=["#A0ABB3", "#247A89", "#C55B40"],
        )
        ax.set(
            xticks=positions,
            xticklabels=names,
            ylim=(0, 1.10),
            ylabel=metric.replace("_", " ").upper(),
        )
        ax.tick_params(axis="x", labelsize=9)
    save_figure(
        fig,
        directory,
        "01_nested_cv_comparison.png",
        "How do the tuned models compare?",
        "242 development records; 5 outer folds, 3 inner folds. Bars: mean; whiskers: fold SD, not confidence intervals.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(
        target,
        probability,
        name=DISPLAY_FAMILIES[selected_family],
        ax=axes[0],
        plot_chance_level=True,
    )
    PrecisionRecallDisplay.from_predictions(
        target,
        probability,
        name=DISPLAY_FAMILIES[selected_family],
        ax=axes[1],
        plot_chance_level=True,
    )
    save_figure(
        fig,
        directory,
        "02_selected_test_curves.png",
        "Selected model: held-out discrimination",
        "Reserved test set (n=61); model chosen before test evaluation. Probabilities are not externally calibrated.",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        target,
        prediction,
        labels=[0, 1],
        display_labels=["Disease absent", "Disease present"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    save_figure(
        fig,
        directory,
        "03_selected_test_confusion.png",
        "Selected model: correct predictions and errors",
        "Reserved test set (n=61); classifier's default 0.5 probability boundary; rows=actual, columns=predicted.",
    )


def write_model_report(summary, metadata, directory):
    selected = metadata["selected_family"]
    test = metadata["test_metrics"]
    text = """# Feature engineering and model development

## Feature representation

All 13 original predictors are retained. `num` and `target` are excluded from
model inputs. Five numerical measurements use training-fitted median imputation.
Eight coded predictors use training-fitted mode imputation and one-hot encoding
with UCI's documented categories. This produces 28 transformed features.
Logistic Regression also standardizes numerical measurements; Random Forest
uses the imputed numerical values without scaling. Regularized Logistic Regression
can use all category indicators, while the forest captures nonlinear interactions.
No ad hoc clinical ratios, target-derived predictors, outlier deletion, feature
selection, or oversampling are introduced. The small source cohort and EDA do not
justify these additional choices for an initial comparison.

## Evaluation design

The fixed development partition contains 242 rows; 61 source rows are reserved
for final evaluation. Preprocessed EDA previews are never used for model fitting.
A shared shuffled, stratified 5-fold outer split compares model families.
Within each outer training subset, a stratified 3-fold GridSearchCV tunes settings.
Every fit includes a fresh preprocessing Pipeline. Accuracy, precision, recall,
and ROC-AUC are recorded for all search candidates and outer validation folds.
The majority-class baseline uses the same outer validation partitions.

The predeclared selection rule is maximum mean outer ROC-AUC, with exact ties
favoring Logistic Regression. ROC-AUC evaluates probability ranking without
choosing a threshold. Precision and recall describe the default classifier boundary.
Nested evaluation separates parameter tuning from outer-fold scoring; selecting
the best family can still favor a noisy estimate. Fold SD is descriptive spread,
not a confidence interval or proof of a statistically significant difference.
Method reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html.

## Search spaces

- Logistic Regression: C={0.1, 1, 10}; class_weight={None, balanced}; six candidates.
  Fixed: lbfgs solver, max_iter=3000, L2 regularization via default l1_ratio=0.
- Random Forest: max_depth={3, None}; min_samples_leaf={1, 4};
  class_weight={None, balanced}; eight candidates. Fixed: 200 trees,
  max_features=sqrt, seed=42, single worker per forest.
- Search refits on ROC-AUC and raises on fit errors rather than ignoring failures.

## Nested cross-validation results

Each value is outer-fold mean +/- sample standard deviation:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
"""
    for _, row in summary.iterrows():
        values = [f"{row[f'{m}_mean']:.4f} +/- {row[f'{m}_std']:.4f}" for m in METRICS]
        text += f"| {DISPLAY_FAMILIES[row['family']]} | " + " | ".join(values) + " |\n"
    text += f"""
Selected family: **{DISPLAY_FAMILIES[selected]}** by the outer-fold selection rule.
Each family is then tuned with stratified 5-fold CV on the complete development
partition and saved separately. These final tuning scores are selection scores,
not independent generalization estimates. Only the already selected family is
evaluated on the reserved test labels; the losing family is not test-ranked.

Final selected parameters:

```json
{json.dumps(metadata["selected_parameters"], indent=2)}
```

## Reserved test evaluation

| Metric | Result |
|---|---:|
| Accuracy | {test["accuracy"]:.4f} |
| Precision | {test["precision"]:.4f} |
| Recall | {test["recall"]:.4f} |
| ROC-AUC | {test["roc_auc"]:.4f} |
| Brier score (lower is better) | {test["brier_score"]:.4f} |

Counts: TN={test["true_negative"]}, FP={test["false_positive"]},
FN={test["false_negative"]}, TP={test["true_positive"]}.
With only 61 test records, these estimates have substantial sampling uncertainty.
The same test set must not be used to revise features, search spaces, or thresholds.
Repeated reruns reproduce this evaluation rather than supply new independent tests.

## Artifacts and reproducibility

Run ID: `{metadata["run_id"]}`. The run records source/split checksums, code hashes,
package versions, seeds, fold definitions, every candidate's CV metrics, per-fold
selected parameters, out-of-fold predictions, final search results, and test
predictions. Trained family pipelines are under `models/runs/{metadata["run_id"]}/`.
The selected release's `model.joblib` is the full preprocessing-plus-classifier
pipeline; metadata and a feature schema accompany it. The default inference loader
resolves `reports/modeling/latest.json`. Candidate model, tracking and bundle
validation must pass before that pointer is switched to a complete versioned
release under `models/releases/`. Save/load probability equality
was verified before publishing this artifact. No refit on the reserved test rows
is performed. `docs/sample_request.json` exercises prediction on a development row.

## Interpretation and next stage

These are reproducible academic baseline models for a small historical dataset.
Outputs indicate modeled recorded disease presence, not future disease risk.
Probability estimates have not undergone a separate calibration study; the API's
eventual confidence value must be described as a model estimate. No clinical
validation or population-wide claims follow from these results.
MLflow records this training session under experiment
`{metadata["experiment_tracking"]["experiment_name"]}`, parent run
`{metadata["experiment_tracking"]["parent_run_id"]}`. Nested child runs retain all
hyperparameter candidates, per-fold CV metrics, fold definitions, search tables,
metric plots and final model artifacts. Both final pipelines have MLflow signatures,
input examples and environments; native MLflow save/load prediction and probability
equality is verified. `tracking_runs.csv` and `tracking_validation.json` export
tracking evidence alongside this report. Open the UI using
`python -m heart_disease.tracking` and see `docs/experiment_tracking.md`.
The serving API,
Docker image, Kubernetes deployment, and monitoring are still pending.

## Rerun

Install `requirements-train.txt` or `requirements-dev.txt`, then run
`python -m heart_disease.train --jobs 2`. If data artifacts are absent, run
acquisition and EDA first. Use `python -m heart_disease.predict --input
docs/sample_request.json` for an inference smoke check. Every training run writes
to a new directory, retaining previous experiment evidence.
"""
    (directory / "model_development.md").write_text(text, encoding="utf-8")
