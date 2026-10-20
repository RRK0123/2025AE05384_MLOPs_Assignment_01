# Feature engineering and model development

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
| Majority baseline | 0.5413 +/- 0.0072 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5000 +/- 0.0000 |
| Logistic Regression | 0.8434 +/- 0.0524 | 0.8700 +/- 0.1018 | 0.7945 +/- 0.1111 | 0.9101 +/- 0.0385 |
| Random Forest | 0.7938 +/- 0.0545 | 0.8057 +/- 0.0523 | 0.7316 +/- 0.1313 | 0.9037 +/- 0.0387 |

Selected family: **Logistic Regression** by the outer-fold selection rule.
Each family is then tuned with stratified 5-fold CV on the complete development
partition and saved separately. These final tuning scores are selection scores,
not independent generalization estimates. Only the already selected family is
evaluated on the reserved test labels; the losing family is not test-ranked.

Final selected parameters:

```json
{
  "classifier__C": 0.1,
  "classifier__class_weight": "balanced"
}
```

## Reserved test evaluation

| Metric | Result |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| ROC-AUC | 0.9654 |
| Brier score (lower is better) | 0.0893 |

Counts: TN=28, FP=5,
FN=2, TP=26.
With only 61 test records, these estimates have substantial sampling uncertainty.
The same test set must not be used to revise features, search spaces, or thresholds.
Repeated reruns reproduce this evaluation rather than supply new independent tests.

## Artifacts and reproducibility

Run ID: `20261017T071103958837Z`. The run records source/split checksums, code hashes,
package versions, seeds, fold definitions, every candidate's CV metrics, per-fold
selected parameters, out-of-fold predictions, final search results, and test
predictions. Trained family pipelines are under `models/runs/20261017T071103958837Z/`.
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
`heart-disease-classification`, parent run
`ef91491617354673bc0a54a0005ec791`. Nested child runs retain all
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
