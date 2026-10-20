# Experiment tracking

Training automatically writes to MLflow 3.17.0 using a local SQLite database
(`tracking/mlflow.db`) and local artifact storage (`tracking/artifacts/`). No
tracking server or cloud account is needed to train. The backend and artifact
paths are derived from the project root, independently of the command's working
directory. The UI connects to the same database.

## Run and inspect

From the repository root, after dependency installation and EDA:

```powershell
.\.venv\Scripts\python.exe -m heart_disease.train --jobs 2
.\.venv\Scripts\python.exe -m heart_disease.validate_tracking
.\.venv\Scripts\python.exe -m heart_disease.tracking --port 5000
```

Open http://127.0.0.1:5000 and select **heart-disease-classification**.
The equivalent installed commands are `heart-train`, `heart-validate-tracking`,
and `heart-tracking-ui`. Keep the UI command running; Ctrl+C stops it. Linux/macOS
uses `.venv/bin/python` instead. Change `--port` if 5000 is occupied.

Expand a training session to see its folds and final models, then expand a fold
or final model to inspect its search candidates. Filter with
`tags.phase = 'final_model'` to compare the final tuning scores, or
`tags.phase = 'search_candidate'` for candidate runs. All descendants carry
`training_session`, `phase`, and `family` tags; outer descendants also identify
the outer fold. `search_phase` distinguishes inner tuning from final tuning.
Failed attempts are retained with failure status and type. Use the latest parent
ID in `metadata.json` in the directory returned by `resolve_model_directory()`,
or filter by its `training_session` tag,
to inspect the completed submission run separately from earlier attempts.

## What is recorded

Each successful invocation creates **102 runs**:

| Run level | Count | Recorded evidence |
|---|---:|---|
| Training session | 1 | Data hashes, seeds, selection rule, selected settings, outer comparison means/SDs, selected test metrics, code snapshot, environment lock, reports, plots and both joblib pipelines |
| Outer validation | 15 | Baseline and two families across five folds; fitted classifier settings, preprocessing choices, validation metrics/plot, row boundaries and search tables |
| Hyperparameter candidate | 84 | Six Logistic Regression and eight forest settings in each of five inner searches and one final search; classifier settings, fold count, metric means/SDs, individual fold scores, JSON evidence and metric plot |
| Final family model | 2 | Full-development tuning settings/scores, search table/plot, joblib artifact and native MLflow model with signature, input example and pinned environment |

CV metrics retain accuracy, precision, recall and ROC-AUC. Candidate fold scores
use MLflow metric history steps starting at 1. Candidate SD uses GridSearchCV's
population SD; outer comparison SD uses sample SD as documented in the model
report. Candidate/final tuning results are selection scores. Outer validation
scores measure the nested tuning procedure. Test metrics occur only on the
session parent for the already selected family, under `selected_test_*`.
Neither the losing family nor search candidates are evaluated on test labels.

Each run has parameters, metrics and artifacts, including a plot. The candidate
runs represent GridSearchCV settings; individual fold-fitted candidate models are
not retained by GridSearchCV. Both final pipelines are saved in MLflow's sklearn
flavor, and their labels and probabilities must match after reload exactly.
The current serving release is resolved by `resolve_model_directory()`, which
follows the atomic selection pointer in `reports/modeling/latest.json` to an
immutable directory under `models/releases/`. The retained `models/selected/`
directory is the legacy bootstrap snapshot and may differ from the current selection.

The resolved release's `metadata.json` records experiment/run IDs and `models:/...`
URIs. An example native model reload using the current selection is:

```python
import json
import mlflow
import mlflow.sklearn
from heart_disease.tracking import local_uri
from heart_disease.releases import resolve_model_directory

metadata = json.loads((resolve_model_directory() / "metadata.json").read_text())
mlflow.set_tracking_uri(local_uri())
uri = metadata["experiment_tracking"]["model_uris"][metadata["selected_family"]]
pipeline = mlflow.sklearn.load_model(uri)
# pipeline.predict(raw_feature_dataframe)
# pipeline.predict_proba(raw_feature_dataframe)
```

The signature describes the original 13 inputs. Numerical JSON input is converted
to floating-point values before prediction, as in `heart_disease.predict`.
Missing values are handled by the saved preprocessing pipeline.
MLflow uses its current sklearn default, skops serialization, with an explicit
allowlist containing `numpy.dtype` (retained by the documented category arrays)
and `sklearn.tree._tree.Tree` (created by our locally fitted forest). It does not
automatically trust types reported by an arbitrary model file. The skops version
is pinned in each model environment.

## Verification and evaluator access

Every completed training invocation validates tracking records and exports
`tracking_runs.csv` and `tracking_validation.json` into its
`reports/modeling/<training_session>/` directory. The validator checks all 102
finished runs, hierarchy, complete candidate counts, fold metric histories,
parameters, plots, artifacts, both native final models and agreement between
parent test metrics and the saved report. It raises on missing evidence.
Exceptions in tracked run contexts mark those runs `FAILED` and re-raise.
Completeness and status guards use explicit exceptions, so optimization with
`python -O` cannot disable validation. A training run validates its candidate
model, tracking and bundle before switching the atomic selected-release pointer.

Outer-CV evidence is reconciled with its saved predictions: source labels, seeded
fold membership, all fold metrics (including Brier score and confusion counts),
comparison means/sample SDs, and the selected family. Tracking validation checks
the corresponding MLflow fold metrics, row counts, fold-definition artifacts and
parent comparison scores. Altering the CSV predictions and reports together is
detected when they disagree with tracking. Individual inner-search fold models
are not retained, so this check does not claim to regenerate their predictions or
refit models during validation.

Tuning evidence is checked separately from outer evaluation. Every inner/final
search table must contain the declared candidate grid in its original order,
matching parameter columns, the declared train/test split scores, recomputed
means and population SDs, and correct metric ranks. The first maximum ROC-AUC
candidate must match the recorded outer/final choice and selected parameters.
Saved classifiers must have the corresponding tuned and fixed parameters.
Tracking checks parent/final-model parameters and scores, all candidate settings,
fold histories, best-candidate tags, and `cross_validation.json` parameters,
scores and summaries against those tables. Release publication requires completed
model and tracking tuning checks. These checks remain active under `python -O`;
they reconcile recorded tuning evidence and do not retrain inner-fold models.

The CSV records run IDs, parents, tags, parameters, metrics and artifact paths;
the JSON records validation results. These exports belong in the repository.
The SQLite database and artifact store are generated and ignored by Git.
An evaluator can rebuild the live UI by installing requirements and rerunning
training from the supplied fixed data. This reproduces the same evaluation;
it does not provide another independent test sample.

To transfer an existing live tracking store, copy **both** the database and its
artifact tree. MLflow stores absolute local artifact locations, so moving a
database alone or changing the checkout location does not preserve artifact
access. Rebuilding tracking from a fresh checkout is the supported submission
procedure. The serving container will use the selected joblib pipeline and does
not need the tracking database or a running MLflow server.

MLflow references: [Tracking](https://mlflow.org/docs/latest/ml/tracking/),
[SQLite local tracking](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/),
[sklearn model API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).
