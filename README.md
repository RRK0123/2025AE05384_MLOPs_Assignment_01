# Heart Disease MLOps Assignment

This project uses an installable Python package and command-line stages.
The repository includes data/EDA, model development, MLflow, saved pipelines,
reproducible packaging, a prediction API, monitoring, Docker and Kubernetes.
Two independent full acceptance tracks train and verify from a fresh workspace:

- **Windows:** double-click `RUN_ALL.bat` on the virtualization-capable PC.
- **GitHub:** Actions runs CI and deploys to its own temporary Kubernetes cluster.

Both generate detailed test reports, code/requirement coverage, statistical
evidence, screenshots, continuous/chaptered videos and a ten-page submission PDF.
Healthy/degraded monitoring scenarios demonstrate explicit retraining requests.
Read [complete execution instructions](docs/acceptance-suite.md) and
[commands to publish the repository yourself](docs/git-publication.md).
No Git commit or push is performed by the automation.

## Setup

Verified with Python 3.13. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For training without developer tools, install `requirements-train.txt` instead.
For EDA only, install `requirements-eda.txt`.
It installs the local package with `-e .` and the plotting dependencies.
On Linux/macOS replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Dependency roles:

- `requirements.txt`: shared data and preprocessing dependencies.
- `requirements-runtime.txt`: the complete pinned inference dependency set.
- `requirements-package.txt`: application installation and pinned bundle build tools.
- `requirements-eda.txt`: package installation and plotting tools.
- `requirements-train.txt`: training, evaluation, plotting, MLflow and release build tools.
- `requirements-dev.txt`: training/EDA, packaging tools, Pytest, and Ruff.
- `requirements-lock.txt`: exact development dependency versions, plus `-e .`,
  with platform-specific dependency markers;
  this can be installed directly to reproduce the verified environment.
- `requirements-api.txt`: complete pinned serving dependencies, excluding application installation.

## Run the current pipeline

```powershell
.\.venv\Scripts\python.exe -m heart_disease.download
.\.venv\Scripts\python.exe -m heart_disease.eda
.\.venv\Scripts\python.exe -m heart_disease.validate
.\.venv\Scripts\python.exe -m heart_disease.train --jobs 2
.\.venv\Scripts\python.exe -m heart_disease.validate_model
.\.venv\Scripts\python.exe -m heart_disease.validate_tracking
.\.venv\Scripts\python.exe -m heart_disease.predict --input docs\sample_request.json
```

Equivalent installed console commands are `heart-download`, `heart-eda`,
`heart-validate`, `heart-train`, `heart-validate-model`, and `heart-predict`.
The module commands above do not require environment activation.
Downloading needs internet access; subsequent runs validate and reuse cached data.
Use `python -m heart_disease.download --force` to explicitly refresh the source.
The installed package finds its checkout root independently of the current working
directory. Set `HEART_DISEASE_PROJECT_ROOT` to explicitly choose a data workspace
when running an installed wheel or from another environment.

## Checks

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m pytest
```

Tests cover binary target mapping, source immutability, retention of missing
predictors, invalid diagnosis/category/measurement rejection, and reuse of
training-fitted preprocessing statistics for held-out input.
Model tests also cover metrics, selection rules, complete pipeline serialization,
forest preprocessing, and prediction-input validation.
Tracking tests check persisted candidate metrics/history/artifacts and failed-run status.

## Continuous integration

GitHub Actions runs lint, formatting, tests, fresh EDA/training, model and tracking
validation, a reproducible bundle rebuild, and installation/prediction checks in
an isolated inference environment. Logs and reports are uploaded on failure;
the verified bundle is uploaded only after every check succeeds.

Run the same pipeline locally without changing the current selected model:

```powershell
.\.venv\Scripts\python.exe scripts/ci.py
```

See [CI instructions](docs/ci.md) for test stages, evidence, and GitHub activation.

## Prediction API and container

```powershell
.\.venv\Scripts\python.exe -m pip install -e . -r requirements-api.txt
.\.venv\Scripts\python.exe -m heart_disease.api
```

Open http://127.0.0.1:8000/docs. `POST /predict` accepts the same original-feature
JSON as the CLI. `/health` checks liveness and `/ready` checks the loaded model.
See [API and Docker instructions](docs/api_docker.md) for requests, response fields,
container preparation/build/run commands and automated HTTP/container proof.
Default Docker commands use the approved selected release and verify its recorded
bundle hash and model identity. Explicit `--bundle` supports candidate testing.
Open http://127.0.0.1:8000/monitoring for the live dashboard or `/metrics` for JSON
service metrics. See [monitoring and logging](docs/monitoring.md) for definitions,
log capture, limitations, and verification.
`heart-validate` additionally verifies source hashes, deterministic splits, saved
row alignment, encoded feature schemas, transformed values, and required figures.
It writes the results to `reports/validation.json` and fails on invalid artifacts.
`heart-validate-model` verifies candidate/selected model hashes, nested-CV fold
boundaries, training-fitted statistics, and inference output. It regenerates test
predictions from the exact saved artifact and reconciles them with the recorded
predictions and metrics. Tracking validation uses explicit exceptions and remains
active with `python -O`. Regression tests cover wrong classifiers, same-version
code changes, failed release promotion and optimized tracking checks.
CV validation reconstructs seeded outer/inner/final folds, checks each out-of-fold
record against its source label and validation fold, recomputes outer metrics and
summary means/SDs, and derives the winner from those scores. Tracking validation
also reconciles the logged fold metrics, row counts, fold definitions and parent
comparison scores with that evidence.
Tuning validation also checks each declared grid, parameter columns, split scores,
means/population SDs, ranks and best-candidate choice. It reconciles selected
parameters with fitted classifiers and with MLflow's parent/final-model settings,
candidate histories and JSON artifacts. Publication requires these checks to pass.

## Local Kubernetes deployment

For the complete independent Windows workflow, read
[the acceptance-suite guide](docs/acceptance-suite.md), configure
`configs/acceptance.json`, and double-click `RUN_ALL.bat`.
It trains, tracks, tests, deploys, evaluates and records the entire demonstration.
`COLLECT_DIAGNOSTICS.bat` collects evidence and `STOP_DEMO.bat` stops owned services.
The independent GitHub track deploys into its own temporary kind cluster.

The older `RUN_DOCKER_KUBERNETES.bat` and
[serving-only guide](deployment/windows/START_HERE.md) remain available for
deployment of an already approved model.

The [Minikube deployment guide](deployment/kubernetes/README.md) covers the approved
image build/test/load, namespace and Deployment, LoadBalancer exposure, health
probes, resource limits, deployed prediction checks, monitoring, evidence capture
and rollback. Full execution records its actual deployment results and screenshots
in that run's evidence; static validation alone is never deployment proof.

## Structure

- `src/heart_disease/`: schema, cleaning, preprocessing, acquisition, EDA,
  modeling, training, JSON prediction, plotting, reporting, and validation modules.
- `tests/`: automated data, preprocessing, model, and inference checks.
- `data/raw/`: preserved source and provenance/checksum manifest.
- `data/processed/`: cleaned data, fixed train/test splits, and encoded previews.
- `reports/`: findings, numerical tables, seven figures, and validation output.
- `docs/`: field definitions and project design.
- `models/runs/`: separately saved model pipelines and metadata for every run.
- `models/selected/`: bootstrap snapshot retained for older checkouts.
- `models/releases/`: complete versioned model releases with validation evidence.
- `models/bundles/`: exact approved serving ZIPs; commit with releases and their pointer.
- `dist/`: generated model bundle ZIP, application wheel and checksum (Git ignored).
- `reports/packaging/`: manifest and build/isolated-environment verification evidence.
- `tracking/`: generated MLflow SQLite database and artifact store (Git ignored).
- `deployment/kubernetes/`: Namespace, Deployment, LoadBalancer Service and execution guide.
- `screenshots/`: location reserved for real execution evidence.

See [the project structure and delivery path](docs/project_structure.md) for the
planned training -> saved pipeline -> Docker -> Kubernetes -> monitored API flow.
Training produces the model before the serving image is built. The API loads
that fitted model pipeline at startup and processes prediction requests.

## Model development and results

The implemented comparison uses Logistic Regression and Random Forest, with a
majority-class baseline. Every model includes fold-fitted imputation and one-hot
encoding; Logistic Regression scales continuous inputs, while the forest uses
their imputed original values. All 13 predictors become 28 transformed features.

Model families are compared using five stratified outer folds, with three inner
folds for parameter tuning. Selection uses mean outer ROC-AUC only. Each family
is then retuned on all 242 development rows using five folds; only the already
selected family is evaluated on the reserved 61 rows.

The completed run selected Logistic Regression: mean outer ROC-AUC 0.9101,
versus 0.9037 for Random Forest. The difference does not establish statistical
superiority. Selected parameters: C=0.1, class_weight=balanced.
Test accuracy=0.8852, precision=0.8387, recall=0.9286, ROC-AUC=0.9654.
There were 28 true negatives, 5 false positives, 2 false negatives, and 26 true positives.

`reports/modeling/latest.json` atomically selects the latest validated model
release and its development report. The prediction, packaging and validation
commands resolve this pointer automatically. Training stages a candidate, verifies
the saved model, tracking and bundle, then switches this single pointer. A failed
validation or publication leaves the previous release selected.
Each `reports/modeling/<run_id>/` stores candidate search results, outer-fold
scores, fold definitions, predictions, comparison/ROC/PR/confusion plots, metrics,
and a written report. Each training invocation retains the prior run directories.
Do not alter features or settings based on the test result. These scores describe
a small historical sample; probability calibration has not been separately validated.

The selected release's `model.joblib` contains the full fitted preprocessing-plus-model
Pipeline. Older checkouts without a release pointer use `models/selected/`.
The prediction command accepts the original 13 feature fields in an
`instances` list, supports JSON null for imputation, rejects invalid fields/codes,
and returns prediction, modeled disease probability, and confidence for the
predicted class. It verifies the saved artifact hash and schema before loading.

## Experiment tracking

Every successful training invocation creates a parent MLflow run, 15 outer validation runs,
84 hyperparameter candidate runs and two final-model runs. They record settings,
metrics, plots and artifacts; both final models also have native MLflow exports
with signatures, input examples and pinned environments.

```powershell
.\.venv\Scripts\python.exe -m heart_disease.tracking --port 5000
```

Open http://127.0.0.1:5000 and select `heart-disease-classification`.
The UI reads the generated local SQLite backend; training needs no running server.
Each modeling report directory includes `tracking_runs.csv` and
`tracking_validation.json` as portable tracking evidence. An evaluator can
recreate the live UI by rerunning training from a fresh checkout.
See [experiment tracking instructions](docs/experiment_tracking.md) for the
run hierarchy, model reload example and backend portability details.

## Packaging and reproducibility

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-package.txt
.\.venv\Scripts\python.exe -m heart_disease.package_model
```

The generated model bundle includes an application wheel, the selected complete
pipeline, schema, fitted preprocessing statistics, pinned runtime requirements,
raw-input prediction fixtures and a checksum manifest. Rebuilding unchanged
inputs with the same build environment produces identical archive bytes.
Successful publication retains the approved ZIP under `models/bundles/` for
version control with the selected release and pointer. A clean checkout can
prepare the default Docker context directly without ignored build outputs,
training data or a tracking database. Commit the matching source revision; its
LF line endings preserve the bundled application's identity checks.
The reusable predictor loads once, verifies hashes/schema/versions, and applies
the saved transformers for every request.

See [packaging and clean-environment instructions](docs/packaging.md) to install
and verify the bundle independently of this checkout. Serving needs no tracking
server, training data or plotting dependencies. Docker and Kubernetes execution
proof is generated by each complete acceptance run.

## Read the EDA

- [Findings and decisions](reports/eda_findings.md).
- [Data dictionary](docs/data_dictionary.md).
- `reports/figures/`: missingness, class balance, histograms, continuous
  correlations, categorical disease frequencies, boxplots, and outlier flags.
- `reports/tables/`: the numerical values behind the figures.
- `reports/data_quality.json`, `reports/preprocessing_audit.json`, and
  `reports/validation.json`: machine-readable evidence.

## Data and evaluation choices

The original UCI processed Cleveland subset has 303 records, 13 predictors,
and one diagnosis column. Diagnosis `num=0` maps to target 0; `num=1,2,3,4` map
to target 1. The original diagnosis is excluded from inputs. The assignment's
wording about 14+ features is clarified in the findings.

`heart_clean.csv` preserves nulls until training-fitted imputation handles them.
`train.csv` and `test.csv` use an 80/20 stratified split with seed 42; the split
manifest records zero-based source row positions and the raw checksum.
Only integrity checks and overall target balance use the full source.
Feature distributions and outcome relationships use training observations only.
Statistical outliers are flagged and retained. Integer category codes are not
included in continuous Pearson correlations.

The `*_preprocessed_preview.csv` files demonstrate train-fitted imputation,
scaling, and one-hot encoding. Rows match the corresponding split CSVs; labels
are excluded. For model selection, fit a fresh preprocessor inside each
cross-validation Pipeline, rather than using these already transformed previews.

## Data attribution

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
*Heart Disease* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X.

Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
The target indicates recorded disease presence, not future disease risk.
