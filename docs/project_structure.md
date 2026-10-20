# Project structure and delivery path

The project runs through Python commands. Shared schema and preprocessing modules
will be reused by training and inference so the API receives the same feature
representation that the selected model saw during training.

## Current structure

```text
src/heart_disease/
  config.py           Workspace paths and fixed split settings
  schema.py           Original feature names, category codes, and target definition
  data.py             Source parsing, validation, and deterministic cleaning
  download.py         UCI acquisition, provenance, and checksum checks
  preprocessing.py    Unfitted imputation, encoding, and scaling transformer
  modeling.py         Model families, parameter grids, metrics and selection rules
  train.py            Nested CV, final tuning, selection, evaluation and model export
  tracking.py         Local MLflow run hierarchy, logging, model exports and UI command
  validate_tracking.py Tracking completeness checks and portable evidence export
  predict.py          Validated JSON inference using the saved pipeline
  api.py              FastAPI prediction, startup model loading, readiness and request logs
  monitoring.py       Thread-safe process counters and bounded traffic history
  dashboard.py        Self-contained live monitoring page, packaged with source checksums
  package_model.py    Deterministic application wheel and fitted model bundle
  verify_bundle.py    Payload, runtime and prediction fixture verification
  releases.py         Versioned model releases and atomic selection pointer
  model_report.py     Comparison and test figures plus modeling report
  validate_model.py   Model artifact and CV boundary verification
  cv_validation.py    Prediction-derived CV metrics, fold membership, and report reconciliation
  tuning_validation.py Search-table, fitted-parameter and MLflow tuning reconciliation
  plotting.py         Headless EDA figures
  reporting.py        Findings from computed analysis tables
  eda.py              EDA orchestration and artifact generation
  validate.py         Reproducibility and generated-artifact checks
tests/                Data, modeling, tracking, release and CI regression tests
scripts/ci.py         Fresh-workspace CI orchestration and retained evidence
scripts/prepare_docker.py Verified serving bundle to immutable container build context
scripts/docker_smoke.py Image build, real container HTTP checks and cleanup
scripts/verify_api.py Installed API HTTP verification with an empty data workspace
scripts/smoke_api.py  Shared HTTP contract and prediction fixture checks
scripts/deployment_runner.py Docker/Minikube execution, failure gates and diagnostic ZIPs
scripts/run_deployment.ps1 Windows prerequisite checks and isolated dependency setup
scripts/deployment_tunnel.ps1 Interactive LoadBalancer tunnel with retained logs
scripts/create_deployment_handoff.py Small approved serving-code/model transfer ZIP
RUN_DOCKER_KUBERNETES.bat One-click Windows deployment after prerequisites
COLLECT_DEPLOYMENT_LOGS.bat Read-only collection of fresh diagnostics
docs/monitoring.md    Metric definitions, dashboard and request log instructions
Dockerfile            Non-root Linux serving image (execution requires Docker)
.dockerignore         Small serving-only context allowlist
.github/workflows/ci.yml GitHub Actions checks, training and verified bundle upload
data/raw/             Original dataset and provenance manifest
data/processed/       Cleaned data, fixed splits, and preprocessing previews
reports/              EDA findings, plots, tables, and validation output
docs/                 Data dictionary and project design
models/runs/          Fitted candidate pipelines and run metadata
models/selected/      Retained bootstrap snapshot for legacy checkouts
models/releases/      Validated versioned releases selected by reports/modeling/latest.json
models/bundles/       Approved content-addressed serving ZIPs retained for clean checkouts
dist/                 Generated model bundle ZIP and checksum (Git ignored)
reports/packaging/    Build manifests and reproducibility verification evidence
reports/ci/           Timestamped local CI evidence and fresh workspace (Git ignored)
tracking/             Generated MLflow SQLite database and artifacts (Git ignored)
deployment/kubernetes/ Namespace, Deployment, LoadBalancer Service and Minikube guide
deployment/windows/   Standalone instructions for execution without ChatGPT
screenshots/          Location reserved for execution evidence
pyproject.toml        Installable package and command-line entry points
requirements.txt      Shared data and preprocessing dependencies
requirements-runtime.txt Complete pinned prediction dependency set
requirements-package.txt Pinned package/bundle build tools
requirements-eda.txt  Local package installation plus plotting dependencies
requirements-train.txt Training/evaluation dependencies
requirements-dev.txt  Training/EDA plus tests and linting
requirements-api.txt  Complete pinned serving dependencies
requirements-lock.txt Exact installed development environment
```

## Planned stages

The following are implementation destinations, not completed components:

| Stage | Implementation destination | Output / contract |
|---|---|---|
| Container execution | Implemented Docker build and test scripts | Build/run proof on a friend's PC with a working Linux engine |
| Kubernetes execution | Implemented `deployment/kubernetes/` manifests and guide | Actual rollout, LoadBalancer predictions and monitoring evidence on a working PC |
| CD extensions | `.github/workflows/ci.yml` | Add publishing and local-cluster deployment to CI's implemented container checks |
| Submission | Report, `screenshots/`, demonstration video | Evidence and evaluator access instructions |

## Lifecycle

```text
Download -> clean and explore -> train + track -> select + save pipeline
                                                   |
                                                   v
                                       build serving Docker image
                                                   |
                                                   v
                                       local Kubernetes deployment
                                                   |
                                                   v
                                       request -> prediction + logs
```

Training consumes the fixed original-feature splits and constructs a fresh
preprocessor inside each cross-validation Pipeline. The transformed EDA preview
files are not inputs for cross-validation. API containers will load the selected
fitted pipeline and serve requests; they will not download data, plot EDA, or
train a model on startup.

The local Kubernetes submission must be reproducible from a fresh checkout. Its
README will specify the cluster choice, Docker image build/load procedure,
manifest application, exposure setup, sample JSON request, expected response
shape, and monitoring access. A local-cluster workflow must actually run before
we claim deployment or CI/CD proof.
