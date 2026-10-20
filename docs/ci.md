# Continuous integration

## Current independent CI/CD workflow

The complete current workflow is documented in [acceptance-suite.md](acceptance-suite.md)
and [git-publication.md](git-publication.md). On pushes to `main` and manual runs,
GitHub first runs a quality job, then independently trains, verifies MLflow,
builds Docker, deploys to a temporary kind cluster with a real LoadBalancer,
evaluates labeled requests, checks monitoring, records video and generates the
submission evidence. Successful trusted runs publish the tested image and create
or update a DEMO retraining issue. Pull requests run read-only quality checks.

The deployment job explicitly grants `packages: write` and `issues: write` to its
built-in token; source checkout remains read-only. It never connects to a friend's
PC and never commits source. Evidence retention is seven days; download final
submission evidence promptly. `requirements-acceptance.txt` pins the full suite.

The standalone `scripts/ci.py` command below remains the reusable inner native
pipeline. `--evidence` additionally collects per-test methods and branch coverage.
The following original validation notes are historical, not a claim that the new
hosted full workflow has already executed.

## Historical inner-pipeline validation

The initial CI validation on Windows/Python 3.13.7 passed: 63 tests, all 16 pipeline steps,
102 MLflow runs, an identical bundle rebuild, and seven prediction cases from the
installed wheel with an empty data workspace. See the
[verification record](../reports/ci_validation_20261017.json).
A hosted Ubuntu run remains pending creation of the GitHub repository.
API and Docker checks were subsequently added; see [API/container instructions](api_docker.md).

The workflow `.github/workflows/ci.yml` runs on pushes, pull requests and manual
dispatch. It uses an Ubuntu 24.04 runner and Python 3.13.7. No repository secrets
or running MLflow server are required. Workflow permissions are `contents: read`.

## What runs

1. Install the exact development environment in `requirements-lock.txt`.
2. Check dependencies, lint and formatting, then run tests marked `not integration`.
3. Recreate EDA and the split from the cached, checksum-validated UCI raw data.
4. Run nested cross-validation, final tuning, evaluation and MLflow logging.
5. Validate model predictions and all 102 MLflow runs. Training publishes its
   selection pointer only after its model, tracking and bundle validations pass.
6. Run the integration regression tests against the freshly generated model.
7. Rebuild the bundle and require the same archive checksum as the training build.
8. Verify the bundle, install its wheel in a separate virtual environment with
   inference dependencies only, and verify prediction fixtures using `python -I`
   with an empty data workspace.
9. Install API dependencies in that isolated environment and exercise real HTTP
   predictions, readiness, invalid requests and OpenAPI, with an empty data workspace.
10. With `--docker`, require the serving image build and container smoke test.
    The GitHub workflow enables this flag; local runs may omit it without Docker.

`scripts/ci.py` creates a new workspace inside its evidence directory. It copies
source/configuration/docs and `data/raw/`, and regenerates processed data, models,
reports and the tracking store. Historical checkout outputs are never prerequisites.
The raw cache avoids making training depend on UCI website availability. Its
attribution and `source.json` must be committed along with the source data.

Tracking validator tests generate a synthetic, complete MLflow lineage in a temporary
store and check corruptions under normal and optimized Python. Separate tracking tests
fit and reload actual native MLflow models. The model/API `integration` tests require a
completed model workspace; CI runs them after training, never skips them entirely.

## Reproduce locally

From the project root, install the pinned environment and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/ci.py
```

On Linux use `.venv/bin/python`. The default output is a new timestamped directory
under `reports/ci/`. An explicit `--output PATH` must name a directory that does not
exist, so a previous run cannot accidentally supply stale evidence. Runtime installation
needs package-index access or a populated pip cache. Existing project model selections
and experiment records are not changed.

For fast checks only:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m pytest -m "not integration"
```

## Failures and evaluator evidence

The isolated API HTTP check also verifies JSON metric deltas, dashboard delivery,
and request-ID correlation in captured structured logs. Monitoring regression
tests run with the unit suite. Docker verification exercises the same HTTP
metrics/dashboard checks inside the container when an engine is available.

Commands run sequentially and the first nonzero exit stops the pipeline. Each
command has a console log, `steps.json` records exit codes and timings, and
`summary.json` records the overall result. Pytest exports JUnit XML reports.
Installation uses Bash `pipefail` so piping output to a log cannot hide pip errors.

The workflow always attempts to upload `ci-evidence-RUN-ATTEMPT`, containing
installation/stage logs, JUnit results, summaries, data, generated reports, models
and tracking evidence, including partial output after failures. Only a successful
pipeline uploads the separate `verified-model-RUN-ATTEMPT` bundle artifact. Both
artifacts are retained for 30 days; download evidence needed for final submission.
The diagnostic artifact can contain an intermediate model when a later check fails;
use only the separate verified bundle for serving.

The workflow does not deploy or change repository files. To enforce CI before
merging, configure the GitHub branch rule to require `Test, train and verify`.
This repository setting is separate from the workflow file.

## Activating GitHub Actions

Commit the source, tests, scripts, requirement files, docs, raw data and `.github/`.
`.gitattributes` preserves bytes of the recorded data/models/reports used in checksum
checks. Commit the approved `models/bundles/` ZIP with its selected release and
selection pointer; `dist/` remains disposable. Source/configuration files use LF.
Local virtual environments, generated CI workspaces, build output and MLflow
databases are ignored. Push to a GitHub repository, open **Actions → CI**, and retain
the successful run URL and screenshot for the assignment.

GitHub action references follow the official usage documentation:
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python), and
[upload-artifact](https://github.com/actions/upload-artifact).
