# Prediction API and Docker

The API reuses the complete saved preprocessing-and-classifier pipeline. Each
process loads and verifies it once at startup. A corrupt or incompatible artifact
prevents startup. Serving never downloads data, trains, or fits preprocessing.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -e . -r requirements-api.txt
.\.venv\Scripts\python.exe -m heart_disease.api
```

Open http://127.0.0.1:8000/docs. The default model is the release selected by
`reports/modeling/latest.json`. To serve an extracted bundle independently, install
its wheel and the serving requirements, then pass `--model-dir PATH_TO_BUNDLE/model`.
`HEART_MODEL_DIR` is also supported; the explicit CLI argument takes precedence.

```powershell
$body = Get-Content -Raw docs/sample_request.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

For Bash:

```bash
curl -sS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @docs/sample_request.json
```

## Contract

| Endpoint | Behavior |
|---|---|
| `POST /predict` | JSON `{"instances": [...]}`; 1–256 records per request |
| `GET /health` | Process liveness: `{"status":"alive"}` |
| `GET /ready` | Loaded model identity; 503 when no model is loaded |
| `GET /docs` | Interactive Swagger UI |
| `GET /openapi.json` | Machine-readable request and response schemas |
| `GET /metrics` | Per-process JSON operational metrics |
| `GET /monitoring` | Live dashboard; no external assets |

Each record must have exactly these fields: `age`, `sex`, `cp`, `trestbps`, `chol`,
`fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
Every field is required and can be JSON null for fitted imputation. Use numeric
values and numeric category codes; booleans and numeric strings are rejected.
Category codes and measurement rules match the CLI and data dictionary. Diagnosis
and target fields are rejected. Invalid input returns 422 with JSON error details.

The response contains `model_run_id` and one item per input in `predictions`:
`prediction` (0 or 1), `probability_disease` (P(class 1)), and `confidence`
(probability assigned to the predicted class). These are uncalibrated model
probabilities for recorded disease presence in the historical dataset.

All responses include `X-Request-ID`. JSON logs contain that ID, method, route
template, status, duration and model run ID. They exclude feature records, query
strings and prediction values. Unexpected inference failures return a generic 500
response; logs retain the exception type. Model startup failures remain process errors.
Logs also include UTC timestamps and severity. See [monitoring](monitoring.md) for
metric definitions, dashboard access, and capturing logs.

## Build the image

A running **Linux Docker engine** is required, such as Docker Desktop configured
for Linux containers. Successful training publishes a serving bundle. Prepare its
Docker context with:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_docker.py
```

The command follows `reports/modeling/latest.json` to the approved release and
reads its integrity-protected `bundle.json`. It checks the recorded archive SHA-256,
training run ID, and exact model/metadata/schema bytes against that release.
It prints a context path under `build/docker/`, verifies the bundled application
and prediction fixtures, and copies the Dockerfile and pinned requirements. Runtime pins must match
the bundle. Context names depend on archive/build-file hashes. Unchanged inputs
reuse a verified context; altered contexts are rejected and old contexts are retained.

The default remains on the approved release when a later training attempt builds
a bundle but fails artifact logging or publication. `reports/packaging/build_validation.json`
records build output and does not select what the default Docker commands serve.
Missing/corrupt approved artifacts fail the command. A legacy bootstrap snapshot
does not supply a default container release; complete successful training first.

Successful publication retains the exact approved ZIP at
`models/bundles/<archive-sha256>.zip` and binds its path/hash in the immutable
release. Commit `models/bundles/`, the selected `models/releases/` directory and
`reports/modeling/latest.json` together. A fresh checkout can run the context
command directly; it needs neither the ignored `dist/` directory nor retraining
or the MLflow database. Source files use LF line endings so checkout normalization
preserves the application hashes verified against the bundled wheel.

If a retained ZIP is missing or corrupt, restore it from that same repository
revision or a trusted release backup. The command fails instead of choosing an
unapproved rebuild. Older releases whose reference still points into ignored
`dist/` require their original archive or successful training/publication.

Building a bundle writes disposable output under `dist/` and does not approve a
new application/model combination or replace the retained ZIP.
For intentional candidate testing, pass its path explicitly with `--bundle ZIP_PATH`
to either `prepare_docker.py` or `docker_smoke.py`. Full bundle/context verification
still runs, and this override does not promote the candidate. Successful training
publishes the default release after all validation and artifact logging complete.

Replace `CONTEXT_PATH` with the printed path, including its hash:

```powershell
docker build -t heart-disease:0.2.0 CONTEXT_PATH
docker run --rm --name heart-api -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true heart-disease:0.2.0
```

The root Dockerfile is a template for the **generated context**. `docker build .`
at the repository root is not the build procedure. `.dockerignore` allows only the
Dockerfile, serving requirements and bundle. No dataset, MLflow database, training
reports, virtual environment or repository checkout enters the image. The wheel
contains its Python modules, while installed dependencies omit training, plotting,
experiment-tracking and test tools.

The image uses Python 3.13.7 slim Bookworm, numeric system libraries and UID/GID
10001. Build-time verification runs as that user. Docker's healthcheck probes
`/ready`. Uvicorn runs one worker in exec form on port 8000; Kubernetes replicas
will provide scaling. Python/package versions are pinned, but the base tag and
Debian repository can receive updates, so Docker image bytes are not claimed
identical across builds. Verification records the actual image ID.

For local Kubernetes, continue with the [Minikube deployment guide](../deployment/kubernetes/README.md).
It uses the approved bundle's full SHA in the image tag, loads that image into the
cluster, and exposes the API through a LoadBalancer Service.

## Automated proof

```powershell
.\.venv\Scripts\python.exe scripts/docker_smoke.py
```

This prepares the context, builds the image, runs it on a temporary loopback port,
verifies all seven packaged prediction cases, invalid requests/readiness/OpenAPI,
waits for Docker health, and checks the non-root user and absence of training tools.
It saves build logs, image/container inspections, HTTP results and server logs under
`reports/docker/`, then removes its own temporary container. The image remains.
Use `--bundle ZIP_PATH` or `--image TAG` to choose explicit inputs.

Missing Docker or a stopped/non-Linux engine fails the command and records
**blocked**, never a successful container test. GitHub CI enables
`scripts/ci.py --docker` and requires this check before uploading the verified
bundle. This stage does not push an image or deploy it.

API verification can run without Docker:

```powershell
.\.venv\Scripts\python.exe scripts/ci.py
```

This recreates training outputs, installs the wheel in an isolated runtime, adds
API dependencies and exercises real HTTP with an empty data workspace. Its test
server is stopped afterward. Docker Desktop 4.95.0 is installed on this machine,
but this machine cannot run virtualization, so container execution and its screenshot
will be verified on a friend's PC. Native HTTP validation is
recorded separately from container validation.

Local verification passed all 19 extended pipeline steps, followed by a final suite
of 100 passing tests. Seven HTTP prediction cases passed from the final wheel
installed with serving dependencies only. Artifact publication also preserves
output-parent Windows permissions; normal-user access to the bundle, Docker
context and selected model was verified outside the sandbox account.
See [verification evidence](../reports/api_docker_validation_20261017.json) and
[this PC's Docker/BIOS setup instructions](docker_setup.md).

References: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[FastAPI Docker deployment](https://fastapi.tiangolo.com/deployment/docker/),
and [Docker build practices](https://docs.docker.com/build/building/best-practices/).
