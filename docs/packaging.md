# Model packaging and reproducibility

The serving artifact is one fitted sklearn Pipeline: imputation, documented
one-hot encoding, numerical scaling and the selected classifier. Loading it
never fits a transformer or classifier. The 13 original inputs are converted
into the same 28 features used during training.

## Build from the checkout

Use Python 3.13 (the development interpreter is 3.13.7). After training:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-package.txt
.\.venv\Scripts\python.exe -m heart_disease.package_model
```

`dist/heart-disease-<training-run-id>-<application-version>-<archive-sha256>.zip` contains:

- `wheels/heart_disease_mlops-<application-version>-py3-none-any.whl`: installable
  application code; `bundle_manifest.json` records its exact relative path in `wheel`.
- `model/model.joblib`: complete fitted preprocessing and classification pipeline.
- `model/metadata.json`: selected parameters, scores, data/code hashes and MLflow references.
- `model/feature_schema.json`: original input names, category codes and null handling.
- `model/preprocessing_state.json`: readable fitted medians, modes, scales and encoded feature names.
- `requirements-runtime.txt`: all eleven pinned prediction dependencies.
- `fixtures/`: raw-input requests, case descriptions and frozen expected responses.
- `bundle_manifest.json`: payload SHA-256 hashes, environment, build tools and code lineage.
- `packaging.md`: these instructions.

A sibling `.zip.sha256` records the archive hash. SHA-256 detects accidental
changes; it is not a signature or a substitute for trusting the artifact source.

The package builder verifies the selected artifact and environment, builds the
wheel with pinned tools, and never retunes or retrains. It fixes wheel/ZIP timestamps
using the original model creation time, sorts files and normalizes archive
permissions. Building twice from unchanged code, documentation, model and build
environment must produce the same archive SHA-256. This is a same-environment
build guarantee; different platforms/build tools are not promised identical ZIP bytes.

## Install and predict in a fresh environment

Extract the ZIP into a new directory, such as `D:\model-release`. Keep the
virtual environment outside the extracted bundle so its integrity check can
reject extra payload files. The extracted bundle has no source-checkout dependency.
These PowerShell commands use the bundle directory as the working directory and
create a new sibling `model-runtime` directory. Choose another sibling name if
that environment already exists. Read the wheel path from the manifest instead
of substituting a package version:

```powershell
py -3.13 -m venv ..\model-runtime
$python = (Resolve-Path ..\model-runtime\Scripts\python.exe).Path
$manifest = Get-Content -Raw .\bundle_manifest.json | ConvertFrom-Json
$wheel = Join-Path (Get-Location).Path $manifest.wheel
& $python -m pip install -r requirements-runtime.txt
& $python -m pip install --no-deps $wheel
& $python -m pip check
& $python -I -m heart_disease.verify_bundle --bundle .
& $python -I -m heart_disease.predict --model-dir .\model --input .\fixtures\request.json
```

On Linux/macOS, from the extracted bundle directory, use the equivalent commands:

```bash
python3.13 -m venv ../model-runtime
python=../model-runtime/bin/python
wheel=$("$python" -c 'import json; print(json.load(open("bundle_manifest.json"))["wheel"])')
"$python" -m pip install -r requirements-runtime.txt
"$python" -m pip install --no-deps "$wheel"
"$python" -m pip check
"$python" -I -m heart_disease.verify_bundle --bundle .
"$python" -I -m heart_disease.predict --model-dir ./model --input ./fixtures/request.json
```

The PowerShell procedure has been exercised in a fresh Windows environment.
Linux/macOS execution remains unverified; Docker execution requires a working
Linux Docker engine. Dependency installation needs package-index
access or a separately prepared dependency wheel cache. The ZIP contains the
application wheel, not the third-party dependency wheels.

`heart-verify-bundle` and `heart-predict` are equivalent installed console commands.
Verification checks file integrity before loading the model, all pinned runtime
versions, Python major/minor, application version, classes and probabilities.
It also compares the imported package and bundled wheel source bytes with the
manifest's application hashes. A same-version application with different code
is rejected. Content-addressed archive names preserve earlier releases when the
application changes around an unchanged model; identical rebuilds reuse the same
verified release without overwriting it. Successful release publication also
retains the exact approved ZIP under `models/bundles/<archive-sha256>.zip`.
Commit that directory with the selected release and selection pointer. A clean
checkout then has the original serving bytes without reconstructing a historical
ZIP from its source or requiring the training/tracking workspace. `dist/` remains
disposable build output. Source text uses LF; historical model/data/report bytes
are preserved by `.gitattributes`.
Fixtures cover ordinary development records, original missing values, explicit
numeric/categorical nulls and all-null imputation. Expected values are predictions
from the already fitted model, not clinical truth or additional test-set evaluation.
Classes must match exactly; probabilities/confidence use rtol=atol=1e-12.

## Reuse in the API

```python
from pathlib import Path
from heart_disease.predict import load_predictor

predictor = load_predictor(Path("/app/model"))  # once at process startup
response = predictor.predict({"instances": [raw_feature_record]})
```

In the checkout, `load_predictor()` resolves `reports/modeling/latest.json` to a
complete immutable directory under `models/releases/`. Training creates an
unselected candidate first and completes saved-model reconciliation, tracking,
bundle verification and artifact logging before atomically replacing that one
selection pointer. The model, metadata/schema and validation evidence become
visible together. Failed checks leave the previous selection intact. Existing
checkouts without the new pointer fields fall back to the retained bootstrap
snapshot in `models/selected/`. Extracted bundles still use their explicit
`model/` directory, as shown above.

The loader validates the model hash, feature schema, saved structure and the four
training dependency versions before returning the reusable predictor. It refuses
an incompatible environment instead of silently loading an unsupported sklearn
artifact. Each request uses the fitted pipeline. The JSON contract returns the
model run ID, predicted class, modeled disease probability and predicted-class
confidence. No EDA previews, labels, data download, model fitting or tracking
server are required for prediction. MLflow references in metadata retain lineage;
prediction does not contact that backend.

## Reproduce training versus reproduce serving

Reproducing **serving** means installing the wheel and runtime requirements and
loading the exact saved pipeline. Reproducing **training** means rebuilding the
pipeline from fixed raw data, split manifest, seed 42 and documented grids:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m heart_disease.download
.\.venv\Scripts\python.exe -m heart_disease.eda
.\.venv\Scripts\python.exe -m heart_disease.train --jobs 2
.\.venv\Scripts\python.exe -m heart_disease.validate_model
.\.venv\Scripts\python.exe -m heart_disease.validate_tracking
.\.venv\Scripts\python.exe -m heart_disease.package_model
```

Training fits preprocessing inside CV folds and refits the final pipeline on the
development partition only. The reserved test rows remain excluded from fitting.
Each retraining has a new run ID and new archive metadata, even if predictions
match. The bundle preserves original training code hashes separately from current
application code hashes, so later packaging/inference changes do not rewrite
historical training provenance.

`requirements.txt` pins the shared core dependencies; `requirements-runtime.txt`
pins the complete inference dependency closure; `requirements-package.txt` pins
build tooling; `requirements-train.txt` includes plots and MLflow; developer tools
are in `requirements-dev.txt`. `requirements-lock.txt` captures the verified
development environment with platform-specific dependency markers.
`requirements-api.txt` extends the runtime set with the pinned API server dependencies.

The generated ZIP and temporary test environments are ignored by Git. Commit the
package/build code, requirements, approved ZIP in `models/bundles/`, selected model
and schema, fixtures' source
generation logic, documentation and `reports/packaging/` evidence. An evaluator
can rebuild the bundle using these commands from a fresh checkout. A Docker build
and isolated container prediction proof use the implemented
[container preparation and verification scripts](api_docker.md) in the source
checkout; actual execution
requires a running Linux Docker engine.
