# Complete acceptance suite

## Two independent tracks

Both tracks start from source and the attributed raw-data cache, create a fresh
workspace, train, verify MLflow, test, package, build Docker, deploy Kubernetes,
evaluate the actual endpoint, test monitoring/retraining requests, and generate
their own evidence. They never depend on each other's run results.

- Windows: `RUN_ALL.bat`; Docker Desktop and a dedicated Minikube cluster.
- GitHub: Actions `Independent GitHub CI and CD`; hosted Ubuntu with a temporary
  kind cluster and an actual LoadBalancer supplied by cloud-provider-kind.

GitHub deletes its cluster after evidence. Windows leaves API/MLflow/demo services
running after success. No automatic model retraining or replacement occurs.

## Windows prerequisites

Use a Windows 10/11 PC with working hardware virtualization/WSL2, Python 3.13 with
the Python launcher, Docker Desktop using Linux containers, kubectl and Minikube
on PATH. Recommended host resources: 4 CPU cores, 16 GB RAM and 10 GB free disk.
Minikube receives two CPUs and 4 GB; training/browser/recording need additional
host resources. Start Docker Desktop before launching. First setup requires
internet for Python packages and container images. Close the old port-8080 demo
tunnel before running this new suite.
After saving the earlier kit's evidence, stop its dedicated cluster if necessary
with `minikube stop -p heart-mlops`; the new suite never stops unrelated clusters.

Keep an active, unlocked desktop during capture. Edge/Chrome can be used if
installed; otherwise bootstrap installs Playwright Chromium. If routing needs
elevation, run the batch as administrator using the same Windows account. Any
Windows UAC/prerequisite setup is performed before the unattended test stages.

Extract the full repository package to a writable local directory. Open
`configs/acceptance.json` and supply student/course/repository details. Then
double-click `RUN_ALL.bat`. No environment activation or staged commands are
needed. Environment setup may take several minutes.

Recording begins after dependency bootstrap, before the suite's prerequisite
tests. Bootstrap has its own transcript. It does not pretend to be part of the
screen recording. Keep other private windows closed because Windows capture
records the actual desktop. Linux Actions capture records the actual Xvfb desktop.

## What is executed

1. Browser and screen recorder readiness; execution prerequisites.
2. Fresh CI workspace: dependency checks, lint/format, unit tests, EDA, data
   validation, training/tuning/comparison, model/tracking validation, integration
   tests, identical bundle rebuild and isolated wheel/API installation checks.
3. Live MLflow and native API, with screenshots from this training session.
4. Docker build, non-root/read-only container checks and real HTTP fixture tests.
5. Owned Kubernetes cluster, strict server validation, rollout, ready image IDs,
   real LoadBalancer exposure and HTTP model identity.
6. Every reserved test record through the deployed API; saved-model agreement;
   accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix,
   Brier score and seeded stratified bootstrap 95% intervals.
7. Null inputs, batch boundaries, invalid values/schema, and concurrent traffic.
8. Counter reconciliation, request-log correlation, drift and performance checks,
   healthy replay and controlled input-corruption retraining-request demonstration.
9. Detailed HTML/PDF, JUnit, requirement coverage, code coverage, ten-page
   submission PDF, continuous video, chaptered final video and evidence archive.

The unit-test evidence includes each executed assertion, fixture references,
parameterized inputs, source location and actual pytest outcome. Failed setup or
teardown is a failure; skipped tests retain their reasons. Cluster unit mocks
exercise controller logic only. Real Docker/Kubernetes acceptance cases provide
deployment proof. A failed prerequisite blocks dependent cases instead of
creating synthetic passing results.

## Configuration and retraining criteria

The entire configuration is copied into the evidence. Defaults:

| Setting | Default | Rationale |
|---|---:|---|
| Minimum labeled records | 50 | Educational evidence gate for the 61-row holdout |
| Minimum records per class | 20 | Prevent a decision from a nearly absent class |
| Recall drop | 0.10 | Material absolute change from recorded reference |
| ROC-AUC drop | 0.10 | Material ranking deterioration from reference |
| Numeric KS effect size | 0.25 | Configurable distribution-change alert |
| Categorical total variation | 0.20 | Configurable distribution-change alert |
| Maximum load error rate | 0.01 | Operational acceptance criterion |
| Maximum p95 latency | 2000 ms | Generous local-demo operational bound |

These are demonstration criteria, not clinically validated thresholds. The
reserved test set never enters tuning. Healthy and corrupted scenarios replay its
labels; repeated rows are explicitly not independent production observations.
Corruption independently permutes input columns and preserves ground-truth labels
and the deployed model. Synthetic inputs establish robustness, not clinical
accuracy. Numeric/categorical drift alone raises investigation, not retraining.
Infrastructure errors require operational remediation. Longitudinal production
use would need new labels, monitoring windows and validated decision criteria.

Windows writes `retraining/request.json` and a readable request. After a successful
trusted GitHub push/manual run, Actions creates or updates one clearly marked
DEMO issue. Pull requests cannot publish images/issues. The built-in workflow
token needs `issues: write` and GHCR publication needs `packages: write`; no
personal token is required for this repository's own automation.

## Output and cleanup

`reports/acceptance/<run>/START_HERE.html` links the detailed reports, coverage,
statistics, screenshots and videos. The matching `*-evidence.zip` is suitable for
transfer/diagnosis. It excludes virtual environments and the large private working
cache, while preserving portable data/model reports and prediction evidence.
Live SQLite databases and model artifacts stay in the run workspace for the demo.
The evidence also retains the verified model bundle, selected release, tracking
database snapshot and tracking artifacts. SQLite artifact URIs retain their
original workspace; moving that snapshot needs URI relocation or a fresh run.
`services.json` records the actual URLs; ports for native API and MLflow are
allocated dynamically. The deployed LoadBalancer uses port 8080.

`COLLECT_DIAGNOSTICS.bat` gathers fresh logs without redeployment.
It can also archive an early bootstrap failure using only Python's standard library.
`STOP_DEMO.bat` stops only the latest run's owned services and Minikube profile;
it does not delete evidence. Copy the final ZIP/video before leaving the PC.

Coverage reports explicitly scope instrumented native application/acceptance
Python code; container and isolated-wheel/API checks remain black-box. There is
no fabricated code coverage for uninstrumented processes, and requirement coverage
is not an automatic grade. Statistical evidence includes sample sizes, missing
values, class balance, CV means/SDs, test metrics/intervals, latency percentiles,
throughput and model/data/source/image identities.

The submission PDF is exactly ten pages and is separate from the unrestricted
detailed test report. Unsupplied student/repository details remain marked so they
can be corrected in configuration before the submission run.
