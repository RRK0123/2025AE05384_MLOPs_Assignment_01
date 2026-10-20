# Docker and Kubernetes without ChatGPT

Use this guide on your friend's **Windows PC**, with internet access. The project
contains all serving code, Dockerfile, Kubernetes manifests and automation. After
the one-time prerequisites below, double-click **RUN_DOCKER_KUBERNETES.bat** in
the extracted project's root. Do not run it from inside the ZIP viewer.

## Before leaving your own PC

Create a small transfer ZIP with the current approved model:

```powershell
.\.venv\Scripts\python.exe scripts/create_deployment_handoff.py
```

Copy the printed `dist/windows-deployment-handoff-....zip` to a USB drive. It has
the application, exact approved bundle/release, scripts and instructions. It
excludes this PC's virtual environments, training data, MLflow database, caches
and old results. This is a **serving-only transfer**, not the full assignment
submission. Never copy `.venv` or `.deployment-venv` between computers.

Extract the ZIP into a local writable folder such as `C:\MLOpsDemo`. Keep its
directory structure intact. Avoid a network drive, a read-only USB folder or
running from inside another Python environment. Open this guide there before
starting; ChatGPT is unnecessary during execution.

After a successful run, follow [CAPTURE_BEFORE_LEAVING.md](CAPTURE_BEFORE_LEAVING.md)
to save all deployment evidence before losing access to this PC. The transfer
ZIP also places copies of both guides at its top level for quick access.

## One-time prerequisites on the other PC

1. Install [Python 3.13](https://www.python.org/downloads/windows/), including the
   Python launcher (`py`) or the option to add Python to PATH.
2. Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/).
   Enable virtualization/WSL 2 as its installer requests, restart if needed,
   open Docker Desktop and wait for the **Linux engine** to be running.
3. Install [Minikube](https://minikube.sigs.k8s.io/docs/start/) for Windows and
   [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/).
   Both commands must be on PATH. The simplest documented Minikube installer or
   Windows package-manager method is sufficient. Match kubectl to the cluster
   within one minor version. Docker Desktop's own Kubernetes need not be enabled.
4. Have internet access, two available CPUs, at least 4 GB allocated for Minikube,
   additional host memory for Windows/Docker, and several GB of free disk space.
   Installation and tunnel routing can require administrator permission.

Open a **new** PowerShell window after installing and check:

```powershell
py -3.13 --version
docker info --format '{{.OSType}}'
minikube version
kubectl version --client
```

`docker info` must say `linux`. Fix failed prerequisites before running the batch.
The batch reports missing tools; it cannot repair BIOS/virtualization, install
system software, accept Docker licensing dialogs or reboot Windows for you.

## One-click run

Double-click `RUN_DOCKER_KUBERNETES.bat`. It automatically:

1. Creates a private `.deployment-venv` and installs pinned serving/build packages.
2. Verifies the approved bundle, model identity and application bytes.
3. Builds a uniquely tagged image; tests seven predictions, invalid requests,
   health, monitoring and the actual non-root container.
4. Starts the dedicated Minikube profile **heart-mlops**, loads the image, runs
   strict Kubernetes server validation and deploys the manifests.
5. Opens a second window for the LoadBalancer tunnel. **Keep it open.** It may
   request routing permission. The Service uses port 8080 and the tunnel binds
   to loopback on Windows where supported.
6. Waits for the LoadBalancer URL, checks the exact deployed predictions and
   model identity, verifies readiness/image identity and correlates request logs.
7. Opens Swagger `/docs` and the `/monitoring` dashboard in your default browser.
8. Saves a readable result, per-stage reports/logs and an uploadable diagnostics ZIP.

The first run may take 15–40 minutes depending on downloads and hardware. The
console prints each stage and its log path; while a command is running it prints
an elapsed-time message every ten seconds. Logs update on disk. No stage is
silently skipped after a failure. Each main run keeps separate evidence and a
new image tag; generated manifests are stored with that run rather than changing
the committed templates. The profile name is fixed: reserve `heart-mlops` for
this assignment. Rerunning updates this application's deployment in that profile.

No training or MLflow server is needed. The batch leaves the cluster running for
your screenshots and video. A successful Docker test removes its temporary test
container; Kubernetes then serves the same tested image.

## Find the result and upload logs

The batch prints an exact path similar to:

```text
reports\deployment\20261017T150000000Z-ab12cd\RESULTS.txt
reports\deployment\20261017T150000000Z-ab12cd-diagnostics.zip
```

Read `RESULTS.txt`: it states Docker and Kubernetes verification separately,
the failed stage/reason if any, and the browser URLs if available. For a setup
failure before Python starts, read `launcher-failure.json` and `bootstrap.log`.
**Upload the `-diagnostics.zip` to ChatGPT** when you regain access. Say what you
clicked, whether Docker Desktop was running, and what appeared in the tunnel
window. You do not need to locate individual logs.

The ZIP includes the setup transcript, stage commands/exit codes, Docker build
and container logs, generated manifests, bundle/source hashes, expected and
observed predictions, pod/image details, events, current/previous API logs,
metrics and the tunnel log when present. Failed diagnostic reads are recorded
as failed steps rather than being presented as successful deployment evidence.
It does not intentionally collect kubeconfig, credentials, Secrets or the whole
repository. Share only this generated report ZIP, not your `.kube` directory.

After interacting with the demo, double-click **COLLECT_DEPLOYMENT_LOGS.bat**.
It creates another ZIP containing the original run evidence plus fresh pod state,
events, API logs and metrics. It does **not** build, deploy, stop or delete anything.
Its result is `COLLECTED`, not a new deployment pass. Use this newer ZIP for diagnosis.

## Screenshots and short video

- In `/docs`, expand `POST /predict`, click **Try it out**, and paste the JSON
  from `docs/sample_request.json`. Execute and capture the response with the
  loaded run ID, prediction and confidence.
- Open `/monitoring`, send a few requests, and capture its counters/charts.
- In PowerShell, from the project folder, record:

```powershell
kubectl --context heart-mlops -n heart-disease get pods,service -o wide
kubectl --context heart-mlops -n heart-disease rollout status deployment/heart-api
kubectl --context heart-mlops -n heart-disease get endpointslices
kubectl --context heart-mlops -n heart-disease logs deployment/heart-api --tail=20
```

Capture the ready pod, LoadBalancer external address, endpoints and request logs.
Record the batch's final pass and browser demonstrations in the submission video.
Save screenshots in `screenshots/` (create it if using the small handoff ZIP).
The URLs work locally on this PC while Minikube and the tunnel are running; they
are not public URLs for remote access.

## If something fails

| Failed stage / message | What to do without ChatGPT |
|---|---|
| Missing command / Python 3.13 | Install the prerequisite above, then open a new terminal or rerun the batch. |
| Docker connection / Windows engine | Open Docker Desktop, wait for startup, switch to Linux containers, check `docker info`. |
| Virtualization / WSL error | This PC needs working virtualization; follow Docker's setup checks and restart if requested. |
| pip / download / certificate / proxy | Check internet/proxy settings. Preserve the ZIP; do not disable certificate checks or change pinned versions. |
| Approved bundle / source hash mismatch | Re-extract a fresh transfer ZIP. Keep the exact source, retained bundle and selected release together. |
| Docker build/test | Read `docker/validation.json`, `docker/build.log` and container logs in the run folder; upload the ZIP if unresolved. |
| Minikube memory/start | Free RAM/disk space; inspect the `cluster-start` log. This path reserves 4 GB for Minikube. |
| kubectl/client version mismatch | Read the client/server versions in the `cluster-versions` log. Install kubectl with the same minor version as that server using the official Windows instructions linked above, then rerun. |
| Image load / ErrImageNeverPull | Check the `image-load` log. Rerun the main batch; it loads the exact new tag before deploying. |
| Rollout / OOM / CrashLoop | Inspect pod details, events and API logs in the ZIP. Preserve the failure before retrying. |
| Tunnel route permission / external IP pending | Read the tunnel window. Close it, then right-click the main batch and **Run as administrator** under the same Windows account; approve UAC and rerun. |
| Port 8080 already in use / tunnel error | Close an older assignment tunnel with Ctrl+C. Stop any other local app using that port, then rerun. |
| Browser does not open | Copy the URLs from `RESULTS.txt` into the browser. |

Do not change model files, skip verification or replace LoadBalancer proof with a
port-forward success. A failed run is useful diagnostic evidence. The manual
alternative is in `deployment/kubernetes/README.md`.

## Finish and clean up

1. Save screenshots/video and run `COLLECT_DEPLOYMENT_LOGS.bat`.
2. Copy the resulting diagnostics ZIPs and screenshots back to your USB drive.
3. Press Ctrl+C in the tunnel window, then close it.
4. Stop only this assignment's cluster:

```powershell
minikube stop -p heart-mlops
```

The launcher does not automatically delete resources or uninstall anything from
your friend's PC. Removing the dedicated profile later (`minikube delete -p
heart-mlops`) destroys that local cluster; do it only after preserving evidence.
