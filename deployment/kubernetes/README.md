# Local Kubernetes deployment

For automated Windows execution, use the
[one-click launcher and offline instructions](../windows/START_HERE.md).
The steps below remain available for manual execution and troubleshooting.

The manifests run the approved prediction image on **Minikube with the Docker
driver**, exposed through a **LoadBalancer Service on port 8080**. This satisfies
the assignment's Ingress-or-LoadBalancer design requirement. Actual deployment,
endpoint checks and screenshots still need a PC with working virtualization.

The instructions below run from the repository root in PowerShell on that PC.
They do not require a registry, cloud account, training run or MLflow server.
Use the same source revision, `reports/modeling/latest.json`, selected release
and retained `models/bundles/` ZIP together. Never copy only the YAML files.

## 1. Prepare the other PC

Install [Python 3.13](https://www.python.org/downloads/),
[Docker Desktop](https://docs.docker.com/desktop/),
[kubectl](https://kubernetes.io/docs/tasks/tools/) and
[Minikube](https://minikube.sigs.k8s.io/docs/start/).
Enable virtualization and start Docker Desktop with a Linux engine.
Use a kubectl version within one minor version of the cluster. Allow at least
two CPUs and 4 GB of memory for this cluster; image builds need additional disk
space. Internet is needed for dependencies, the base image and cluster images.
Docker Desktop's own Kubernetes feature is unnecessary for this Minikube path.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
docker info --format '{{.OSType}}'
minikube version
kubectl version --client
minikube start -p heart-mlops --driver=docker --cpus=2 --memory=4096
kubectl --context heart-mlops get nodes
```

`docker info` must print `linux` and the node must become `Ready`.
Every kubectl command below specifies the dedicated `heart-mlops` context.
Do not substitute a shared or production context.

## 2. Verify the approved bundle and build/test the image

```powershell
$prepared = .\.venv\Scripts\python.exe scripts/prepare_docker.py
if ($LASTEXITCODE -ne 0) { throw 'Approved bundle verification failed' }
$dockerContext = ($prepared | ConvertFrom-Json).context
$contextManifest = Get-Content -Raw (Join-Path $dockerContext 'context.json') | ConvertFrom-Json
$bundleSha = $contextManifest.identity.bundle_sha256
$deploymentYaml = Get-Content -Raw deployment/kubernetes/deployment.yaml
$imageMatches = [regex]::Matches($deploymentYaml, '(?m)^\s*image:\s*(\S+)\s*$')
if ($imageMatches.Count -ne 1) { throw 'Expected exactly one API image in the manifest' }
$image = $imageMatches[0].Groups[1].Value
if ($image -notmatch "^heart-disease:bundle-$bundleSha(-[a-z0-9][a-z0-9.-]*)?$") {
    throw 'Deployment image does not match the approved bundle; follow the release update section'
}
.\.venv\Scripts\python.exe scripts/docker_smoke.py --image $image --output reports/docker/kubernetes-image
if ($LASTEXITCODE -ne 0) { throw 'Docker build or container verification failed' }
docker image inspect $image --format '{{.Id}}'
minikube -p heart-mlops image load $image
if ($LASTEXITCODE -ne 0) { throw 'Loading image into Minikube failed' }
minikube -p heart-mlops image ls
```

The smoke command builds the verified context, runs a real container, checks all
seven packaged predictions plus invalid input, health, metrics and dashboard,
and removes its temporary container. It retains the image and evidence. Choose
a new output directory when repeating it; existing evidence is preserved.
The first command also provides `$dockerContext` for later HTTP verification.
The repository root is not a Docker build context.

The full bundle SHA identifies the intended package, not a reproducible Docker
image digest: base images and OS packages can change. Record the built image ID
and deployed pod's `imageID` below. `imagePullPolicy: Never` requires the local
image to be loaded and prevents a remote image with the same name being pulled.

## 3. Validate and apply

Create the dedicated namespace first so server validation can resolve it.

```powershell
kubectl --context heart-mlops apply -f deployment/kubernetes/namespace.yaml
if ($LASTEXITCODE -ne 0) { throw 'Namespace creation failed' }
kubectl --context heart-mlops apply --dry-run=server --validate=strict -f deployment/kubernetes/deployment.yaml -f deployment/kubernetes/service.yaml
if ($LASTEXITCODE -ne 0) { throw 'Kubernetes server validation failed' }
kubectl --context heart-mlops apply -f deployment/kubernetes/deployment.yaml -f deployment/kubernetes/service.yaml
if ($LASTEXITCODE -ne 0) { throw 'Kubernetes apply failed' }
kubectl --context heart-mlops -n heart-disease rollout status deployment/heart-api --timeout=300s
if ($LASTEXITCODE -ne 0) { throw 'Deployment did not become ready; inspect events and logs' }
kubectl --context heart-mlops -n heart-disease get pods,svc -o wide
kubectl --context heart-mlops -n heart-disease get endpointslices -l kubernetes.io/service-name=heart-api
```

Expect one `1/1 Running` pod and a Service with type `LoadBalancer`. Its external
IP may remain `<pending>` until the tunnel starts. A startup probe gives model
loading about 120 seconds; readiness routes traffic only after `/ready` succeeds.
Liveness checks `/health` separately. Kubernetes uses these probes rather than
the Dockerfile's `HEALTHCHECK`.

The container runs as UID/GID 10001, drops capabilities, has a read-only root
filesystem and a 64 MiB memory-backed `/tmp`. Requests are 100m CPU/256 MiB;
limits are one CPU/512 MiB. These are starting values that must be observed on
the actual cluster. The fitted pipeline is inside the image; no host model/data
mount or persistent volume is required. Logs go to stdout/stderr.

## 4. Expose through the LoadBalancer

In a **second PowerShell terminal**, keep this command running:

```powershell
minikube -p heart-mlops tunnel
```

Minikube may request administrator permission to configure the host route.
Back in the first terminal, wait for an external address, then derive the URL:

```powershell
kubectl --context heart-mlops -n heart-disease get service heart-api --watch
# Press Ctrl+C once EXTERNAL-IP has an address; leave the tunnel terminal open.
$service = kubectl --context heart-mlops -n heart-disease get service heart-api -o json | ConvertFrom-Json
$address = $service.status.loadBalancer.ingress[0].ip
if (-not $address) { $address = $service.status.loadBalancer.ingress[0].hostname }
if (-not $address) { throw 'LoadBalancer address is pending; check minikube tunnel' }
if ($address.Contains(':')) { $address = "[$address]" }
$baseUrl = "http://${address}:8080"
Invoke-RestMethod "$baseUrl/ready"
```

Use the Service's reported address rather than assuming the Minikube node IP is
reachable from Windows. The URL is accessible from this cluster host while the
tunnel is open; it is not a public URL and is not a remote evaluator endpoint.
An evaluator reproduces these steps locally or watches the recorded local demo.
Port forwarding can help debugging but does not replace this LoadBalancer proof.

## 5. Verify predictions, model identity, monitoring and logs

```powershell
$evidence = 'reports/kubernetes/first-deployment'
New-Item -ItemType Directory -Path $evidence -ErrorAction Stop | Out-Null
kubectl --context heart-mlops version -o json | Set-Content -Encoding utf8 "$evidence/versions.json"
minikube -p heart-mlops status | Set-Content -Encoding utf8 "$evidence/minikube-status.txt"
docker image inspect $image | Set-Content -Encoding utf8 "$evidence/built-image.json"
kubectl --context heart-mlops -n heart-disease get deployment,service,pods,endpointslices -o json | Set-Content -Encoding utf8 "$evidence/resources.json"
kubectl --context heart-mlops -n heart-disease get pods -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'
.\.venv\Scripts\python.exe scripts/smoke_api.py --url $baseUrl --bundle (Join-Path $dockerContext 'bundle') --output "$evidence/http-validation.json"
if ($LASTEXITCODE -ne 0) { throw 'Deployed HTTP validation failed' }
$body = Get-Content -Raw docs/sample_request.json
Invoke-RestMethod -Method Post -Uri "$baseUrl/predict" -ContentType application/json -Body $body | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 "$evidence/sample-prediction.json"
Invoke-RestMethod "$baseUrl/metrics" | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 "$evidence/metrics.json"
kubectl --context heart-mlops -n heart-disease logs deployment/heart-api --timestamps=true | Set-Content -Encoding utf8 "$evidence/api.log"
kubectl --context heart-mlops -n heart-disease describe deployment heart-api | Set-Content -Encoding utf8 "$evidence/deployment.txt"
kubectl --context heart-mlops -n heart-disease get events --sort-by=.metadata.creationTimestamp | Set-Content -Encoding utf8 "$evidence/events.txt"
```

The HTTP check compares seven raw-input cases against the exact packaged expected
classes and probabilities, and checks loaded run identity, 422 rejection, OpenAPI,
request IDs, metrics and dashboard. Its JSON must say `status: passed`. The expected
run ID is in `$dockerContext/bundle/fixtures/expected_response.json`; do not compare
against a hardcoded prediction. `resources.json` records the pod's image ID as well
as its readiness and restart count. Retain failed evidence and use a different
`$evidence` directory for each attempt.

Open `$baseUrl/docs` and `$baseUrl/monitoring` in a browser. Capture screenshots of:

- `rollout status`, ready pod, LoadBalancer external address and endpoints.
- A successful `/predict` response showing `model_run_id` and confidence.
- The monitoring dashboard after sending requests, and matching request logs.
- The image ID, deployed `imageID`, HTTP verification result and cluster versions.

Save real screenshots in `screenshots/` and record these steps for the submission
video. Commit the evidence used in the report, excluding virtual environments and
build contexts. A passing native API check or static YAML validation alone does
not establish that Kubernetes deployment succeeded.

One replica is intentional: metrics are per process and reset on restart. During
a rolling update two pods can briefly coexist, so verify HTTP metrics after the
rollout has completed. Extra replicas require metrics aggregation for a consistent
service-wide dashboard. `/metrics` supplies JSON rather than Prometheus format.

## Update or roll back a release

For a newly approved release, run `prepare_docker.py` again, derive its bundle SHA,
and change the single `image:` field in `deployment.yaml` to
`heart-disease:bundle-<FULL_APPROVED_ZIP_SHA256>`. Commit the manifest with the
matching source, release, pointer and retained ZIP. Repeat build/smoke, image load,
server dry-run, apply, rollout and all HTTP checks; capture the new image IDs.
Do not reuse the old tag for changed package contents. For a new build of the same
bundle, append a new suffix such as `-build2` to its manifest image tag before
building/loading. The commands above read that tag and verify its bundle SHA;
applying the new tag triggers a rollout even if the package itself is unchanged.

To revert a failed update while its previous image is still loaded:

```powershell
kubectl --context heart-mlops -n heart-disease rollout undo deployment/heart-api
kubectl --context heart-mlops -n heart-disease rollout status deployment/heart-api --timeout=300s
```

Restore the previous manifest and matching bundle/release revision as well before
applying again. HTTP checks must use the restored image's bundle fixtures.

## Troubleshooting and stopping

- `ErrImageNeverPull`: compare the manifest image tag with `minikube image ls`;
  load the exact image into profile `heart-mlops` before applying.
- `CrashLoopBackOff` or probe failures: use `kubectl --context heart-mlops -n
  heart-disease describe pod POD_NAME` and `kubectl --context heart-mlops -n
  heart-disease logs POD_NAME --previous`. Bundle verification/startup failures
  must be fixed rather than bypassed. `OOMKilled` requires checking memory usage
  and adjusting limits with evidence.
- Pod `Pending`: inspect events for unavailable CPU/memory or image platform
  mismatch. The image architecture must match the cluster node architecture.
- External IP pending/unreachable: inspect the separate tunnel terminal and host
  firewall, verify the profile and port 8080, and check ready EndpointSlices.
- To debug only: `kubectl --context heart-mlops -n heart-disease port-forward
  service/heart-api 18080:8080`, then use `http://127.0.0.1:18080`. Keep the separate
  LoadBalancer verification for the submission.

After saving evidence, stop the tunnel with Ctrl+C and stop this dedicated cluster:

```powershell
minikube stop -p heart-mlops
```

To remove only this assignment's resources, the following command deletes the
namespace and everything in it. Run it only after preserving logs and screenshots:

```powershell
kubectl --context heart-mlops delete namespace heart-disease
```

Delete resources while the cluster is running, before stopping Minikube.

References: [Minikube image loading](https://minikube.sigs.k8s.io/docs/handbook/pushing/),
[LoadBalancer access and tunnel](https://minikube.sigs.k8s.io/docs/handbook/accessing/),
[Kubernetes Services](https://kubernetes.io/docs/concepts/services-networking/service/),
and [startup, readiness and liveness probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).
