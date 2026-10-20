# Before leaving the other PC

This checklist covers the assignment's Docker, Kubernetes, monitoring and local
access evidence. Keep the Minikube tunnel window open until everything below is
saved. Use the URLs printed in your run's `RESULTS.txt`; do not assume an IP.

## Confirm the deployment passed

- [ ] The **main deployment** `RESULTS.txt` says `RESULT: PASSED`.
- [ ] Its `summary.json` has both `docker_verified: true` and
  `kubernetes_verified: true`.
- [ ] `docker/validation.json` says `status: passed`, verifies seven prediction
  cases, the Docker healthcheck and the non-root user, and records an image ID.
- [ ] `http-validation.json` says `status: passed`, verifies seven cases, and
  shows the same model run ID as the packaged expected responses.
- [ ] The deployed pod is ready; the Service is **LoadBalancer** with an external
  address; requests work through that address on port 8080.

A log collector result of `COLLECTED` preserves diagnostics; it does not mean
that the deployment passed. If a main run failed, retain its ZIP and the error.
Do not replace failed evidence with a screenshot from a different run.

## Save screenshots

Create a `screenshots` folder in the extracted project. Use Windows+Shift+S or
your normal screenshot tool, and save each capture with a descriptive name.

- [ ] **01-docker-proof.png:** show the main run's `docker/validation.json` with
  the image ID, passing checks and prediction cases. Also retain `docker/build.log`
  and `docker/container-logs.log` in the diagnostic ZIP.
- [ ] **02-kubernetes-ready.png:** show ready pods, the LoadBalancer external
  address and ready endpoints using the commands below.
- [ ] **03-kubernetes-predict.png:** open `/docs`, run `POST /predict` with the
  contents of `docs/sample_request.json`, and show HTTP 200, `model_run_id`,
  `prediction`, `probability_disease` and `confidence`. Include the browser URL.
- [ ] **04-monitoring.png:** open `/monitoring` after sending predictions and
  show model identity, request counts and prediction counts.
- [ ] **05-request-logs.png:** show successful and rejected request log entries
  with request IDs, status codes and duration. Logs omit feature records.
- [ ] **06-final-result.png:** show the main run's final `PASSED` result.

From the extracted project folder, use PowerShell for terminal screenshots:

```powershell
kubectl --context heart-mlops -n heart-disease get pods,service -o wide
kubectl --context heart-mlops -n heart-disease get endpointslices
kubectl --context heart-mlops -n heart-disease rollout status deployment/heart-api
kubectl --context heart-mlops -n heart-disease logs deployment/heart-api --tail=30
```

Docker's temporary smoke-test container is removed after verification. Its image,
inspection, prediction results and logs remain in the run evidence; Kubernetes
then runs that same uniquely tagged image. The Deployment pod has its own
container/image identity, recorded in `summary.json` and the pod details.

## Record the demonstration

- [ ] Record a short clip showing the Docker verification result, Kubernetes
  resources, a live prediction, monitoring updates and request logs.
- [ ] Explain that the saved preprocessing/classifier pipeline is inside the
  image and serving does not train or use MLflow/data at runtime.
- [ ] Explain that the LoadBalancer URL is local to this PC while the tunnel
  runs. Record the URL and the `heart-mlops` profile used for the demo.

The assignment asks for a video of the **overall pipeline**. This clip covers
deployment. Record or combine it with acquisition/EDA, training/model selection,
MLflow and the successful GitHub Actions run when those are available. The small
deployment kit intentionally omits the training workspace and GitHub workflow.

## Copy back everything needed

- [ ] Double-click **COLLECT_DEPLOYMENT_LOGS.bat** after the demo.
- [ ] Copy the main run's `*-diagnostics.zip` and the newer collector ZIP to your
  USB drive. The collector ZIP contains the original run and current state.
- [ ] Copy the screenshots and video; open them once from the USB drive to check
  they are readable. Screenshots/video are not automatically included in the
  diagnostic ZIP.
- [ ] Keep the exact transfer ZIP used, so its code/model can be matched to the
  recorded bundle SHA, source hashes and generated deployment manifests.
- [ ] Upload the diagnostic ZIP here when you return. If a step failed, also
  mention the visible error and whether the tunnel window remained open.

Only after preserving this evidence, press Ctrl+C in the tunnel window and stop
this dedicated cluster:

```powershell
minikube stop -p heart-mlops
```

The full GitHub repository, hosted CI screenshot/run URL, architecture diagram
and final 10-page report are separate submission tasks. The deployment kit and
these screenshots do not replace those deliverables.
