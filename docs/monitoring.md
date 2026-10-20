# Monitoring and request logging

This implements the assignment's **API metrics and simple dashboard** option.
It runs inside the prediction API without extra dependencies or Docker.

## Run and inspect

From the project root, start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe -m heart_disease.api
```

Open http://127.0.0.1:8000/monitoring. It refreshes every three seconds and can
be paused. The dashboard includes request totals, mean response time, 4xx/5xx
counts, predicted records, five-minute traffic history, endpoint/status counts,
model identity, uptime, active requests, and mean inference time. All assets are
included in the application wheel; no CDN or external service is needed.

Send requests from another terminal and watch the counters change:

```powershell
$body = Get-Content -Raw docs/sample_request.json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
# Deliberately rejected request: expect HTTP 422 and an increased 4xx counter.
Invoke-WebRequest http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body '{"instances":[]}'
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

The last endpoint returns **JSON**, not Prometheus exposition format. It is
documented in `/docs`. Its response has `Cache-Control: no-store`.

## Metric definitions

| Field | Meaning |
|---|---|
| `requests.total` | Completed counted HTTP requests, including rejections and failures |
| `requests.in_flight` | Counted requests currently being handled |
| `requests.client_errors` / `server_errors` | HTTP 4xx / 5xx responses respectively |
| `requests.duration_seconds_*` | Sum, arithmetic mean, and maximum elapsed time through response headers |
| `requests.by_route` | Counts by HTTP method, route template, and status |
| `predictions.successful_batches` | Scored batches whose output passed response validation |
| `predictions.failed_batches` | Exceptions during scoring or output validation; excludes input rejection and model-not-ready responses |
| `predictions.records` / `by_class` | Successful output records and aggregate class 0/1 counts |
| `predictions.inference_seconds_*` | Scoring plus output-validation time, summed or averaged per successful batch |
| `recent.buckets` | Up to 60 five-second intervals, including zeros during idle periods; current interval is partial |
| `model` | Whether a model is loaded and its training run ID |

`/health`, `/ready`, `/metrics`, and `/monitoring` are excluded from HTTP counters,
including unsuccessful calls to those paths. Documentation and unknown-route
requests are counted. Unknown paths collapse to `unmatched` and unfamiliar
methods to `OTHER`, so user-supplied URLs do not create unlimited labels.
Updates and snapshots use a lock because scoring runs in worker threads.

These are **per-process, in-memory metrics**. Every startup resets counters;
the snapshot includes process ID and start time to identify resets. Traffic
history expires after about five minutes; cumulative totals last until restart.
The supported serving command uses one Uvicorn worker. With multiple Kubernetes
replicas, each pod has separate values; a load-balanced dashboard would switch
between pods. Use one replica for the assignment demo or forward a specific pod.
This implementation does not aggregate replicas or retain historical metrics.

Class counts describe output activity, not model quality. Online accuracy needs
ground-truth labels, and drift monitoring needs a defined reference distribution;
neither is claimed by this dashboard. Model evaluation remains in the training
reports and MLflow.

## Logs

Application events are JSON lines on stdout, compatible with container log
collection. Lifecycle events record model load, unload, and load failure.
Every HTTP request, including probes and monitoring reads, records:

```json
{"event":"http_request","timestamp":"2026-10-17T12:00:00+00:00","level":"INFO","request_id":"...","method":"POST","route":"/predict","status":200,"duration_ms":4.2,"model_run_id":"..."}
```

`X-Request-ID` in the response correlates with the log. Successful requests use
INFO, 4xx use WARNING, and 5xx use ERROR. Unexpected errors include only the
exception type. A cancelled HTTP handler is recorded as 499 and releases the
in-flight count. Payloads, query strings, headers, raw unknown paths, individual
predictions, and exception messages are excluded. Uvicorn's separate startup and
shutdown messages may also appear as ordinary text.

To retain a local demonstration log, first create the output directory and then
start the API with terminal output captured:

```powershell
New-Item -ItemType Directory -Force reports/monitoring
.\.venv\Scripts\python.exe -m heart_disease.api 2>&1 | Tee-Object -FilePath reports/monitoring/api.log
```

In Docker, inspect `docker logs heart-api`; after Kubernetes deployment, use
`kubectl logs POD_NAME`. Log rotation/retention belongs to the runtime or collector;
the API does not write files inside the container.

## Verification and submission evidence

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_monitoring.py
```

Tests exercise successful batches, 422 rejection, injected 500 failures, invalid
model outputs, concurrent updates, idle expiry, per-app reset, bounded route labels,
and privacy of request logs. `scripts/verify_api.py` additionally starts an API
in an empty workspace and verifies real HTTP, JSON metric deltas, dashboard HTML,
and request-ID correlation in captured JSON logs. It is part of `scripts/ci.py`.
The Docker smoke check uses the same HTTP metrics/dashboard verification.

Local evidence from 17 October 2026 is in
[`reports/monitoring_validation_20261017.json`](../reports/monitoring_validation_20261017.json).
The full suite passed 108 tests; after correcting documentation-route labels,
all 36 API/monitoring checks passed again. The final isolated wheel verified
seven prediction fixtures, metrics deltas, dashboard HTML delivery, and ten
structured request-log events. Browser visual inspection was unavailable because
the in-app automation runtime failed to initialize; screenshots remain to capture.

For submission, capture the dashboard after accepted and rejected predictions,
the JSON metrics, and a correlated request-log event. Local native verification
does not replace the separate Docker/Kubernetes execution evidence from the
friend's PC.

Implementation references: [FastAPI HTML responses](https://fastapi.tiangolo.com/advanced/custom-response/#html-response)
and [Kubernetes logging architecture](https://kubernetes.io/docs/concepts/cluster-administration/logging/).
