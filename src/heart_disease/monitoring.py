"""Bounded, thread-safe operational metrics for one serving process."""

import os
import time
from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock

EXCLUDED_ROUTES = frozenset({"/metrics", "/monitoring", "/health", "/ready"})
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
BUCKET_SECONDS = 5
BUCKET_COUNT = 60


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ServiceMetrics:
    """No request identifiers, records, arbitrary paths, or unbounded histories."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._started = clock()
        self.started_at_utc = utc_now()
        self._lock = Lock()
        self._in_flight = 0
        self._requests = self._client_errors = self._server_errors = 0
        self._duration = self._max_duration = 0.0
        self._routes = Counter()
        self._history = deque(maxlen=BUCKET_COUNT)
        self._predictions = Counter({"0": 0, "1": 0})
        self._batches = self._prediction_errors = 0
        self._prediction_duration = 0.0

    def _bucket(self, now):
        index = int((now - self._started) // BUCKET_SECONDS)
        # Bounded work even after a long idle period or suspended laptop.
        previous = self._history[-1]["index"] if self._history else -1
        for number in range(max(previous + 1, index - BUCKET_COUNT + 1), index + 1):
            self._history.append(
                {
                    "index": number,
                    "requests": 0,
                    "client_errors": 0,
                    "server_errors": 0,
                    "duration_seconds": 0.0,
                }
            )
        return self._history[-1]

    def begin_request(self):
        with self._lock:
            self._in_flight += 1

    def finish_request(self, method, route, status, duration):
        method = method if method in METHODS else "OTHER"
        with self._lock:
            self._in_flight -= 1
            self._requests += 1
            self._client_errors += 400 <= status < 500
            self._server_errors += status >= 500
            self._duration += duration
            self._max_duration = max(self._max_duration, duration)
            self._routes[method, route, status] += 1
            bucket = self._bucket(self._clock())
            bucket["requests"] += 1
            bucket["client_errors"] += 400 <= status < 500
            bucket["server_errors"] += status >= 500
            bucket["duration_seconds"] += duration

    def record_predictions(self, classes, duration):
        with self._lock:
            self._batches += 1
            self._prediction_duration += duration
            self._predictions.update(str(value) for value in classes)

    def prediction_failed(self):
        with self._lock:
            self._prediction_errors += 1

    def snapshot(self, model_run_id):
        with self._lock:
            now = self._clock()
            self._bucket(now)
            history = [
                {
                    "start_seconds": row["index"] * BUCKET_SECONDS,
                    **{key: value for key, value in row.items() if key != "index"},
                }
                for row in self._history
            ]
            return {
                "schema_version": 1,
                "scope": "single_process_since_startup",
                "process_id": os.getpid(),
                "started_at_utc": self.started_at_utc,
                "observed_at_utc": utc_now(),
                "uptime_seconds": now - self._started,
                "model": {"ready": model_run_id is not None, "run_id": model_run_id},
                "requests": {
                    "total": self._requests,
                    "in_flight": self._in_flight,
                    "client_errors": self._client_errors,
                    "server_errors": self._server_errors,
                    "duration_seconds_sum": self._duration,
                    "duration_seconds_mean": self._duration / self._requests
                    if self._requests
                    else 0,
                    "duration_seconds_max": self._max_duration,
                    "by_route": [
                        {"method": method, "route": route, "status": status, "count": count}
                        for (method, route, status), count in sorted(self._routes.items())
                    ],
                },
                "predictions": {
                    "successful_batches": self._batches,
                    "failed_batches": self._prediction_errors,
                    "records": sum(self._predictions.values()),
                    "by_class": dict(self._predictions),
                    "inference_seconds_sum": self._prediction_duration,
                    "inference_seconds_mean": self._prediction_duration / self._batches
                    if self._batches
                    else 0,
                },
                "recent": {
                    "bucket_seconds": BUCKET_SECONDS,
                    "maximum_buckets": BUCKET_COUNT,
                    "buckets": history,
                },
                "excluded_routes": sorted(EXCLUDED_ROUTES),
            }
