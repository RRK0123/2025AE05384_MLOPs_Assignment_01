"""Check a running API over real HTTP against packaged prediction fixtures."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url, payload=None):
    body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.load(response), dict(response.headers)
    except HTTPError as error:
        return error.code, json.load(error), dict(error.headers)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wait_ready(base_url, timeout=60, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"API process exited during startup: {process.returncode}")
        try:
            if request_json(base_url + "/ready")[0] == 200:
                return
        except (URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError("API did not become ready")


def verify_service(base_url, bundle):
    bundle = Path(bundle)
    payload = json.loads((bundle / "fixtures/request.json").read_text())
    expected = json.loads((bundle / "fixtures/expected_response.json").read_text())
    status, before, _ = request_json(base_url + "/metrics")
    require(status == 200 and before["schema_version"] == 1, "JSON metrics are absent")
    status, health, _ = request_json(base_url + "/health")
    require(status == 200 and health["status"] == "alive", "Liveness check failed")
    status, ready, _ = request_json(base_url + "/ready")
    require(
        status == 200 and ready["model_run_id"] == expected["model_run_id"],
        "Readiness model identity differs",
    )
    status, actual, headers = request_json(
        base_url + "/predict?private=monitoring-validation-secret", payload
    )
    require(status == 200, "Prediction request failed")
    require(actual["model_run_id"] == expected["model_run_id"], "Prediction model differs")
    require(len(actual["predictions"]) == len(expected["predictions"]), "Batch size differs")
    for observed, reference in zip(actual["predictions"], expected["predictions"]):
        require(observed["prediction"] == reference["prediction"], "Class differs from fixture")
        for field in ["probability_disease", "confidence"]:
            require(
                math.isclose(observed[field], reference[field], rel_tol=1e-12, abs_tol=1e-12),
                "Probability differs from fixture",
            )
    require(any(key.lower() == "x-request-id" for key in headers), "Request ID is absent")
    require(
        request_json(base_url + "/predict", {"instances": []})[0] == 422,
        "Empty batch was not rejected",
    )
    invalid = {"instances": [{**payload["instances"][0], "age": True}]}
    require(
        request_json(base_url + "/predict", invalid)[0] == 422, "Boolean input was not rejected"
    )
    status, schema, _ = request_json(base_url + "/openapi.json")
    require(status == 200 and "/predict" in schema["paths"], "OpenAPI contract is absent")
    status, after, _ = request_json(base_url + "/metrics")
    require(status == 200 and after["model"]["ready"], "Metrics readiness is incorrect")
    require(after["model"]["run_id"] == actual["model_run_id"], "Metrics model differs")
    require(
        after["requests"]["total"] - before["requests"]["total"] >= 4,
        "Metrics did not count application requests",
    )
    require(
        after["requests"]["client_errors"] - before["requests"]["client_errors"] >= 2,
        "Metrics did not count rejected requests",
    )
    require(
        after["predictions"]["records"] - before["predictions"]["records"]
        >= len(actual["predictions"]),
        "Metrics did not count batch records",
    )
    for value in [0, 1]:
        observed = (
            after["predictions"]["by_class"][str(value)]
            - before["predictions"]["by_class"][str(value)]
        )
        require(
            observed >= sum(item["prediction"] == value for item in actual["predictions"]),
            "Metrics prediction class counts differ",
        )
    with urlopen(base_url + "/monitoring", timeout=10) as dashboard:
        require(
            dashboard.status == 200 and "API monitoring" in dashboard.read().decode(),
            "Monitoring dashboard is absent",
        )
    return {
        "status": "passed",
        "base_url": base_url,
        "model_run_id": actual["model_run_id"],
        "prediction_cases_verified": len(actual["predictions"]),
        "health_and_readiness_verified": True,
        "invalid_requests_rejected": True,
        "openapi_verified": True,
        "request_id_verified": True,
        "prediction_request_id": next(
            value for key, value in headers.items() if key.lower() == "x-request-id"
        ),
        "monitoring_metrics_verified": True,
        "monitoring_dashboard_verified": True,
        "metrics_snapshot": after,
        "predictions": actual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--bundle", type=Path, required=True, help="Extracted verified bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wait_ready(args.url)
    result = verify_service(args.url, args.bundle)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
