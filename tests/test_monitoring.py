"""Check aggregate correctness, thread safety, bounded history, and log privacy."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from test_packaging import saved_model

from heart_disease import api
from heart_disease.monitoring import ServiceMetrics


@pytest.fixture
def service(tmp_path):
    features, _ = saved_model(tmp_path)
    app = api.create_app(tmp_path)
    with TestClient(app) as client:
        yield client, app, {"instances": [features.iloc[1].to_dict()]}


def test_operational_metrics_reconcile_success_rejection_failure_and_class_counts(
    service, monkeypatch
):
    client, app, payload = service
    response = client.post("/predict", json={"instances": payload["instances"] * 3})
    assert response.status_code == 200
    classes = [item["prediction"] for item in response.json()["predictions"]]
    assert client.post("/predict", json={"instances": []}).status_code == 422

    def fail(*args):
        raise RuntimeError("private feature detail")

    monkeypatch.setattr(app.state.predictor, "predict", fail)
    assert client.post("/predict", json=payload).status_code == 500
    result = client.get("/metrics")
    assert result.headers["cache-control"] == "no-store"
    data = result.json()
    assert data["model"] == {"ready": True, "run_id": "test-model"}
    assert data["requests"]["total"] == 3
    assert data["requests"]["in_flight"] == 0
    assert data["requests"]["client_errors"] == data["requests"]["server_errors"] == 1
    assert sum(row["count"] for row in data["requests"]["by_route"]) == 3
    assert data["requests"]["duration_seconds_mean"] > 0
    assert data["predictions"]["records"] == 3
    assert data["predictions"]["successful_batches"] == data["predictions"]["failed_batches"] == 1
    assert data["predictions"]["by_class"] == {str(i): classes.count(i) for i in [0, 1]}
    assert sum(row["requests"] for row in data["recent"]["buckets"]) == 3
    assert "private feature detail" not in result.text


def test_scrapes_dashboard_and_probes_do_not_inflate_traffic(service):
    client, _, _ = service
    for _ in range(3):
        for route in ["/metrics", "/monitoring", "/health", "/ready"]:
            assert client.get(route).status_code == 200
    assert client.get("/metrics").json()["requests"]["total"] == 0
    dashboard = client.get("/monitoring")
    assert dashboard.headers["content-type"].startswith("text/html")
    assert "fetch('metrics'" in dashboard.text
    assert "<script src=" not in dashboard.text
    assert client.get("/openapi.json").json()["paths"]["/metrics"]["get"]
    assert client.get("/metrics").json()["requests"]["by_route"] == [
        {"method": "GET", "route": "/openapi.json", "status": 200, "count": 1}
    ]


def test_unknown_paths_collapse_and_private_inputs_are_absent(service, caplog):
    client, _, payload = service
    with caplog.at_level(logging.INFO, logger="heart_disease.api"):
        for index in range(20):
            client.get(f"/patient-secret-{index}?private=secret")
        response = client.post("/predict?private=secret", json=payload)
        client.post("/predict", json={"instances": []})
    metrics = client.get("/metrics").json()
    assert {row["route"] for row in metrics["requests"]["by_route"]} == {"unmatched", "/predict"}
    events = [json.loads(row.message) for row in caplog.records if row.name == "heart_disease.api"]
    assert events[-2]["request_id"] == response.headers["x-request-id"]
    assert events[-1]["level"] == "WARNING"
    for event in events:
        assert datetime.fromisoformat(event["timestamp"]).utcoffset().total_seconds() == 0
        assert event["duration_ms"] >= 0
    serialized = json.dumps(events) + json.dumps(metrics)
    for private in [
        "patient-secret",
        "private",
        "secret",
        "instances",
        "probability_disease",
        "chol",
    ]:
        assert private not in serialized


def test_invalid_model_outputs_are_errors_not_successful_predictions(service, monkeypatch, caplog):
    client, app, payload = service
    monkeypatch.setattr(app.state.predictor, "predict", lambda _: {"unexpected": "private"})
    with caplog.at_level(logging.ERROR, logger="heart_disease.api"):
        response = client.post("/predict", json=payload)
    assert response.status_code == 500
    event = json.loads(caplog.records[-1].message)
    assert event["level"] == "ERROR" and event["error_type"] == "ValidationError"
    assert "private" not in json.dumps(event)
    metrics = client.get("/metrics").json()
    assert metrics["predictions"]["records"] == 0
    assert metrics["predictions"]["failed_batches"] == 1


def test_metrics_reset_on_new_lifespan_and_are_isolated_between_apps(tmp_path):
    features, _ = saved_model(tmp_path)
    first = api.create_app(tmp_path)
    with TestClient(first) as client:
        client.post("/predict", json={"instances": [features.iloc[1].to_dict()]})
        assert client.get("/metrics").json()["requests"]["total"] == 1
    for app in [first, api.create_app(tmp_path)]:
        with TestClient(app) as client:
            assert client.get("/metrics").json()["requests"]["total"] == 0


def test_metrics_indicate_not_ready_without_lifespan():
    with_client = TestClient(api.create_app())
    assert with_client.get("/metrics").json()["model"] == {"ready": False, "run_id": None}


def test_concurrent_updates_are_not_lost():
    metrics = ServiceMetrics()

    def update(_):
        metrics.begin_request()
        metrics.record_predictions([0, 1], 0.01)
        metrics.finish_request("POST", "/predict", 200, 0.02)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(1000)))
    result = metrics.snapshot("run")
    assert result["requests"]["total"] == 1000
    assert result["requests"]["in_flight"] == 0
    assert result["predictions"]["by_class"] == {"0": 1000, "1": 1000}
    assert result["predictions"]["successful_batches"] == 1000


def test_history_expires_after_idle_without_resetting_cumulative_totals():
    clock = [0.0]
    metrics = ServiceMetrics(clock=lambda: clock[0])
    metrics.begin_request()
    metrics.finish_request("ARBITRARY", "unmatched", 404, 0.1)
    clock[0] = 10_000
    result = metrics.snapshot(None)
    assert len(result["recent"]["buckets"]) == 60
    assert sum(row["requests"] for row in result["recent"]["buckets"]) == 0
    assert result["requests"]["total"] == 1
    assert result["requests"]["by_route"][0]["method"] == "OTHER"
