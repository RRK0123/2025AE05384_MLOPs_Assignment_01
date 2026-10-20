"""Exercise the actual ASGI lifecycle, strict contract and saved-model predictions."""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from test_packaging import saved_model

from heart_disease import api
from heart_disease.predict import load_predictor
from heart_disease.schema import FEATURES


@pytest.fixture
def serving(tmp_path):
    features, _ = saved_model(tmp_path)
    payload = {"instances": [features.iloc[1].to_dict()]}
    app = api.create_app(tmp_path)
    with TestClient(app) as client:
        yield client, app, payload, tmp_path


def test_predictions_match_saved_pipeline_and_health_readiness(serving):
    client, app, payload, directory = serving
    assert client.get("/health").json() == {"status": "alive"}
    assert client.get("/ready").json() == {"status": "ready", "model_run_id": "test-model"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32
    assert response.json() == load_predictor(directory).predict(payload)
    assert app.state.predictor is not None


def test_model_loaded_once_per_lifespan(tmp_path, monkeypatch):
    features, _ = saved_model(tmp_path)
    calls = []
    original = api.load_predictor

    def counted(directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(api, "load_predictor", counted)
    app = api.create_app(tmp_path)
    with TestClient(app) as client:
        payload = {"instances": [features.iloc[1].to_dict()]}
        first = client.post("/predict", json=payload)
        assert client.post("/predict", json=payload).json() == first.json()
        assert calls == [tmp_path]
    assert app.state.predictor is None


def test_batch_and_null_values_use_saved_imputation(serving):
    client, _, payload, directory = serving
    payload["instances"].append(dict.fromkeys(FEATURES))
    payload["instances"][0]["chol"] = None
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    assert result.json() == load_predictor(directory).predict(payload)


def test_maximum_documented_batch_is_accepted(serving):
    client, _, payload, _ = serving
    payload["instances"] *= api.MAX_BATCH_SIZE
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    assert len(result.json()["predictions"]) == api.MAX_BATCH_SIZE


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_in_complete_records_are_rejected(serving, value):
    client, _, payload, _ = serving
    payload["instances"][0]["age"] = value
    result = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert result.status_code == 422
    assert result.json()["detail"]


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "too_many",
        "top_extra",
        "missing",
        "extra",
        "boolean",
        "string",
        "negative",
        "zero",
        "category",
        "huge_integer",
        "list_record",
    ],
)
def test_invalid_inputs_return_422(serving, problem):
    client, _, payload, _ = serving
    record = payload["instances"][0]
    if problem == "empty":
        payload["instances"] = []
    elif problem == "too_many":
        payload["instances"] *= api.MAX_BATCH_SIZE + 1
    elif problem == "top_extra":
        payload["target"] = 1
    elif problem == "missing":
        del record["age"]
    elif problem == "extra":
        record["target"] = 1
    elif problem == "category":
        record["thal"] = 99
    elif problem == "list_record":
        payload["instances"] = [[]]
    else:
        record["age"] = {
            "boolean": True,
            "string": "54",
            "negative": -1,
            "zero": 0,
            "huge_integer": 10**400,
        }[problem]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]
    assert all("input" not in item and "ctx" not in item for item in response.json()["detail"])


@pytest.mark.parametrize("body", ["{", "null", '{"instances":[{"age":NaN}]}'])
def test_malformed_or_nonfinite_json_returns_json_error(serving, body):
    client, _, _, _ = serving
    result = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert result.status_code == 422
    assert result.headers["content-type"] == "application/json"


def test_openapi_documents_all_required_nullable_fields(serving):
    client, _, _, _ = serving
    schema = client.get("/openapi.json").json()
    record = schema["components"]["schemas"]["FeatureRecord"]
    assert set(record["required"]) == set(record["properties"]) == set(FEATURES)
    assert record["additionalProperties"] is False
    assert client.get("/docs").status_code == 200


def test_corrupt_model_prevents_startup(tmp_path):
    saved_model(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        with TestClient(api.create_app(tmp_path)):
            pytest.fail("Corrupt artifact must not become ready")


def test_readiness_requires_lifespan():
    client = TestClient(api.create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_request_logs_do_not_contain_input_records(serving, caplog):
    client, _, payload, _ = serving
    with caplog.at_level(logging.INFO, logger="heart_disease.api"):
        response = client.post("/predict?private=value", json=payload)
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "heart_disease.api"
    ]
    event = events[-1]
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["route"] == "/predict" and event["status"] == 200
    assert "instances" not in event and "private" not in json.dumps(event)


def test_internal_error_returns_generic_500_and_service_stays_alive(serving, monkeypatch):
    client, app, payload, _ = serving

    def broken(*args):
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(app.state.predictor, "predict", broken)
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "sensitive" not in response.text
    assert client.get("/health").status_code == 200


@pytest.mark.integration
def test_selected_model_over_http_matches_cli_predictor():
    from heart_disease.config import ROOT

    payload = json.loads((ROOT / "docs/sample_request.json").read_text())
    with TestClient(api.create_app()) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == load_predictor().predict(payload)
