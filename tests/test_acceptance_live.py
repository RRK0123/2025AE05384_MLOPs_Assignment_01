"""Real labeled HTTP/monitoring acceptance against the freshly trained CI model."""

import json
import os
import sys

import pandas as pd
import pytest

from heart_disease.config import ROOT
from heart_disease.predict import load_predictor
from heart_disease.releases import resolve_model_directory
from scripts.acceptance.deployment import free_port
from scripts.acceptance.evaluation import load_check, payload_for
from scripts.acceptance.evidence import Ledger, read_json, stop
from scripts.acceptance.run import PROJECT, evaluate_endpoint, monitoring, robustness
from scripts.smoke_api import request_json, wait_ready

pytestmark = pytest.mark.integration


def test_model_labeled_endpoint_metrics_robustness_load_and_retraining_request(tmp_path):
    """Real HTTP; unchanged labels; healthy replay and input-corruption request."""
    config = json.loads((PROJECT / "configs/acceptance.json").read_text())
    config["bootstrap_samples"] = 20
    ledger = Ledger(tmp_path / "live-acceptance", "windows", config)
    release = resolve_model_directory()
    predictor = load_predictor(release)
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    expected = predictor.predict(payload_for(test))
    url = f"http://127.0.0.1:{free_port()}"
    server = ledger.background(
        "api",
        [
            sys.executable,
            "-m",
            "heart_disease.api",
            "--model-dir",
            release,
            "--port",
            url.rsplit(":", 1)[-1],
        ],
        env={**os.environ, "HEART_DISEASE_PROJECT_ROOT": str(ROOT), "PYTHONUTF8": "1"},
    )
    try:
        wait_ready(url, process=server)
        observed = evaluate_endpoint(ledger, url, test, expected, config, native=True)
        assert observed["saved_model_predictions_match"]
        for name in ["accuracy", "precision", "recall", "roc_auc", "brier_score"]:
            assert observed["metrics"][name] == pytest.approx(
                predictor.metadata["test_metrics"][name]
            )
        assert len(observed["confidence_intervals"]["intervals"]) == 7
        outcomes = robustness(ledger, url, test)
        assert len(outcomes) == 10
        assert all(case["expected_http"] == case["actual_http"] for case in outcomes)
        before = request_json(url + "/metrics")[1]
        load = load_check(url, payload_for(test.iloc[:1]), requests=12, concurrency=2)
        after = request_json(url + "/metrics")[1]
        assert load["error_rate"] == 0
        assert after["predictions"]["records"] - before["predictions"]["records"] == 12
        decision = monitoring(
            ledger, url, test, observed["metrics"], config, expected["model_run_id"]
        )
        assert decision["healthy_decision"] == "healthy"
        assert decision["degraded_decision"] == "request_retraining"
        request = read_json(ledger.output / "retraining/request.json")
        assert request["demo"] and not request["automatic_retraining_performed"]
        assert request["model_run_id"] == predictor.metadata["run_id"]
        assert (ledger.output / "retraining/request.html").exists()
    finally:
        stop(server)
    assert server.poll() is not None
