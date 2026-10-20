"""Acceptance instrumentation, statistics and monitoring regressions (no live cluster)."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from heart_disease.schema import FEATURES, RAW_COLUMNS
from scripts.acceptance import evaluation, publish, reporting
from scripts.acceptance.evidence import Ledger, inventory, read_json, write_json
from scripts.acceptance.run import archive_evidence, validate_config

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((PROJECT / "configs/acceptance.json").read_text())


@pytest.fixture
def frame():
    raw = pd.read_csv(
        PROJECT / "data/raw/processed.cleveland.data", names=RAW_COLUMNS, na_values="?"
    )
    raw["target"] = (raw["num"] > 0).astype(int)
    return raw.drop(columns="num").iloc[:61].copy()


def test_metrics_are_recomputed_from_labels_and_probabilities():
    result = evaluation.metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
    assert result["accuracy"] == 0.5
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["confusion_matrix_actual_rows_predicted_columns"] == [[1, 1], [1, 1]]
    assert result["brier_score"] == pytest.approx((0.01 + 0.64 + 0.36 + 0.01) / 4)


@pytest.mark.parametrize("probability", [[float("nan")], [-0.1], [1.1]])
def test_metrics_reject_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probabilities"):
        evaluation.metrics([1], [1], probability)


def test_single_class_auc_is_unavailable_rather_than_invented():
    assert evaluation.metrics([1, 1], [1, 0], [0.8, 0.2])["roc_auc"] is None


def test_bootstrap_is_seeded_and_includes_all_reported_scalar_metrics():
    args = ([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
    first = evaluation.confidence_intervals(*args, samples=10, seed=42)
    assert first == evaluation.confidence_intervals(*args, samples=10, seed=42)
    assert len(first["intervals"]) == 7
    assert all(a <= b for a, b in first["intervals"].values())


def test_monitoring_distinguishes_healthy_degraded_and_insufficient(config):
    baseline = {"records": 60, "class_counts": {"0": 30, "1": 30}, "recall": 0.95, "roc_auc": 0.96}
    args = dict(policy=config["monitoring"], scenario="DEMO", model_run_id="model-1")
    healthy = evaluation.monitoring_decision(baseline, baseline, **args)
    degraded = evaluation.monitoring_decision(baseline, {**baseline, "recall": 0.7}, **args)
    insufficient = evaluation.monitoring_decision(
        baseline, {**baseline, "records": 4, "recall": 0.1}, **args
    )
    assert healthy["decision"] == "healthy"
    assert degraded["request_retraining"] and degraded["triggers"][0]["metric"] == "recall"
    assert insufficient["decision"] == "insufficient_evidence"
    assert not insufficient["request_retraining"]
    assert not degraded["automatic_retraining_performed"]


def test_distribution_shift_does_not_invent_a_performance_measure(config, frame):
    unchanged = evaluation.drift(frame, frame, config["monitoring"])
    shifted = frame.copy()
    shifted["age"] += 100
    detected = evaluation.drift(frame, shifted, config["monitoring"])
    assert not unchanged["drift_alert"]
    assert detected["drift_alert"]
    assert "request_retraining" not in detected


def test_corruption_demo_preserves_labels_and_feature_marginals(frame):
    original = frame.copy(deep=True)
    result = evaluation.degraded_replay(frame, seed=42)
    pd.testing.assert_frame_equal(frame, original)
    assert result["target"].equals(frame["target"])
    for field in FEATURES:
        np.testing.assert_array_equal(np.sort(result[field]), np.sort(frame[field]))
    assert not result[FEATURES].equals(frame[FEATURES])


def test_endpoint_predictions_require_model_identity_and_batch_length(monkeypatch, frame):
    frame = frame.iloc[:2]
    monkeypatch.setattr(
        evaluation,
        "request_json",
        lambda *args: (200, {"model_run_id": "wrong", "predictions": []}, {}),
    )
    with pytest.raises(ValueError, match="identity"):
        evaluation.endpoint_predictions("http://example", frame, "expected")
    monkeypatch.setattr(
        evaluation,
        "request_json",
        lambda *args: (200, {"model_run_id": "expected", "predictions": []}, {}),
    )
    with pytest.raises(ValueError, match="count"):
        evaluation.endpoint_predictions("http://example", frame, "expected")


def test_ledger_failure_retains_test_method_actual_and_log(tmp_path, config):
    ledger = Ledger(tmp_path / "run", "windows", config)
    with pytest.raises(RuntimeError):
        with ledger.case(
            "TEST-1", "Intentional failure", ["R5"], "Execute a failing subprocess", "Exit 0"
        ):
            ledger.command(
                "failure", [sys.executable, "-c", "print('retained output'); raise SystemExit(3)"]
            )
    result = read_json(ledger.output / "results.json")
    assert result["cases"][0]["status"] == "failed"
    assert "exited 3" in result["cases"][0]["actual"]
    assert "retained output" in (ledger.output / "logs/failure.log").read_text()


@pytest.mark.parametrize(
    "field,value", [("load_requests", 0), ("jobs", True), ("bootstrap_samples", -1)]
)
def test_bad_configuration_is_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_archive_omits_execution_environment_and_has_verified_hashes(tmp_path):
    output = tmp_path / "run"
    (output / "ci/workspace").mkdir(parents=True)
    (output / "ci/workspace/private-cache").write_text("not delivery evidence")
    (output / "ci/temporary").mkdir()
    (output / "ci/temporary/test-fixture").write_text("temporary fixture, not run evidence")
    (output / "screenshots").mkdir()
    (output / "results.json").write_text('{"status":"failed"}')
    (output / "screenshots/image.png").write_bytes(b"test fixture")
    files = inventory(output)
    assert "ci/workspace/private-cache" not in files
    assert "ci/temporary/test-fixture" not in files
    archive = archive_evidence(output)
    with zipfile.ZipFile(archive) as evidence:
        assert evidence.testzip() is None
        assert "results.json" in evidence.namelist()
        assert "ci/workspace/private-cache" not in evidence.namelist()
        import hashlib

        manifest = json.loads(evidence.read("evidence-manifest.json"))
        assert set(manifest) == set(evidence.namelist()) - {"evidence-manifest.json"}
        for name, value in manifest.items():
            assert hashlib.sha256(evidence.read(name)).hexdigest() == value["sha256"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing regression")
def test_live_reader_cannot_abort_atomic_status_updates(tmp_path):
    from threading import Thread
    from time import sleep

    target = tmp_path / "live.json"
    write_json(target, {"value": "old"})
    reader = target.open("rb")

    def release():
        sleep(0.2)
        reader.close()

    thread = Thread(target=release)
    thread.start()
    try:
        write_json(target, {"value": "new"})
    finally:
        thread.join()
        reader.close()
    assert read_json(target) == {"value": "new"}


def test_report_handles_blocked_run_and_creates_exactly_ten_submission_pages(tmp_path, config):
    ledger = Ledger(tmp_path / "run", "windows", config)
    ledger.result.update(
        status="failed",
        error="Docker unavailable",
        cases=[
            {
                "id": "K8S-001",
                "purpose": "Deployment",
                "requirements": ["R7"],
                "status": "blocked",
                "method": "Deploy real manifests",
                "expected": "Ready pods",
                "actual": "Docker unavailable",
                "seconds": 0,
            }
        ],
    )
    ledger.save()
    reporting.generate(ledger.output)
    assert len(PdfReader(ledger.output / "report/assignment-submission.pdf").pages) == 10
    content = (ledger.output / "report/test-report.html").read_text(encoding="utf-8")
    assert "Docker unavailable" in content and "blocked" in content
    assert read_json(ledger.output / "report/requirement-coverage.json")[6]["status"] == "blocked"


def test_github_issue_is_explicitly_demo_and_rejects_missing_request(tmp_path):
    with pytest.raises(ValueError, match="DEMO"):
        publish.issue_payload(tmp_path, "owner/repo", "123")
    write_json(tmp_path / "results.json", {"commit": "abc123"})
    write_json(
        tmp_path / "retraining/request.json",
        {"demo": True, "request_retraining": True, "model_run_id": "model1"},
    )
    payload = publish.issue_payload(tmp_path, "owner/repo", "123")
    assert payload["title"].startswith("[DEMO]")
    assert "https://github.com/owner/repo/actions/runs/123" in payload["body"]
    assert "No automatic retraining" in payload["body"]


def test_github_issue_updates_existing_request_instead_of_creating_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    write_json(tmp_path / "results.json", {"commit": "abc123"})
    write_json(
        tmp_path / "retraining/request.json",
        {"demo": True, "request_retraining": True, "model_run_id": "model1"},
    )
    calls = []

    def api(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        if kwargs.get("method") == "PATCH":
            return {"number": 7, "html_url": "https://github.com/owner/repo/issues/7"}
        return [{"number": 7, "body": "<!-- heart-mlops-retraining-demo -->"}]

    monkeypatch.setattr(publish, "github_api", api)
    publish.publish_issue(tmp_path)
    assert len(calls) == 2 and calls[-1][0].endswith("/7")
    assert read_json(tmp_path / "retraining/github-issue.json")["updated_existing"]


def test_pytest_evidence_contains_inputs_expected_actual_and_skipped_reason(tmp_path):
    tests = tmp_path / "test_example.py"
    tests.write_text(
        "import pytest\n@pytest.mark.parametrize('x',[1,2])\ndef test_value(x):\n    assert x > 0\n@pytest.mark.skip(reason='platform fixture')\ndef test_skip():\n    assert False\ndef test_patch(monkeypatch):\n    def fail(*args):\n        raise OSError('injected replace failure')\n    monkeypatch.setattr('os.replace', fail)\n    assert True\n"
    )
    evidence = tmp_path / "cases.json"
    configuration = tmp_path / "pytest.ini"
    configuration.write_text("[pytest]\n")
    import os

    response = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "-p",
            "scripts.acceptance.pytest_evidence",
            "--noconftest",
            "-c",
            str(configuration),
        ],
        cwd=PROJECT,
        env={
            **os.environ,
            "ACCEPTANCE_TEST_EVIDENCE": str(evidence),
            "ACCEPTANCE_RUN_STARTED_AT": "2020-01-01T00:00:00+00:00",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert response.returncode == 0, response.stdout + response.stderr
    records = read_json(evidence)
    assert len(records) == 4
    assert records[0]["inputs"]["x"] == "1"
    assert "assert x > 0" in records[0]["expected"]
    assert records[0]["status"] == "passed"
    assert records[0]["start_seconds"] > 0
    assert records[0]["started_at"].endswith("+00:00")
    assert "platform fixture" in records[2]["actual"]
    assert records[-1]["status"] == "passed"


def test_bootstrap_diagnostics_do_not_require_training_dependencies(tmp_path, monkeypatch):
    from scripts.acceptance import manage

    project = tmp_path / "project"
    bootstrap = project / "reports/acceptance/bootstrap-case"
    bootstrap.mkdir(parents=True)
    write_json(bootstrap / "failure.json", {"status": "failed", "error": "Docker missing"})
    write_json(
        project / "reports/acceptance/latest-launch.json",
        {"directory": str(project / "reports/acceptance/not-started"), "bootstrap": str(bootstrap)},
    )
    monkeypatch.setattr(manage, "PROJECT", project)
    assert manage.latest() == bootstrap
    manage.manage("collect")
    archive = bootstrap.with_name(bootstrap.name + "-evidence.zip")
    with zipfile.ZipFile(archive) as output:
        assert json.loads(output.read("failure.json"))["error"] == "Docker missing"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bootstrap argument contract")
def test_one_click_bootstrap_preserves_arguments_and_rejects_native_failure(tmp_path):
    bootstrap = str(PROJECT / "scripts/run_acceptance.ps1").replace("'", "''")
    python = sys.executable.replace("'", "''")
    script = tmp_path / "bootstrap contract.ps1"
    script.write_text(
        "$ErrorActionPreference='Stop'\n$tokens=$null; $errors=$null\n"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{bootstrap}',[ref]$tokens,[ref]$errors)\n"
        "if($errors.Count){throw 'parse failure'}\n"
        "$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-AcceptanceNative'},$true)\n"
        "Invoke-Expression $function.Extent.Text\n"
        f"Invoke-AcceptanceNative -Executable '{python}' -NativeArguments @('-c','import json,sys; print(json.dumps(sys.argv[1:]))','argument with spaces','last argument')\n"
        "$failed=$false\n"
        f"try{{Invoke-AcceptanceNative -Executable '{python}' -NativeArguments @('-c','raise SystemExit(9)')}}catch{{$failed=$true}}\n"
        "if(-not $failed){throw 'Native failure was ignored'}\n",
        encoding="utf-8",
    )
    response = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert response.returncode == 0, response.stdout + response.stderr
    assert json.loads(response.stdout.strip()) == ["argument with spaces", "last argument"]
