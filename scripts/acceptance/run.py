"""One execution: fresh training, tracking, tests, deployment and submission evidence."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.acceptance import deployment, evaluation, reporting
from scripts.acceptance.evidence import (
    Ledger,
    archive_evidence,
    inventory,
    read_json,
    stop,
    write_json,
)
from scripts.smoke_api import request_json, verify_service, wait_ready

PROJECT = Path(__file__).resolve().parents[2]
PLANNED = [
    ("PRE-001", "Execution prerequisites", ["R5", "R7"]),
    ("MEDIA-001", "Screen recording and browser readiness", ["R9"]),
    ("PIPE-001", "Fresh model pipeline and automated tests", ["R1", "R2", "R3", "R4", "R5"]),
    ("TRACK-001", "Live MLflow experiment and saved artifacts", ["R3"]),
    ("COV-001", "Line and branch coverage", ["R5"]),
    ("API-001", "Native API predictions and monitoring contract", ["R4", "R6", "R8"]),
    ("DOCKER-001", "Build and verify this run's Docker image", ["R6"]),
    ("K8S-001", "Deploy and expose this run's selected model", ["R7"]),
    ("EVAL-001", "Labeled deployed endpoint evaluation", ["R2", "R7"]),
    ("ROBUST-001", "Synthetic boundary and validation cases", ["R6", "R8"]),
    ("LOAD-001", "Concurrent deployed prediction traffic", ["R7", "R8"]),
    ("MON-001", "Healthy, drift and degraded monitoring decisions", ["R8"]),
    ("LOG-001", "Correlated deployed request logging", ["R8"]),
    ("REPORT-001", "Detailed and assignment reports", ["R9"]),
]


def validate_config(config):
    for name in ["jobs", "load_requests", "load_concurrency", "bootstrap_samples"]:
        if (
            not isinstance(config.get(name), int)
            or isinstance(config[name], bool)
            or config[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    policy = config["monitoring"]
    for name in [
        "recall_drop",
        "roc_auc_drop",
        "numeric_ks_threshold",
        "categorical_total_variation_threshold",
        "maximum_error_rate",
    ]:
        if not 0 < policy[name] <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    if (
        policy["minimum_labeled_records"] < 1
        or policy["minimum_per_class"] < 1
        or policy["maximum_p95_latency_ms"] <= 0
    ):
        raise ValueError("Monitoring minimum counts and latency must be positive")
    return config


def wait_media(ledger, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = read_json(ledger.output / "media-status.json", {})
        if state.get("status") == "ready":
            return
        if process.poll() is not None or state.get("status") == "failed":
            raise RuntimeError(f"Capture startup failed: {state}; see logs/media.log")
        time.sleep(0.5)
    raise TimeoutError("Browser/screen recording was not ready after 60 seconds")


def preflight(ledger, native):
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Python 3.13 is required")
    required_disk = (1 if native else 8) * 1024**3
    if shutil.disk_usage(ledger.output).free < required_disk:
        raise RuntimeError(
            f"Less than {required_disk // 1024**3} GiB free disk; free space before running"
        )
    if native:
        return {"scope": "DEVELOPMENT NATIVE VALIDATION ONLY; Docker/Kubernetes not executed"}
    tools = (
        ["docker", "kubectl", "minikube"]
        if ledger.result["track"] == "windows"
        else ["docker", "kubectl", "kind", "cloud-provider-kind"]
    )
    missing = [name for name in tools if not shutil.which(name)]
    if missing:
        raise FileNotFoundError("Missing tools: " + ", ".join(missing))
    info = ledger.command(
        "docker-preflight", ["docker", "info", "--format", "{{.OSType}}"], timeout=30
    )
    if info["output"].strip() != "linux":
        raise RuntimeError("Docker Desktop must be running Linux containers")
    versions = {
        "docker": ledger.command(
            "docker-version", ["docker", "version", "--format", "{{json .}}"], timeout=30
        )["output"],
        "kubectl": ledger.command(
            "kubectl-version", ["kubectl", "version", "--client=true", "-o", "json"], timeout=30
        )["output"],
        "cluster_driver": ledger.command(
            "cluster-driver-version",
            ["minikube" if ledger.result["track"] == "windows" else "kind", "version"],
            timeout=30,
        )["output"],
    }
    write_json(ledger.output / "environment/tool-versions.json", versions)
    if ledger.result["track"] == "windows":
        with socket.socket() as check:
            try:
                check.bind(("127.0.0.1", 8080))
            except OSError as error:
                raise RuntimeError(
                    "Port 8080 is in use. Close the old demo tunnel before starting a new complete run."
                ) from error
    return {
        "tools": {name: shutil.which(name) for name in tools},
        "versions": versions,
        "linux_engine": True,
    }


def start_services(ledger, workspace, release):
    env = {
        **os.environ,
        "HEART_DISEASE_PROJECT_ROOT": str(workspace),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
    }
    api_port, tracking_port = deployment.free_port(), deployment.free_port()
    api = ledger.background(
        "native-api",
        [sys.executable, "-m", "heart_disease.api", "--port", api_port, "--model-dir", release],
        env=env,
    )
    api_url = f"http://127.0.0.1:{api_port}"
    wait_ready(api_url, process=api)
    # Launch MLflow directly, avoiding a parent process that owns a second server.
    uri = "sqlite:///" + (workspace / "tracking/mlflow.db").as_posix()
    tracking = ledger.background(
        "mlflow",
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            uri,
            "--host",
            "127.0.0.1",
            "--port",
            tracking_port,
            "--workers",
            "1",
        ],
        env=env,
    )
    tracking_url = f"http://127.0.0.1:{tracking_port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            status, _, _ = request_json(
                tracking_url + "/api/2.0/mlflow/experiments/search", {"max_results": 100}
            )
            if status == 200:
                break
        except (OSError, json.JSONDecodeError):
            pass
        if tracking.poll() is not None:
            raise RuntimeError("MLflow UI exited; see logs/mlflow.log")
        time.sleep(1)
    else:
        raise TimeoutError("MLflow UI did not become ready")
    services = {
        "native_api": api_url,
        "mlflow": tracking_url,
        "processes": [
            {"pid": api.pid, "name": "native-api"},
            {"pid": tracking.pid, "name": "mlflow"},
        ],
        "workspace": str(workspace),
    }
    write_json(ledger.output / "services.json", services)
    return services


def evaluate_endpoint(ledger, url, test, reference, config, *, native):
    observed = evaluation.endpoint_predictions(url, test, reference["model_run_id"])
    labels = [item["prediction"] for item in observed]
    probabilities = [item["probability_disease"] for item in observed]
    if labels != [item["prediction"] for item in reference["predictions"]]:
        raise ValueError("Endpoint classes differ from the saved model")
    np.testing.assert_allclose(
        probabilities,
        [item["probability_disease"] for item in reference["predictions"]],
        rtol=1e-12,
        atol=1e-12,
    )
    for item in observed:
        expected = (
            item["probability_disease"]
            if item["prediction"] == 1
            else 1 - item["probability_disease"]
        )
        if not np.isclose(item["confidence"], expected):
            raise ValueError("Predicted-class confidence is inconsistent")
    result = {
        "metrics": evaluation.metrics(test["target"], labels, probabilities),
        "confidence_intervals": evaluation.confidence_intervals(
            test["target"],
            labels,
            probabilities,
            samples=config["bootstrap_samples"],
            seed=config["seed"],
        ),
        "dataset": "Reserved labeled test.csv; no tuning uses these observations",
        "model_run_id": reference["model_run_id"],
        "saved_model_predictions_match": True,
        "endpoint": url,
        "scope": "native" if native else "Kubernetes LoadBalancer",
    }
    write_json(ledger.output / "evaluation" / "endpoint-evaluation.json", result)
    records = [
        {"row": index, "target": int(test.iloc[index]["target"]), **item}
        for index, item in enumerate(observed)
    ]
    write_json(ledger.output / "evaluation" / "test-predictions.json", records)
    return result


def robustness(ledger, url, test):
    example = evaluation.payload_for(test.iloc[:1])["instances"][0]
    checks = [
        ("null imputation", {"instances": [{name: None for name in example}]}, 200),
        ("maximum batch", {"instances": [example] * 256}, 200),
        ("oversize batch", {"instances": [example] * 257}, 422),
        ("empty batch", {"instances": []}, 422),
        ("missing feature", {"instances": [{k: v for k, v in example.items() if k != "age"}]}, 422),
        ("unknown feature", {"instances": [{**example, "num": 1}]}, 422),
        ("boolean measurement", {"instances": [{**example, "age": True}]}, 422),
        ("unknown category", {"instances": [{**example, "cp": 99}]}, 422),
        ("negative measurement", {"instances": [{**example, "chol": -1}]}, 422),
        ("numeric string", {"instances": [{**example, "age": "50"}]}, 422),
    ]
    outcomes = []
    for name, payload, expected in checks:
        status, response, _ = request_json(url + "/predict", payload)
        outcomes.append(
            {
                "test": name,
                "input": payload,
                "expected_http": expected,
                "actual_http": status,
                "response": response,
            }
        )
        write_json(ledger.output / "evaluation/robustness.json", outcomes)
        if status != expected:
            raise ValueError(f"{name}: HTTP {status}, expected {expected}")
    return outcomes


def monitoring(ledger, url, test, baseline, config, run_id):
    policy = config["monitoring"]
    healthy = evaluation.monitoring_decision(
        baseline, baseline, policy, scenario="healthy labeled replay", model_run_id=run_id
    )
    if healthy["decision"] != "healthy":
        raise ValueError("Healthy replay did not satisfy the documented evidence gate")
    replay = evaluation.degraded_replay(test, seed=config["seed"])
    predicted = evaluation.endpoint_predictions(url, replay, run_id)
    degraded_metrics = evaluation.metrics(
        test["target"],
        [r["prediction"] for r in predicted],
        [r["probability_disease"] for r in predicted],
    )
    degraded = evaluation.monitoring_decision(
        baseline,
        degraded_metrics,
        policy,
        scenario="DEMO independently permuted measurement columns; labels unchanged",
        model_run_id=run_id,
    )
    if not degraded["request_retraining"]:
        raise ValueError(
            "Controlled corruption did not trigger the configured degradation criterion; inspect replay metrics and criteria"
        )
    shifted = test.copy()
    shifted["age"] += 35
    distribution = evaluation.drift(test, shifted, policy)
    if not distribution["drift_alert"]:
        raise ValueError("Controlled age distribution shift was not detected")
    write_json(
        ledger.output / "monitoring/decisions.json",
        {"healthy": healthy, "degraded": degraded, "drift": distribution},
    )
    write_json(
        ledger.output / "monitoring/degraded-predictions.json",
        {
            "data_scope": "DEMO corruption replay; original labels preserved",
            "metrics": degraded_metrics,
            "predictions": predicted,
        },
    )
    request = {
        **degraded,
        "run_id": ledger.result["run_id"],
        "demo": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_metrics": baseline,
        "observed_metrics": degraded_metrics,
        "evidence": ["monitoring/decisions.json", "monitoring/degraded-predictions.json"],
        "requested_action": "Review data quality and consider retraining; no automatic training or promotion.",
    }
    write_json(ledger.output / "retraining/request.json", request)
    (ledger.output / "retraining/request.md").write_text(
        f"# DEMO retraining request\n\nRun: {ledger.result['run_id']}\n\nModel: {run_id}\n\n"
        + json.dumps(request, indent=2)
        + "\n",
        encoding="utf-8",
    )
    from html import escape

    (ledger.output / "retraining/request.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>DEMO retraining request</title>"
        "<style>body{font:18px system-ui;margin:40px;max-width:1100px}h1{color:#a62a27}pre{white-space:pre-wrap}</style>"
        "<h1>DEMO: retraining review requested</h1><p>Controlled input-corruption replay. No automatic training or model replacement.</p><pre>"
        + escape(json.dumps(request, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    summary = {
        "healthy_decision": healthy["decision"],
        "degraded_decision": degraded["decision"],
        "drift_alert": distribution["drift_alert"],
        "demo": True,
        "automatic_retraining": False,
    }
    write_json(ledger.output / "monitoring/summary.json", summary)
    return summary


def collect_coverage(ledger):
    folder = ledger.output / "coverage"
    config = folder / "coverage.ini"
    if not config.exists() or not list(folder.glob(".coverage.*")):
        raise RuntimeError("No instrumented coverage data was produced")
    for name, args in [
        ("coverage-combine", ["combine", "--keep", str(folder)]),
        ("coverage-json", ["json", "-o", str(folder / "coverage.json")]),
        ("coverage-xml", ["xml", "-o", str(folder / "coverage.xml")]),
        ("coverage-html", ["html", "-d", str(folder / "html")]),
    ]:
        ledger.command(
            name, [sys.executable, "-m", "coverage", args[0], f"--rcfile={config}", *args[1:]]
        )


def run(output, track, config, *, native=False):
    config = validate_config(config)
    if (
        os.environ.get("GITHUB_REPOSITORY")
        and config["submission"].get("repository_url") == "Not supplied"
    ):
        config["submission"]["repository_url"] = (
            "https://github.com/" + os.environ["GITHUB_REPOSITORY"]
        )
    ledger = Ledger(output, track, config)
    import coverage

    coverage_folder = ledger.output / "coverage"
    coverage_folder.mkdir(exist_ok=True)
    coverage_config = coverage_folder / "coverage.ini"
    coverage_config.write_text(
        "[run]\nbranch = True\nparallel = True\nsource = heart_disease,scripts.acceptance\n"
        f"data_file = {(coverage_folder / '.coverage').as_posix()}\n"
        "[paths]\napplication =\n"
        f"    {(PROJECT / 'src/heart_disease').as_posix()}\n"
        f"    {(ledger.output / 'ci/workspace/src/heart_disease').as_posix()}\n",
        encoding="utf-8",
    )
    coordinator_coverage = coverage.Coverage(
        data_file=str(coverage_folder / ".coverage"),
        data_suffix="coordinator",
        branch=True,
        source=["heart_disease", "scripts.acceptance"],
    )
    coordinator_coverage.start()
    ledger.result["complete_acceptance"] = False
    ledger.result["development_native_validation"] = native
    media = None
    try:
        with ledger.case(
            "MEDIA-001",
            "Screen recording and browser readiness",
            ["R9"],
            "Launch a separate browser/screen capture process and require ready acknowledgment.",
            "Actual continuous recording is active before test execution.",
        ):
            media = ledger.background(
                "media",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    f"--rcfile={coverage_config}",
                    "-m",
                    "scripts.acceptance.media",
                    "--output",
                    ledger.output,
                ],
            )
            wait_media(ledger, media)
            ledger.snapshot("execution-start")
        with ledger.case(
            "PRE-001",
            "Execution prerequisites",
            ["R5", "R7"],
            "Inspect Python, disk, required CLIs and the live Docker engine.",
            "Required environment is available.",
        ) as case:
            case["actual"] = preflight(ledger, native)
        with ledger.case(
            "PIPE-001",
            "Fresh model pipeline and automated tests",
            ["R1", "R2", "R3", "R4", "R5"],
            "Run the CI CLI in a newly created workspace with per-test evidence and branch coverage.",
            "Unit/integration tests, EDA, model/tracking validation, bundle rebuild and isolated-wheel checks pass.",
        ) as case:
            ledger.command(
                "fresh-pipeline",
                [
                    sys.executable,
                    "scripts/ci.py",
                    "--output",
                    ledger.output / "ci",
                    "--jobs",
                    config["jobs"],
                    "--evidence",
                ],
                timeout=5400,
                env={**os.environ, "ACCEPTANCE_RUN_STARTED_AT": ledger.result["started_at"]},
            )
            summary = read_json(ledger.output / "ci/summary.json")
            if summary["status"] != "passed":
                raise ValueError("Fresh CI pipeline did not pass")
            case["actual"] = summary
            case["evidence"] = [
                "ci/summary.json",
                "ci/steps.json",
                "ci/unit-tests-details.json",
                "ci/integration-tests-details.json",
            ]
        with ledger.case(
            "COV-001",
            "Line and branch coverage",
            ["R5"],
            "Combine instrumented tests/CLI coverage and export HTML, JSON and XML.",
            "Actual line/branch coverage and missing locations are available.",
        ) as case:
            coordinator_coverage.save()
            collect_coverage(ledger)
            case["actual"] = read_json(ledger.output / "coverage/coverage.json")["totals"]
        workspace = ledger.output / "ci/workspace"
        pointer = read_json(workspace / "reports/modeling/latest.json")
        release = workspace / pointer["model_directory"]
        model_run = pointer["run_id"]
        ledger.result.update(
            model_run_id=model_run,
            model_sha256=hashlib.sha256((release / "model.joblib").read_bytes()).hexdigest(),
            bundle_sha256=summary["bundle_sha256"],
        )
        statistics = evaluation.statistical_data(workspace)
        write_json(ledger.output / "statistics.json", statistics)
        # Preserve portable reports/data separately from the large working environment.
        shutil.copytree(workspace / "reports", ledger.output / "model-evidence")
        shutil.copytree(workspace / "data", ledger.output / "data-evidence")
        model_artifacts = ledger.output / "model-artifacts"
        model_artifacts.mkdir()
        shutil.copyfile(summary["bundle"], model_artifacts / "verified-model-bundle.zip")
        shutil.copytree(release, model_artifacts / "selected-release")
        for artifact in (workspace / "models/runs" / model_run).glob("*.joblib"):
            shutil.copyfile(artifact, model_artifacts / artifact.name)
        # A consistent SQLite snapshot is taken while no training writes occur.
        import sqlite3

        tracking_dir = ledger.output / "mlflow"
        tracking_dir.mkdir(exist_ok=True)
        shutil.copytree(workspace / "tracking/artifacts", tracking_dir / "artifacts")
        write_json(
            tracking_dir / "portability.json",
            {
                "original_workspace": str(workspace),
                "artifact_snapshot": "mlflow/artifacts",
                "database_snapshot": "mlflow/mlflow.db",
                "note": "Database retains original absolute artifact URIs. The live UI uses the run workspace; moving this snapshot requires relocating those URIs or rerunning training.",
            },
        )
        with sqlite3.connect(str(workspace / "tracking/mlflow.db")) as source:
            with sqlite3.connect(str(tracking_dir / "mlflow.db")) as destination:
                source.backup(destination)
        services = start_services(ledger, workspace, release)
        experiment = statistics["metadata"]["experiment_tracking"]
        with ledger.case(
            "TRACK-001",
            "Live MLflow experiment and saved artifacts",
            ["R3"],
            "Query the actual UI backend and open this session's experiment/run pages.",
            "The session parent and its recorded experiment are visible and tracking validation passes.",
        ) as case:
            parent = experiment["parent_run_id"]
            status, response, _ = request_json(
                services["mlflow"] + "/api/2.0/mlflow/runs/get?run_id=" + parent
            )
            if status != 200 or response["run"]["info"]["status"] != "FINISHED":
                raise ValueError("Actual MLflow parent is not finished or queryable")
            write_json(ledger.output / "mlflow/parent-run.json", response)
            exp_id = str(response["run"]["info"]["experiment_id"])
            ledger.snapshot(
                "mlflow-experiment",
                services["mlflow"] + "/#/experiments/" + exp_id,
                "body",
                action="mlflow",
                view="experiment",
                run_name="training-" + model_run,
                full_page=False,
            )
            ledger.snapshot(
                "mlflow-parent-run",
                services["mlflow"] + f"/#/experiments/{exp_id}/runs/{parent}",
                "body",
                action="mlflow",
                view="overview",
                run_name="training-" + model_run,
                full_page=False,
            )
            for view in ["metrics", "parameters", "artifacts"]:
                ledger.snapshot(
                    "mlflow-" + view,
                    services["mlflow"] + f"/#/experiments/{exp_id}/runs/{parent}",
                    "body",
                    action="mlflow",
                    view=view,
                    run_name="training-" + model_run,
                    full_page=False,
                )
            for family, final_run in experiment["final_model_run_ids"].items():
                ledger.snapshot(
                    "mlflow-final-" + family,
                    services["mlflow"] + f"/#/experiments/{exp_id}/runs/{final_run}",
                    "body",
                    action="mlflow",
                    view="overview",
                    run_name=family + "-final",
                    full_page=False,
                )
            case["actual"] = statistics["tracking"]
            case["evidence"] = [
                "mlflow/parent-run.json",
                "screenshots/mlflow-experiment.png",
                "screenshots/mlflow-parent-run.png",
            ]
        # Capture real EDA/model figures, not screenshots of historical outputs.
        figures = sorted((workspace / "reports/figures").glob("*.png"))
        for index, figure in enumerate(figures):
            ledger.snapshot(
                "eda-figure" if index == 0 else "eda-" + figure.stem,
                "/" + figure.relative_to(ledger.output).as_posix(),
                "img",
            )
        model_figures = sorted(
            (workspace / Path(pointer["report"]).parent / "figures").glob("*.png")
        )
        for index, figure in enumerate(model_figures):
            ledger.snapshot(
                "model-evaluation" if index == 0 else "model-" + figure.stem,
                "/" + figure.relative_to(ledger.output).as_posix(),
                "img",
            )
        with ledger.case(
            "API-001",
            "Native API predictions and monitoring contract",
            ["R4", "R6", "R8"],
            "Exercise real HTTP requests against the freshly trained native API.",
            "Packaged fixtures, readiness, validation and monitoring assertions pass.",
        ) as case:
            case["actual"] = verify_service(services["native_api"], ledger.output / "ci/bundle")
            write_json(ledger.output / "evaluation/native-smoke.json", case["actual"])
        if not native:
            image = f"heart-disease:acceptance-{ledger.result['run_id'][-8:].lower()}"
            with ledger.case(
                "DOCKER-001",
                "Build and verify this run's Docker image",
                ["R6"],
                "Build the exact current bundle and run container health/security/prediction checks.",
                "Real Docker smoke verification passes.",
            ) as case:
                ledger.command(
                    "docker",
                    [
                        sys.executable,
                        "scripts/docker_smoke.py",
                        "--bundle",
                        summary["bundle"],
                        "--output",
                        ledger.output / "docker",
                        "--image",
                        image,
                    ],
                    timeout=1800,
                )
                case["actual"] = read_json(
                    ledger.output / "docker/validation.json",
                    read_json(ledger.output / "docker/result.json", {}),
                )
                if (
                    case["actual"].get("status") != "passed"
                    or case["actual"].get("image_tag") != image
                ):
                    raise ValueError(
                        "Docker verification did not record the expected successful image"
                    )
                ledger.result["built_image_id"] = case["actual"]["image_id"]
                case["evidence"] = ["logs/docker.log"]
            with ledger.case(
                "K8S-001",
                "Deploy and expose this run's selected model",
                ["R7"],
                "Create an owned cluster, load this image, validate/apply manifests and query the actual LoadBalancer.",
                "Ready pods run the expected image and expose the expected model.",
            ) as case:
                url = deployment.deploy(ledger, PROJECT, image, model_run)
                case["actual"] = {"url": url, "image": image, "model_run_id": model_run}
                ledger.snapshot("kubernetes-resources")
        else:
            url = services["native_api"]
        ledger.snapshot("prediction-swagger", url + "/docs", ".swagger-ui")
        ledger.snapshot(
            "prediction-response",
            url + "/docs",
            ".responses-wrapper",
            action="predict",
            payload=read_json(ledger.output / "ci/bundle/fixtures/request.json"),
            model_run_id=model_run,
        )
        from heart_disease.predict import load_predictor

        test = pd.read_csv(workspace / "data/processed/test.csv")
        reference = load_predictor(release).predict(evaluation.payload_for(test))
        with ledger.case(
            "EVAL-001",
            "Labeled deployed endpoint evaluation",
            ["R2", "R7"] if not native else ["R2"],
            "Send every reserved test record through HTTP; compare with saved model and known labels; bootstrap uncertainty.",
            "Classes/probabilities match the selected saved pipeline and all metrics are recorded.",
        ) as case:
            endpoint_evaluation = evaluate_endpoint(
                ledger, url, test, reference, config, native=native
            )
            statistics["native_evaluation" if native else "deployed_evaluation"] = (
                endpoint_evaluation
            )
            case["actual"] = endpoint_evaluation
            case["evidence"] = [
                "evaluation/endpoint-evaluation.json",
                "evaluation/test-predictions.json",
            ]
        with ledger.case(
            "ROBUST-001",
            "Synthetic boundary and validation cases",
            ["R6", "R8"],
            "Send null, maximum-size, oversized, malformed-schema and invalid-value requests.",
            "Valid requests return 200; invalid requests return 422.",
        ) as case:
            case["actual"] = [
                {"test": c["test"], "actual_http": c["actual_http"]}
                for c in robustness(ledger, url, test)
            ]
            case["evidence"] = ["evaluation/robustness.json"]
        with ledger.case(
            "LOAD-001",
            "Concurrent deployed prediction traffic",
            ["R7", "R8"] if not native else ["R8"],
            "Issue concurrent one-record requests, measure client latency and reconcile server prediction counters.",
            "No unexpected HTTP failures; record totals agree and operational thresholds are satisfied.",
        ) as case:
            before = request_json(url + "/metrics")[1]
            load = evaluation.load_check(
                url,
                evaluation.payload_for(test.iloc[:1]),
                requests=config["load_requests"],
                concurrency=config["load_concurrency"],
            )
            after = request_json(url + "/metrics")[1]
            write_json(ledger.output / "evaluation/load.json", load)
            write_json(
                ledger.output / "monitoring/load-counters.json", {"before": before, "after": after}
            )
            if (
                after["predictions"]["records"] - before["predictions"]["records"]
                != config["load_requests"]
            ):
                raise ValueError("Prediction counters do not reconcile with load traffic")
            load["operational_alert"] = (
                load["error_rate"] > config["monitoring"]["maximum_error_rate"]
                or load["latency_ms"]["p95"] > config["monitoring"]["maximum_p95_latency_ms"]
            )
            write_json(ledger.output / "evaluation/load.json", load)
            write_json(
                ledger.output / "monitoring/operational.json",
                {
                    "alert": load["operational_alert"],
                    "error_rate": load["error_rate"],
                    "p95_latency_ms": load["latency_ms"]["p95"],
                    "criteria": config["monitoring"],
                    "requested_action": "Investigate service health and resources"
                    if load["operational_alert"]
                    else "No operational action",
                    "request_retraining": False,
                },
            )
            if load["operational_alert"]:
                raise ValueError(
                    "Operational load thresholds exceeded; retraining is not the remediation for this alert"
                )
            statistics["load"] = {k: v for k, v in load.items() if k != "observations"}
            case["actual"] = statistics["load"]
        with ledger.case(
            "MON-001",
            "Healthy, drift and degraded monitoring decisions",
            ["R8"],
            "Evaluate unchanged labeled replay, independent feature-column corruption and a controlled age distribution shift.",
            "Healthy replay has no request; degradation writes a DEMO retraining request; drift raises an investigation alert.",
        ) as case:
            case["actual"] = monitoring(
                ledger, url, test, endpoint_evaluation["metrics"], config, model_run
            )
            case["evidence"] = ["monitoring/decisions.json", "retraining/request.json"]
        ledger.snapshot("monitoring-dashboard", url + "/monitoring", "#class0")
        ledger.snapshot("retraining-request", "/retraining/request.html", "h1")
        with ledger.case(
            "LOG-001",
            "Correlated deployed request logging",
            ["R8"],
            "Find a unique request ID from a successful HTTP prediction in the actual serving log stream.",
            "Exactly one successful structured event contains the known request ID and no raw inputs.",
        ) as case:
            status, _, headers = request_json(
                url + "/predict", evaluation.payload_for(test.iloc[:1])
            )
            request_id = next(v for k, v in headers.items() if k.lower() == "x-request-id")
            if status != 200:
                raise ValueError("Correlation request failed")
            if native:
                log = (ledger.output / "logs/native-api.log").read_text(
                    encoding="utf-8", errors="replace"
                )
            else:
                log = ledger.command(
                    "correlated-api-logs",
                    [
                        "kubectl",
                        "--context",
                        ledger.result["cluster"]["context"],
                        "--request-timeout=20s",
                        "-n",
                        deployment.NAMESPACE,
                        "logs",
                        "deployment/heart-api",
                        "--tail=3000",
                    ],
                )["output"]
            events = []
            for line in log.splitlines():
                try:
                    event = json.loads(line)
                    if (
                        event.get("request_id") == request_id
                        and event.get("event") == "http_request"
                    ):
                        events.append(event)
                except (json.JSONDecodeError, AttributeError):
                    pass
            if (
                len(events) != 1
                or events[0]["status"] != 200
                or any(key in events[0] for key in ["instances", "age", "query"])
            ):
                raise ValueError("Structured log correlation/privacy verification failed")
            case["actual"] = events
        write_json(ledger.output / "statistics.json", statistics)
        services.update(deployed_api=None if native else url, cluster=ledger.result.get("cluster"))
        services["processes"] = [
            {"pid": p.pid, "name": "owned-run-service"}
            for p in ledger.processes
            if p is not media and p.poll() is None
        ]
        write_json(ledger.output / "services.json", services)
        ledger.result.update(
            status="partial" if native else "passed", complete_acceptance=not native
        )
        ledger.save()
        if track == "github" and not native:
            if os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
                raise RuntimeError("GitHub publication requires a trusted push/manual workflow")
            from scripts.acceptance.publish import publish_image, publish_issue

            with ledger.case(
                "GHCR-001",
                "Publish verified image",
                ["R6"],
                "Push the successfully deployed and tested image to this repository's GHCR package.",
                "Registry publication succeeds and its tag is recorded.",
            ) as case:
                publish_image(ledger.output)
                ledger.result["published_image"] = read_json(ledger.output / "results.json")[
                    "published_image"
                ]
                case["actual"] = read_json(ledger.output / "deployment/registry-publication.json")
            with ledger.case(
                "ISSUE-001",
                "Publish DEMO retraining request",
                ["R5", "R8"],
                "Create or update the marked DEMO GitHub issue through the scoped workflow token.",
                "An issue URL and number are saved; duplicate open requests are updated.",
            ) as case:
                publish_issue(ledger.output)
                case["actual"] = read_json(ledger.output / "retraining/github-issue.json")
            ledger.snapshot("github-retraining-issue", "/retraining/github-issue.json", "body")
    except BaseException as error:
        ledger.result.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(ledger.result["error"], flush=True)
    finally:
        ledger.save()
        coordinator_coverage.stop()
        coordinator_coverage.save()
        if (ledger.output / "coverage/coverage.ini").exists() and list(
            (ledger.output / "coverage").glob(".coverage.*")
        ):
            try:
                collect_coverage(ledger)
            except Exception as error:
                ledger.result["coverage_error"] = str(error)
        try:
            deployment.diagnostics(ledger)
        except Exception as error:
            ledger.result["diagnostic_error"] = str(error)
        completed = {case["id"] for case in ledger.result["cases"]}
        for identifier, title, requirements in PLANNED:
            if identifier not in completed and identifier != "REPORT-001":
                ledger.result["cases"].append(
                    {
                        "id": identifier,
                        "purpose": title,
                        "requirements": requirements,
                        "status": "skipped"
                        if native and identifier in {"DOCKER-001", "K8S-001"}
                        else "blocked",
                        "actual": "Development native validation"
                        if native
                        else "An earlier required stage failed",
                        "seconds": 0,
                    }
                )
        try:
            with ledger.case(
                "REPORT-001",
                "Detailed and assignment reports",
                ["R9"],
                "Render all current results, statistics and coverage to HTML/PDF, and verify the submission page count.",
                "Detailed report, JUnit, requirement matrix and exactly ten submission pages exist.",
            ) as case:
                reporting.generate(ledger.output)
                case["actual"] = (
                    "Generated detailed HTML/PDF and a verified ten-page assignment PDF."
                )
            if media and media.poll() is None:
                ledger.snapshot(
                    "final-test-report", "/report/test-report.html", "h1", full_page=False
                )
        except Exception as error:
            ledger.result.update(status="failed", reporting_error=str(error))
        ledger.save()
        if media:
            write_json(ledger.output / "stop-media.json", {"stop": True})
            try:
                media.wait(timeout=180)
            except subprocess.TimeoutExpired:
                stop(media)
            if read_json(ledger.output / "media-status.json", {}).get("status") != "passed":
                ledger.result.update(
                    status="failed", recording_error="Execution video did not finalize successfully"
                )
        if track == "github" or native or ledger.result["status"] == "failed":
            for process in ledger.processes:
                stop(process)
            try:
                deployment.cleanup_cluster(ledger)
            except Exception as error:
                ledger.result.update(status="failed", cleanup_error=str(error))
        ledger.result["finished_at"] = datetime.now(timezone.utc).isoformat()
        if (ledger.output / "ci/summary.json").exists():
            try:
                collect_coverage(ledger)
            except Exception as error:
                ledger.result["coverage_error"] = str(error)
                if ledger.result["status"] == "passed":
                    ledger.result.update(status="failed", complete_acceptance=False)
        ledger.result["complete_acceptance"] = ledger.result["status"] == "passed" and not native
        ledger.save()
        # Refresh report to reflect recorder, cleanup, and report-generation outcomes.
        try:
            reporting.generate(ledger.output)
        except Exception as error:
            ledger.result.update(status="failed", reporting_error=str(error))
            ledger.save()
        archive = "Not created; inspect the report folder"
        try:
            inventory(ledger.output)
            archive = archive_evidence(ledger.output)
        except Exception as error:
            ledger.result.update(
                status="failed", complete_acceptance=False, packaging_error=str(error)
            )
            ledger.save()
            try:
                reporting.generate(ledger.output)
            except Exception:
                pass  # Original logs/results remain available even if reporting also fails.
        print(
            f"\nRESULT: {ledger.result['status'].upper()}\nREPORT: {ledger.output / 'START_HERE.html'}\nEVIDENCE: {archive}",
            flush=True,
        )
        write_json(
            PROJECT / "reports/acceptance/latest.json",
            {"directory": str(ledger.output), "track": track, "status": ledger.result["status"]},
        )
    return ledger.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["windows", "github"], required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/acceptance.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-validation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output or PROJECT / "reports/acceptance" / name
    result = run(output, args.track, read_json(args.config), native=args.native_validation)
    raise SystemExit(0 if result["status"] in {"passed", "partial"} else 1)


if __name__ == "__main__":
    main()
