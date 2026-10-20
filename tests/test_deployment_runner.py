"""Exercise deployment failure gates and diagnostic retention without a cluster."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import deployment_runner as runner


@pytest.fixture
def evidence(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    return runner.Evidence(project / "reports/deployment/test-run", project=project)


def test_command_failure_retains_log_exit_status_and_summary(evidence):
    with pytest.raises(RuntimeError, match="exit 7"):
        evidence.run(
            "failure", [sys.executable, "-c", "print('diagnostic detail'); raise SystemExit(7)"]
        )
    summary = runner.read_json(evidence.output / "summary.json")
    assert summary["steps"][0]["returncode"] == 7
    assert summary["steps"][0]["status"] == "failed"
    assert "diagnostic detail" in (evidence.output / summary["steps"][0]["log"]).read_text()


def test_command_timeout_stops_child_and_retains_failure(evidence):
    with pytest.raises(TimeoutError):
        evidence.run("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)
    assert evidence.steps[0]["status"] == "failed"
    assert "TimeoutError" in evidence.steps[0]["error"]


def test_optional_diagnostic_failure_is_recorded_without_replacing_run_outcome(evidence):
    evidence.summary["status"] = "failed"
    evidence.run("optional", [sys.executable, "-c", "raise SystemExit(8)"], required=False)
    assert evidence.summary["status"] == "failed"
    assert evidence.steps[0]["returncode"] == 8


def test_output_cannot_escape_report_directory(tmp_path):
    with pytest.raises(ValueError, match="reports/deployment"):
        runner.Evidence(tmp_path / "outside", project=tmp_path / "project")


def test_existing_run_evidence_is_not_overwritten(evidence):
    original = (evidence.output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="not be overwritten"):
        runner.Evidence(evidence.output, project=evidence.project)
    assert (evidence.output / "summary.json").read_bytes() == original


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_rendered_image_is_unique_and_bound_to_approved_bundle(line_ending):
    digest = "a" * 64
    template = (
        f"  image: heart-disease:bundle-{digest}{line_ending}  imagePullPolicy: Never{line_ending}"
    )
    image, rendered = runner.image_for_bundle(template, digest, "123abc")
    assert image == f"heart-disease:bundle-{digest}-r123abc"
    assert "imagePullPolicy: Never" in rendered
    assert rendered.count(image) == 1
    assert "-r" not in template


@pytest.mark.parametrize("template", ["image: other:latest\n", "image: x\nimage: y\n", ""])
def test_stale_or_ambiguous_image_is_rejected(template):
    with pytest.raises(ValueError, match="approved bundle"):
        runner.image_for_bundle(template, "a" * 64, "abc")


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"ip": "127.0.0.1"}, "http://127.0.0.1:8080"),
        ({"ip": "::1"}, "http://[::1]:8080"),
        ({"hostname": "demo.local"}, "http://demo.local:8080"),
    ],
)
def test_loadbalancer_address_uses_reported_host_and_service_port(entry, expected):
    assert runner.loadbalancer_url({"status": {"loadBalancer": {"ingress": [entry]}}}) == expected


def test_pending_or_invalid_loadbalancer_address():
    assert runner.loadbalancer_url({}) is None
    with pytest.raises(ValueError):
        runner.loadbalancer_url(
            {"status": {"loadBalancer": {"ingress": [{"hostname": "bad/path"}]}}}
        )


def test_missing_tools_produce_failed_zip_without_claiming_execution(evidence, monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    assert not runner.run_deployment(evidence, open_browser=False)
    summary = runner.read_json(evidence.output / "summary.json")
    assert summary["status"] == "failed"
    assert not summary["docker_verified"] and not summary["kubernetes_verified"]
    assert "Missing docker" in summary["error"]
    with zipfile.ZipFile(evidence.output.with_name("test-run-diagnostics.zip")) as archive:
        assert {"summary.json", "RESULTS.txt"} <= set(archive.namelist())
        assert json.loads(archive.read("summary.json"))["status"] == "failed"


class FakeTunnel:
    pid = 1234

    def poll(self):
        return None


def fake_workflow(evidence, monkeypatch, *, fail=None, wrong_pod=False):
    """Unit-test simulation only: no Docker/cluster is invoked or certified."""
    context = evidence.project / "context"
    (context / "bundle/fixtures").mkdir(parents=True)
    runner.write_json(context / "context.json", {"identity": {"bundle_sha256": "a" * 64}})
    runner.write_json(context / "bundle/bundle_manifest.json", {})
    runner.write_json(
        context / "bundle/fixtures/expected_response.json", {"model_run_id": "test-model"}
    )
    manifests = evidence.project / "deployment/kubernetes"
    manifests.mkdir(parents=True)
    (manifests / "deployment.yaml").write_text("image: heart-disease:bundle-" + "a" * 64 + "\n")
    for name in ["namespace.yaml", "service.yaml"]:
        (manifests / name).write_text("test fixture\n")
    names = []

    def command(name, arguments, **kwargs):
        names.append(name)
        evidence.summary["current_stage"] = name
        if name == fail:
            raise RuntimeError("simulated " + name)
        if name == "docker-engine":
            return "linux\n"
        if name == "approved-bundle":
            return json.dumps({"context": str(context)})
        if name == "cluster-versions":
            return json.dumps({"clientVersion": {"minor": "35"}, "serverVersion": {"minor": "35"}})
        if name == "docker-build-and-test":
            (evidence.output / "docker").mkdir()
            runner.write_json(
                evidence.output / "docker/validation.json",
                {"status": "passed", "image_id": "test-config-id"},
            )
        if name == "deployed-http":
            runner.write_json(
                evidence.output / "http-validation.json",
                {
                    "status": "passed",
                    "prediction_cases_verified": 7,
                    "model_run_id": "test-model",
                    "prediction_request_id": "test-id",
                },
            )
        if name == "verified-api-logs":
            return "\n".join(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": "test-id" if status == 200 else "bad-id",
                        "route": "/predict",
                        "status": status,
                        "duration_ms": 1,
                        "timestamp": "2026-10-17T00:00:00+00:00",
                        "level": level,
                    }
                )
                for status, level in [(200, "INFO"), (422, "WARNING")]
            )
        if name == "verified-pods":
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {},
                            "spec": {
                                "containers": [
                                    {"image": "wrong" if wrong_pod else evidence.summary["image"]}
                                ]
                            },
                            "status": {
                                "containerStatuses": [
                                    {"ready": True, "imageID": "test-runtime-digest"}
                                ]
                            },
                        }
                    ]
                }
            )
        return ""

    monkeypatch.setattr(evidence, "run", command)
    monkeypatch.setattr(runner.shutil, "which", lambda name: name)
    monkeypatch.setattr(runner, "start_tunnel", lambda output: FakeTunnel())
    monkeypatch.setattr(runner, "wait_for_exposure", lambda *args: "http://127.0.0.1:8080")
    monkeypatch.setattr(runner, "collect_diagnostics", lambda *args: None)
    stopped = []
    monkeypatch.setattr(runner, "stop_owned_process", lambda process: stopped.append(process.pid))
    return names, stopped


@pytest.mark.parametrize(
    "stage,blocked_stage",
    [
        ("docker-build-and-test", "cluster-start"),
        ("server-validation", "apply"),
        ("rollout", "deployed-http"),
    ],
)
def test_failure_gates_prevent_later_actions_and_keep_evidence(
    evidence, monkeypatch, stage, blocked_stage
):
    names, _ = fake_workflow(evidence, monkeypatch, fail=stage)
    assert not runner.run_deployment(evidence, open_browser=False)
    assert blocked_stage not in names
    assert evidence.summary["failed_stage"] == stage
    assert not evidence.summary["kubernetes_verified"]
    assert evidence.output.with_name("test-run-diagnostics.zip").is_file()


def test_simulated_complete_workflow_requires_http_logs_and_pod_identity(evidence, monkeypatch):
    names, stopped = fake_workflow(evidence, monkeypatch)
    assert runner.run_deployment(evidence, open_browser=False)
    assert evidence.summary["kubernetes_verified"]
    assert {"deployed-http", "verified-api-logs", "verified-pods"} <= set(names)
    assert evidence.summary["deployed_image_id"] == "test-runtime-digest"
    assert not stopped  # Tunnel remains available for the user's demo.


def test_wrong_deployed_image_cannot_pass_and_only_owned_tunnel_is_stopped(evidence, monkeypatch):
    _, stopped = fake_workflow(evidence, monkeypatch, wrong_pod=True)
    assert not runner.run_deployment(evidence, open_browser=False)
    assert "different image" in evidence.summary["error"]
    assert not evidence.summary["kubernetes_verified"]
    assert stopped == [1234]


def test_collector_preserves_failed_deployment_status_and_does_not_deploy(evidence, monkeypatch):
    previous = evidence.project / "reports/deployment/previous"
    previous.mkdir()
    runner.write_json(previous / "summary.json", {"status": "failed", "docker_verified": False})
    (previous / "build.log").write_text("original failure evidence")
    runner.write_json(
        previous.parent / "latest.json", {"directory": str(previous.relative_to(evidence.project))}
    )
    monkeypatch.setattr(runner, "collect_diagnostics", lambda evidence: None)
    runner.collect_latest(evidence)
    assert evidence.summary["status"] == "collected"
    assert evidence.summary["deployment_status"] == "failed"
    assert (evidence.output / "original-run/build.log").read_text() == "original failure evidence"
    assert not evidence.steps


def test_collector_rejects_pointer_outside_reports(evidence):
    runner.write_json(
        evidence.project / "reports/deployment/latest.json", {"directory": "../../outside"}
    )
    with pytest.raises(ValueError, match="previous deployment"):
        runner.collect_latest(evidence)


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher contract")
def test_powershell_bootstrap_forwards_native_arguments_without_losing_array_items(tmp_path):
    bootstrap = Path(__file__).resolve().parents[1] / "scripts/run_deployment.ps1"
    script = tmp_path / "argument test.ps1"
    # Parse the function without executing setup, installing tools or launching UI.
    script.write_text(
        "$tokens=$null; $parseErrors=$null\n"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{bootstrap}',[ref]$tokens,[ref]$parseErrors)\n"
        "if($parseErrors.Count){throw 'parse failure'}\n"
        "$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-CheckedNative'},$true)\n"
        "Invoke-Expression $function.Extent.Text\n"
        f"Invoke-CheckedNative -Executable '{sys.executable}' -NativeArguments @('-c','import sys,json; print(json.dumps(sys.argv[1:]))','argument with spaces','last argument')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == ["argument with spaces", "last argument"]


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing contract")
@pytest.mark.parametrize("exclusive_lock", [False, True])
def test_powershell_archive_snapshots_live_log_and_preserves_previous_zip_on_failure(
    tmp_path, exclusive_lock
):
    bootstrap = Path(__file__).resolve().parents[1] / "scripts/run_deployment.ps1"
    source = tmp_path / "run with spaces"
    source.mkdir()
    (source / "RESULTS.txt").write_text("PASSED", encoding="utf-8")
    (source / "commands").mkdir()
    (source / "commands/verified-pods.log").write_text("ready", encoding="utf-8")
    archive = tmp_path / "diagnostics.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("previous.txt", "retain if snapshot fails")
    previous = archive.read_bytes()
    script = tmp_path / "archive test.ps1"

    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"

    sharing = "None" if exclusive_lock else "Read"
    script.write_text(
        "$ErrorActionPreference='Stop'\n"
        "$tokens=$null; $parseErrors=$null\n"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile({literal(bootstrap)},[ref]$tokens,[ref]$parseErrors)\n"
        "if($parseErrors.Count){throw 'parse failure'}\n"
        "$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Write-DiagnosticArchive'},$true)\n"
        "Invoke-Expression $function.Extent.Text\n"
        f"$writer=[IO.File]::Open({literal(source / 'tunnel.log')},[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::{sharing})\n"
        "try {\n"
        "  $bytes=[Text.Encoding]::UTF8.GetBytes('tunnel is running')\n"
        "  $writer.Write($bytes,0,$bytes.Length); $writer.Flush()\n"
        "  $snapshotFailed=$false\n"
        "  try {\n"
        f"    Write-DiagnosticArchive -SourceDirectory {literal(source)} -DestinationPath {literal(archive)}\n"
        f"  }} catch {{ if(-not ${str(exclusive_lock).lower()}){{throw}}; $snapshotFailed=$true }}\n"
        f"  if($snapshotFailed -ne ${str(exclusive_lock).lower()}){{throw 'unexpected snapshot outcome'}}\n"
        "  $bytes=[Text.Encoding]::UTF8.GetBytes(' still running')\n"
        "  $writer.Write($bytes,0,$bytes.Length); $writer.Flush()\n"
        "} finally { $writer.Dispose() }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (source / "tunnel.log").read_text() == "tunnel is running still running"
    assert not list(tmp_path.glob("*.partial"))
    if exclusive_lock:
        assert archive.read_bytes() == previous
    else:
        with zipfile.ZipFile(archive) as output:
            assert output.testzip() is None
            assert output.read("tunnel.log") == b"tunnel is running"
            assert output.read("RESULTS.txt") == b"PASSED"
            assert output.read("commands/verified-pods.log") == b"ready"
            assert "previous.txt" not in output.namelist()
