"""Build, deploy and verify the local demo, retaining diagnostic evidence on failure."""

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import webbrowser
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
PROFILE = "heart-mlops"
NAMESPACE = "heart-disease"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def archive_reports(directory):
    directory = Path(directory).resolve()
    destination = directory.with_name(directory.name + "-diagnostics.zip")
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(directory))
    os.replace(temporary, destination)
    return destination


class Evidence:
    def __init__(self, output, *, project=PROJECT):
        self.project = Path(project).resolve()
        self.output = Path(output).resolve()
        allowed = self.project / "reports/deployment"
        if not self.output.is_relative_to(allowed) or self.output == allowed:
            raise ValueError("Output must be a run directory under reports/deployment")
        if (self.output / "summary.json").exists():
            raise FileExistsError(
                "Existing run evidence will not be overwritten; choose a new output"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "commands").mkdir(exist_ok=True)
        self.steps = []
        self.summary = {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": PROFILE,
            "namespace": NAMESPACE,
            "steps": self.steps,
            "docker_verified": False,
            "kubernetes_verified": False,
            "python": sys.version,
            "platform": platform.platform(),
        }
        self.save()

    def save(self):
        write_json(self.output / "summary.json", self.summary)

    def run(self, name, command, *, timeout=120, required=True):
        log = self.output / "commands" / f"{len(self.steps) + 1:03d}-{name}.log"
        step = {
            "name": name,
            "command": [str(p) for p in command],
            "status": "running",
            "required": required,
            "log": str(log.relative_to(self.output)),
        }
        self.steps.append(step)
        self.summary["current_stage"] = name
        self.save()
        print(f"[{name}] See {log}", flush=True)
        started, process = time.monotonic(), None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    [str(p) for p in command],
                    cwd=self.project,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                while True:
                    try:
                        code = process.wait(
                            timeout=min(10, max(0.01, timeout - (time.monotonic() - started)))
                        )
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started >= timeout:
                            raise TimeoutError(f"{name} exceeded {timeout} seconds") from None
                        print(
                            f"  {name} still running ({int(time.monotonic() - started)}s)",
                            flush=True,
                        )
            step.update(returncode=code, status="passed" if code == 0 else "failed")
            if code:
                raise RuntimeError(f"{name} failed (exit {code}); see {log.name}")
        except Exception as error:
            if process is not None and process.poll() is None:
                stop_owned_process(process)
            step.update(status="failed", error=f"{type(error).__name__}: {error}")
            if required:
                raise
        finally:
            step["seconds"] = round(time.monotonic() - started, 3)
            self.save()
        return log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""


def stop_owned_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def kubectl(*arguments, request_timeout="20s"):
    return ["kubectl", "--context", PROFILE, f"--request-timeout={request_timeout}", *arguments]


def image_for_bundle(template, digest, suffix):
    matches = re.findall(r"(?m)^\s*image:\s*(\S+)\s*$", template)
    expected = f"heart-disease:bundle-{digest}"
    if len(matches) != 1 or not re.fullmatch(
        re.escape(expected) + r"(-[a-z0-9][a-z0-9.-]*)?", matches[0]
    ):
        raise ValueError(
            "Manifest image does not match the approved bundle; restore matching files"
        )
    image = expected + "-r" + suffix
    rendered = re.sub(r"(?m)^(\s*image:)\s*\S+\s*$", lambda m: m[1] + " " + image, template)
    return image, rendered


def loadbalancer_url(service):
    for entry in service.get("status", {}).get("loadBalancer", {}).get("ingress", []):
        host = entry.get("ip") or entry.get("hostname")
        if not host:
            continue
        try:
            address = ipaddress.ip_address(host)
            host = f"[{host}]" if address.version == 6 else host
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
                raise ValueError("Invalid LoadBalancer address") from None
        return f"http://{host}:8080"
    return None


def start_tunnel(output):
    return subprocess.Popen(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT / "scripts/deployment_tunnel.ps1"),
            "-RunDirectory",
            str(output),
            "-Minikube",
            shutil.which("minikube"),
        ],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def collect_diagnostics(evidence):
    """Read only this application's resources; never collect kubeconfig or Secrets."""
    if shutil.which("docker"):
        evidence.run("diagnostic-docker-version", ["docker", "version"], timeout=20, required=False)
    if shutil.which("minikube"):
        evidence.run(
            "diagnostic-cluster-status",
            ["minikube", "-p", PROFILE, "status"],
            timeout=25,
            required=False,
        )
    if shutil.which("kubectl"):
        commands = [
            ("resources", ["get", "deployment,service,pods,endpointslices", "-o", "json"]),
            ("events", ["get", "events", "--sort-by=.metadata.creationTimestamp"]),
            ("pod-details", ["describe", "pods", "-l", "app.kubernetes.io/name=heart-api"]),
            ("api-logs", ["logs", "deployment/heart-api", "--tail=2000"]),
            ("previous-api-logs", ["logs", "deployment/heart-api", "--previous", "--tail=500"]),
        ]
        for name, args in commands:
            evidence.run(
                "diagnostic-" + name, kubectl("-n", NAMESPACE, *args), timeout=25, required=False
            )


def wait_for_exposure(evidence, tunnel, expected_run, timeout=150):
    deadline = time.monotonic() + timeout
    last_error = "No external IP was assigned"
    while time.monotonic() < deadline:
        if tunnel.poll() is not None:
            raise RuntimeError(
                "LoadBalancer tunnel exited; inspect tunnel.log and route permissions"
            )
        service = json.loads(
            evidence.run(
                "service-address",
                kubectl("-n", NAMESPACE, "get", "service", "heart-api", "-o", "json"),
                timeout=30,
            )
        )
        url = loadbalancer_url(service)
        if url:
            try:
                with urlopen(url + "/ready", timeout=3) as response:
                    ready = json.load(response)
                if ready.get("model_run_id") != expected_run:
                    raise ValueError("LoadBalancer reached a different model run")
                return url
            except (OSError, ValueError) as error:
                last_error = str(error)
        time.sleep(3)
    raise TimeoutError(f"LoadBalancer HTTP readiness timed out: {last_error}")


def run_deployment(evidence, *, open_browser=True):
    tunnel = None
    try:
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError("Python 3.13 is required")
        for tool in ["docker", "minikube", "kubectl"]:
            if not shutil.which(tool):
                raise FileNotFoundError(f"Missing {tool}; follow deployment/windows/START_HERE.md")
        engine = evidence.run(
            "docker-engine", ["docker", "info", "--format", "{{.OSType}}"], timeout=30
        )
        if engine.strip() != "linux":
            raise RuntimeError("Start Docker Desktop with a Linux engine")
        evidence.run("kubectl-version", ["kubectl", "version", "--client", "-o", "json"])
        evidence.run("minikube-version", ["minikube", "version"])
        prepared = json.loads(
            evidence.run("approved-bundle", [sys.executable, "scripts/prepare_docker.py"])
        )
        context = Path(prepared["context"]).resolve()
        digest = read_json(context / "context.json")["identity"]["bundle_sha256"]
        expected_run = read_json(context / "bundle/fixtures/expected_response.json")["model_run_id"]
        rendered = evidence.output / "manifests"
        rendered.mkdir()
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
        template = (evidence.project / "deployment/kubernetes/deployment.yaml").read_text(
            encoding="utf-8"
        )
        image, deployment = image_for_bundle(template, digest, suffix)
        (rendered / "deployment.yaml").write_text(deployment, encoding="utf-8")
        for name in ["namespace.yaml", "service.yaml"]:
            shutil.copyfile(evidence.project / "deployment/kubernetes" / name, rendered / name)
        evidence.summary.update(
            image=image,
            bundle_sha256=digest,
            model_run_id=expected_run,
            docker_context=str(context),
        )
        provenance = evidence.output / "provenance"
        provenance.mkdir()
        for source, name in [
            (context / "context.json", "context.json"),
            (context / "bundle/bundle_manifest.json", "bundle_manifest.json"),
            (context / "bundle/fixtures/expected_response.json", "expected_response.json"),
        ]:
            shutil.copyfile(source, provenance / name)
        sources = [
            *evidence.project.glob("requirements*.txt"),
            *evidence.project.glob("src/heart_disease/*.py"),
            *evidence.project.glob("scripts/*.py"),
            *evidence.project.glob("scripts/*.ps1"),
        ]
        write_json(
            provenance / "source-sha256.json",
            {
                path.relative_to(evidence.project).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sources
            },
        )
        evidence.save()
        evidence.run(
            "docker-build-and-test",
            [
                sys.executable,
                "scripts/docker_smoke.py",
                "--image",
                image,
                "--output",
                evidence.output / "docker",
            ],
            timeout=2400,
        )
        docker_result = read_json(evidence.output / "docker/validation.json")
        if docker_result["status"] != "passed":
            raise ValueError("Docker verification did not pass")
        evidence.summary.update(docker_verified=True, built_image_id=docker_result["image_id"])
        evidence.save()
        evidence.run(
            "cluster-start",
            ["minikube", "-p", PROFILE, "start", "--driver=docker", "--cpus=2", "--memory=4096"],
            timeout=900,
        )
        versions = json.loads(evidence.run("cluster-versions", kubectl("version", "-o", "json")))
        client_minor = int(re.match(r"\d+", versions["clientVersion"]["minor"])[0])
        server_minor = int(re.match(r"\d+", versions["serverVersion"]["minor"])[0])
        if abs(client_minor - server_minor) > 1:
            raise RuntimeError(
                "Install kubectl within one minor version of the cluster; see cluster-versions log"
            )
        evidence.run("image-load", ["minikube", "-p", PROFILE, "image", "load", image], timeout=600)
        evidence.run("namespace", kubectl("apply", "-f", rendered / "namespace.yaml"))
        apply_args = ["-f", rendered / "deployment.yaml", "-f", rendered / "service.yaml"]
        evidence.run(
            "server-validation",
            kubectl("apply", "--dry-run=server", "--validate=strict", *apply_args),
        )
        evidence.run("apply", kubectl("apply", *apply_args))
        evidence.run(
            "rollout",
            kubectl(
                "-n",
                NAMESPACE,
                "rollout",
                "status",
                "deployment/heart-api",
                "--timeout=300s",
                request_timeout="320s",
            ),
            timeout=330,
        )
        tunnel = start_tunnel(evidence.output)
        evidence.summary["tunnel_pid"] = tunnel.pid
        evidence.save()
        base_url = wait_for_exposure(evidence, tunnel, expected_run)
        evidence.summary["base_url"] = base_url
        evidence.run(
            "deployed-http",
            [
                sys.executable,
                "scripts/smoke_api.py",
                "--url",
                base_url,
                "--bundle",
                context / "bundle",
                "--output",
                evidence.output / "http-validation.json",
            ],
        )
        http = read_json(evidence.output / "http-validation.json")
        if (
            http["status"] != "passed"
            or http["prediction_cases_verified"] != 7
            or http["model_run_id"] != expected_run
        ):
            raise ValueError("Deployed HTTP evidence differs from the approved bundle")
        logs = evidence.run(
            "verified-api-logs",
            kubectl("-n", NAMESPACE, "logs", "deployment/heart-api", "--tail=2000"),
        )
        if __package__:
            from .verify_api import verify_request_logs
        else:
            from verify_api import verify_request_logs
        log_path = evidence.output / "api.log"
        log_path.write_text(logs, encoding="utf-8")
        log_count = verify_request_logs(log_path, http["prediction_request_id"])
        pods = json.loads(
            evidence.run(
                "verified-pods",
                kubectl(
                    "-n",
                    NAMESPACE,
                    "get",
                    "pods",
                    "-l",
                    "app.kubernetes.io/name=heart-api",
                    "-o",
                    "json",
                ),
            )
        )
        active = [pod for pod in pods["items"] if not pod["metadata"].get("deletionTimestamp")]
        if len(active) != 1:
            raise ValueError("Expected one active API pod after rollout")
        statuses = active[0]["status"].get("containerStatuses", [])
        if len(statuses) != 1 or not statuses[0]["ready"] or not statuses[0].get("imageID"):
            raise ValueError("Deployed pod is not ready or lacks image identity")
        if active[0]["spec"]["containers"][0]["image"] != image:
            raise ValueError("Deployed pod uses a different image")
        evidence.summary.update(
            status="passed",
            kubernetes_verified=True,
            deployed_image_id=statuses[0]["imageID"],
            request_log_events_verified=log_count,
            tunnel_left_running=True,
        )
        if open_browser:
            for path in ["/docs", "/monitoring"]:
                try:
                    webbrowser.open(base_url + path)
                except Exception:
                    pass  # Browser launch is optional; verified URLs remain in RESULTS.txt.
    except Exception as error:
        evidence.summary.update(
            status="failed",
            failed_stage=evidence.summary.get("current_stage", "preflight"),
            error=f"{type(error).__name__}: {error}",
        )
        if tunnel is not None:
            stop_owned_process(tunnel)
        evidence.summary["tunnel_left_running"] = False
    finally:
        outcome_stage = evidence.summary.get("current_stage")
        collect_diagnostics(evidence)
        evidence.summary["current_stage"] = outcome_stage
        evidence.summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        evidence.save()
        write_results(evidence)
        archive = archive_reports(evidence.output)
        print(
            f"\n{evidence.summary['status'].upper()}: {evidence.output / 'RESULTS.txt'}\nUpload: {archive}",
            flush=True,
        )
    return evidence.summary["status"] == "passed"


def write_results(evidence):
    summary = evidence.summary
    lines = [
        f"RESULT: {summary['status'].upper()}",
        f"Docker verified: {summary.get('docker_verified', False)}",
        f"Kubernetes verified: {summary.get('kubernetes_verified', False)}",
    ]
    if summary.get("error"):
        lines += [
            f"Failed stage: {summary.get('failed_stage')}",
            f"Reason: {summary['error']}",
            "Read deployment/windows/START_HERE.md for fixes, then rerun the main batch.",
        ]
    if summary.get("base_url"):
        lines += [
            "",
            "DEMO URLS (this PC, while the tunnel is open):",
            summary["base_url"] + "/docs",
            summary["base_url"] + "/monitoring",
            "Take screenshots of predictions, monitoring, and ready Kubernetes resources.",
        ]
    lines += [
        "",
        "Upload this run's diagnostics ZIP to ChatGPT for diagnosis.",
        "Double-click COLLECT_DEPLOYMENT_LOGS.bat to collect fresh state after your demo.",
        "Keep the tunnel window open while using the API; Ctrl+C in that window closes it.",
        "To stop this dedicated cluster after saving evidence: minikube stop -p heart-mlops",
        "This launcher never deletes the cluster or namespace automatically.",
    ]
    (evidence.output / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_latest(evidence):
    pointer = evidence.project / "reports/deployment/latest.json"
    previous = (evidence.project / read_json(pointer)["directory"]).resolve()
    allowed = evidence.project / "reports/deployment"
    if not previous.is_relative_to(allowed) or previous == allowed or previous == evidence.output:
        raise ValueError("Invalid previous deployment report path")
    original = read_json(previous / "summary.json")
    # Preserve the build/run evidence as well as collecting fresh live state.
    shutil.copytree(
        previous, evidence.output / "original-run", ignore=shutil.ignore_patterns("original-run")
    )
    evidence.summary.update(
        previous_run=str(previous.relative_to(evidence.project)),
        deployment_status=original.get("status"),
        status="collected",
    )
    collect_diagnostics(evidence)
    base_url = original.get("base_url")
    if base_url:
        try:
            with urlopen(base_url + "/metrics", timeout=5) as response:
                write_json(evidence.output / "current-metrics.json", json.load(response))
        except OSError as error:
            evidence.summary["metrics_collection_error"] = str(error)
    evidence.save()
    write_results(evidence)
    archive = archive_reports(evidence.output)
    print(f"Collected diagnostics: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    evidence = Evidence(args.output)
    try:
        if args.collect_only:
            collect_latest(evidence)
            return 0
        write_json(
            PROJECT / "reports/deployment/latest.json",
            {"directory": str(evidence.output.relative_to(PROJECT))},
        )
        return 0 if run_deployment(evidence, open_browser=not args.no_browser) else 1
    except Exception as error:
        evidence.summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        evidence.save()
        write_results(evidence)
        archive_reports(evidence.output)
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
