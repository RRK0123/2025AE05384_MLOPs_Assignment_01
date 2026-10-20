"""Deploy only into this run's explicitly named Minikube/kind cluster."""

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import yaml

from scripts.acceptance.evidence import write_json
from scripts.smoke_api import request_json, wait_ready

NAMESPACE = "heart-disease"


def free_port():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


def deploy(ledger, project, image, model_run_id):
    track = ledger.result["track"]
    cluster = "heart-" + ledger.result["run_id"][-8:].lower()
    context = cluster if track == "windows" else "kind-" + cluster
    ledger.result["cluster"] = {
        "name": cluster,
        "context": context,
        "owned_by_run": True,
        "created": False,
        "creation_attempted": True,
    }
    ledger.save()
    if track == "windows":
        ledger.command(
            "cluster-start",
            ["minikube", "-p", cluster, "start", "--driver=docker", "--cpus=2", "--memory=4096"],
            timeout=1200,
        )
        ledger.result["cluster"]["created"] = True
        ledger.command(
            "cluster-load", ["minikube", "-p", cluster, "image", "load", image], timeout=300
        )
    else:
        ledger.command(
            "cluster-start",
            ["kind", "create", "cluster", "--name", cluster, "--wait", "180s"],
            timeout=600,
        )
        ledger.result["cluster"]["created"] = True
        ledger.command(
            "cluster-load", ["kind", "load", "docker-image", image, "--name", cluster], timeout=300
        )
        provider = shutil.which("cloud-provider-kind")
        if not provider:
            raise FileNotFoundError(
                "cloud-provider-kind is required to verify real LoadBalancer exposure"
            )
        ledger.background("loadbalancer-provider", [provider])
    manifest_dir = ledger.output / "deployment"
    manifest_dir.mkdir(exist_ok=True)
    manifest_args = []
    for name in ["namespace", "deployment", "service"]:
        value = yaml.safe_load((Path(project) / f"deployment/kubernetes/{name}.yaml").read_text())
        if name == "deployment":
            value["spec"]["template"]["spec"]["containers"][0]["image"] = image
        manifest_path = manifest_dir / f"{name}.yaml"
        manifest_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        manifest_args.extend(["-f", manifest_path])

    def kubectl(*args):
        return ["kubectl", "--context", context, "--request-timeout=30s", *args]

    versions = json.loads(
        ledger.command("cluster-versions", kubectl("version", "-o", "json"))["output"]
    )
    client = versions["clientVersion"]
    server = versions["serverVersion"]
    if (
        client["major"] != server["major"]
        or abs(int(client["minor"].rstrip("+")) - int(server["minor"].rstrip("+"))) > 1
    ):
        raise ValueError("kubectl and Kubernetes server must be within one minor version")
    write_json(manifest_dir / "versions.json", versions)

    ledger.command("namespace", kubectl("apply", "-f", manifest_dir / "namespace.yaml"))
    ledger.command(
        "server-validation",
        kubectl("apply", "--dry-run=server", "--validate=strict", *manifest_args),
    )
    ledger.command("apply", kubectl("apply", *manifest_args))
    ledger.command(
        "rollout",
        [
            "kubectl",
            "--context",
            context,
            "--request-timeout=330s",
            "-n",
            NAMESPACE,
            "rollout",
            "status",
            "deployment/heart-api",
            "--timeout=300s",
        ],
        timeout=350,
    )
    if track == "windows":
        tunnel_script = manifest_dir / "tunnel.ps1"
        # The console is visible because the user needs it for the live demo.
        tunnel_script.write_text(
            f"$Host.UI.RawUI.WindowTitle='Acceptance tunnel - keep open during demo'\n"
            f"& '{shutil.which('minikube')}' -p '{cluster}' tunnel --bind-address=127.0.0.1 2>&1 | "
            f"Tee-Object -FilePath '{str(manifest_dir / 'tunnel.log').replace(chr(39), chr(39) * 2)}'\n",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(tunnel_script),
            ],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        ledger.processes.append(process)
    deadline, base_url = time.monotonic() + 180, None
    while time.monotonic() < deadline:
        service_result = subprocess.run(
            kubectl("-n", NAMESPACE, "get", "service", "heart-api", "-o", "json"),
            capture_output=True,
            text=True,
            timeout=40,
        )
        if service_result.returncode == 0:
            service = json.loads(service_result.stdout)
            for entry in service.get("status", {}).get("loadBalancer", {}).get("ingress", []):
                host = entry.get("ip") or entry.get("hostname")
                if host:
                    candidate = f"http://{host}:8080"
                    try:
                        status, ready, _ = request_json(candidate + "/ready")
                        if status == 200 and ready.get("model_run_id") == model_run_id:
                            base_url = candidate
                            write_json(manifest_dir / "service.json", service)
                            break
                    except OSError:
                        pass
        if base_url:
            break
        time.sleep(2)
    if not base_url:
        raise TimeoutError("The LoadBalancer never served this run's ready model")
    pods = ledger.command(
        "ready-pods",
        kubectl(
            "-n", NAMESPACE, "get", "pods", "-l", "app.kubernetes.io/name=heart-api", "-o", "json"
        ),
    )
    items = json.loads(pods["output"])["items"]
    if not items or any(
        item["spec"]["containers"][0]["image"] != image
        or not item.get("status", {}).get("containerStatuses")
        or not all(c["ready"] and c.get("imageID") for c in item["status"]["containerStatuses"])
        for item in items
    ):
        raise ValueError("Deployed pod readiness/image identity was not verified")
    write_json(manifest_dir / "pods.json", items)
    ledger.result.update(base_url=base_url, deployed_image=image)
    ledger.save()
    wait_ready(base_url)
    return base_url


def diagnostics(ledger):
    cluster = ledger.result.get("cluster", {})
    if not cluster.get("created"):
        return
    for name, args in [
        ("resources", ["get", "deployment,pods,service,endpointslices", "-o", "json"]),
        ("events", ["get", "events", "--sort-by=.metadata.creationTimestamp"]),
        ("api-logs", ["logs", "deployment/heart-api", "--tail=3000"]),
        ("pod-details", ["describe", "pods", "-l", "app.kubernetes.io/name=heart-api"]),
    ]:
        ledger.command(
            name,
            [
                "kubectl",
                "--context",
                cluster["context"],
                "--request-timeout=20s",
                "-n",
                NAMESPACE,
                *args,
            ],
            timeout=30,
            required=False,
        )


def cleanup_cluster(ledger):
    cluster = ledger.result.get("cluster", {})
    if not cluster.get("creation_attempted"):
        return
    if ledger.result["track"] == "github":
        ledger.command(
            "cluster-delete", ["kind", "delete", "cluster", "--name", cluster["name"]], timeout=180
        )
        cluster["deleted_after_evidence"] = True
