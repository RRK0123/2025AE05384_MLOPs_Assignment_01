"""Collect live diagnostics or stop only the latest acceptance run's resources."""

import argparse
from pathlib import Path

from scripts.acceptance.evidence import archive_evidence, read_json, write_json

PROJECT = Path(__file__).resolve().parents[2]


def latest():
    launch = read_json(PROJECT / "reports/acceptance/latest-launch.json")
    pointer = read_json(PROJECT / "reports/acceptance/latest.json")
    if launch:
        target = Path(launch["directory"])
        pointer = {
            "directory": str(target if (target / "results.json").exists() else launch["bootstrap"])
        }
    if not pointer:
        raise FileNotFoundError(
            "No completed acceptance run; inspect reports/acceptance/bootstrap-* instead"
        )
    output = Path(pointer["directory"]).resolve()
    if (
        not output.is_relative_to(PROJECT / "reports/acceptance")
        or output == PROJECT / "reports/acceptance"
    ):
        raise ValueError("Invalid latest-run path")
    return output


def manage(action):
    import subprocess
    from datetime import datetime, timezone

    output = latest()
    result = read_json(output / "results.json", {"track": "bootstrap"})
    services = read_json(output / "services.json", {})
    context = result.get("cluster", {}).get("context")
    if action == "collect":
        if context:
            for name, arguments in [
                ("resources", ["get", "deployment,pods,service", "-o", "json"]),
                ("api-logs", ["logs", "deployment/heart-api", "--tail=3000"]),
            ]:
                response = subprocess.run(
                    [
                        "kubectl",
                        "--context",
                        context,
                        "--request-timeout=20s",
                        "-n",
                        "heart-disease",
                        *arguments,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                destination = (
                    output
                    / "collected"
                    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + name + ".log")
                )
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(response.stdout + response.stderr, encoding="utf-8")
        from scripts.acceptance.evidence import inventory

        inventory(output)
        print(archive_evidence(output))
    else:
        # Verify process command lines before killing recorded PIDs: a recycled
        # PID must never terminate unrelated work on the borrowed computer.
        import psutil

        for item in services.get("processes", []):
            try:
                process = psutil.Process(item["pid"])
                command = " ".join(process.cmdline())
                workspace = services.get("workspace", "")
                owned = (
                    str(output) in command
                    or bool(workspace)
                    and workspace.replace("\\", "/") in command.replace("\\", "/")
                )
                if owned:
                    for child in process.children(recursive=True):
                        child.terminate()
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
        cluster = result.get("cluster", {})
        if result["track"] == "windows" and cluster.get("owned_by_run") and cluster.get("created"):
            subprocess.run(["minikube", "stop", "-p", cluster["name"]], check=True, timeout=180)
        write_json(
            output / "stopped.json",
            {"stopped_at": datetime.now(timezone.utc).isoformat(), "cluster": cluster.get("name")},
        )
        print("Stopped owned demo services. Saved evidence was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "stop"])
    manage(parser.parse_args().action)


if __name__ == "__main__":
    main()
