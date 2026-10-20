"""Start the installed API in an empty workspace, verify real HTTP, and stop it."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __package__:
    from .smoke_api import verify_service, wait_ready
else:
    from smoke_api import verify_service, wait_ready


def verify_request_logs(path, request_id):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # Uvicorn's lifecycle messages are separate from application JSON logs.
        if isinstance(item, dict) and item.get("event") == "http_request":
            events.append(item)
    matching = [event for event in events if event["request_id"] == request_id]
    if len(matching) != 1 or matching[0]["route"] != "/predict" or matching[0]["status"] != 200:
        raise ValueError("Request ID did not correlate with a successful prediction log")
    allowed = {
        "event",
        "timestamp",
        "level",
        "request_id",
        "method",
        "route",
        "status",
        "duration_ms",
        "model_run_id",
        "error_type",
    }
    for event in events:
        if set(event) - allowed or event["duration_ms"] < 0:
            raise ValueError("Request log contains unexpected fields or invalid duration")
        if datetime.fromisoformat(event["timestamp"]).utcoffset().total_seconds() != 0:
            raise ValueError("Request log timestamp is not UTC")
    if "monitoring-validation-secret" in json.dumps(events):
        raise ValueError("Private query string leaked into request logs")
    if not any(event["status"] == 422 and event["level"] == "WARNING" for event in events):
        raise ValueError("Rejected request warning log was not found")
    return len(events)


def verify_installed_api(python, bundle, output):
    python, bundle, output = Path(python).resolve(), Path(bundle).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    empty = output / "empty-workspace"
    empty.mkdir()
    env = {**os.environ, "HEART_DISEASE_PROJECT_ROOT": str(empty), "PYTHONUTF8": "1"}
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    result, server = {"status": "running"}, None
    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    str(python),
                    "-I",
                    "-m",
                    "heart_disease.api",
                    "--port",
                    str(port),
                    "--model-dir",
                    str(bundle / "model"),
                ],
                cwd=empty,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            base_url = f"http://127.0.0.1:{port}"
            wait_ready(base_url, process=server)
            result.update(verify_service(base_url, bundle))
            result["request_log_events_verified"] = verify_request_logs(
                output / "server.log", result["prediction_request_id"]
            )
            result["structured_request_logging_verified"] = True
        if list(empty.iterdir()):
            raise ValueError("The inference data workspace was not kept empty")
        result.update(installed_python=str(python), empty_data_workspace_verified=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            result["server_stopped"] = server.poll() is not None
        (output / "validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_installed_api(args.python, args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
